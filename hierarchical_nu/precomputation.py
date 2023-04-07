"""
Module for the precomputation of quantities
to be passed to Stan models.
"""
from collections import defaultdict
from itertools import product
import logging

import astropy.units as u
import numpy as np

from hierarchical_nu.source.source import (
    Sources,
    PointSource,
    DiffuseSource,
    icrs_to_uv,
)
from hierarchical_nu.source.atmospheric_flux import AtmosphericNuMuFlux
from hierarchical_nu.source.parameter import ParScale, Parameter
from hierarchical_nu.backend.stan_generator import StanGenerator
from hierarchical_nu.detector.r2021 import R2021EnergyResolution

m_to_cm = 100  # cm


class ExposureIntegral:
    """
    Handles calculation of the exposure integral.
    This is the convolution of the source spectrum and the
    effective area, multiplied by the observation time.
    """

    @u.quantity_input
    def __init__(
        self,
        sources: Sources,
        detector_model,
        n_grid_points: int = 50,
        event_type=None,
    ):
        """
        Handles calculation of the exposure integral.
        This is the convolution of the source spectrum and the
        effective area, multiplied by the observation time.

        :param source_list: An instance of SourceList.
        :param DetectorModel: A DetectorModel class.
        """

        self._sources = sources
        self._min_src_energy = Parameter.get_parameter("Emin_src").value
        self._max_src_energy = Parameter.get_parameter("Emax_src").value
        self._n_grid_points = n_grid_points

        # Use Emin_det if available, otherwise use per event_type
        try:

            self._min_det_energy = Parameter.get_parameter("Emin_det").value

        except ValueError:

            if event_type == "tracks":

                self._min_det_energy = Parameter.get_parameter("Emin_det_tracks").value

            elif event_type == "cascades":

                self._min_det_energy = Parameter.get_parameter(
                    "Emin_det_cascades"
                ).value

            else:

                raise ValueError("event_type not recognised")

        # Silence log output
        logger = logging.getLogger("hierarchical_nu.backend.code_generator")
        logger.propagate = False

        # Instantiate the given Detector class to access values
        with StanGenerator():
            dm = detector_model(event_type=event_type)
            self._effective_area = dm.effective_area
            self._energy_resolution = dm.energy_resolution

        self._parameter_source_map = defaultdict(list)
        self._source_parameter_map = defaultdict(list)
        self._original_param_values = defaultdict(list)

        for source in sources:
            for par in source.parameters.values():
                if not par.fixed:
                    self._parameter_source_map[par.name].append(source)
                    self._source_parameter_map[source].append(par.name)
                    self._original_param_values[par.name].append(par.value)

        self._par_grids = {}
        for par_name in list(self._parameter_source_map.keys()):
            par = Parameter.get_parameter(par_name)
            if not np.all(np.isfinite(par.par_range)):
                raise ValueError("Parameter {} has non-finite bounds".format(par.name))
            if par.scale == ParScale.lin:
                grid = np.linspace(*par.par_range, num=self._n_grid_points)
            elif par.scale == ParScale.log:
                grid = np.logspace(*np.log10(par.par_range), num=self._n_grid_points)
            elif par.scale == ParScale.cos:
                grid = np.arccos(
                    np.linspace(*np.cos(par.par_range), num=self._n_grid_points)
                )
            else:
                raise NotImplementedError(
                    "This scale ({}) is not yet supported".format(par.scale)
                )

            self._par_grids[par_name] = grid

        self.__call__()

    @property
    def effective_area(self):
        return self._effective_area

    @property
    def energy_resolution(self):
        return self._energy_resolution

    @property
    def par_grids(self):
        return self._par_grids

    @property
    def integral_grid(self):
        return self._integral_grid

    @property
    def integral_fixed_vals(self):
        return self._integral_fixed_vals

    def calculate_rate(self, source):

        z = source.redshift

        lower_e_edges = self.effective_area.tE_bin_edges[:-1] << u.GeV
        upper_e_edges = self.effective_area.tE_bin_edges[1:] << u.GeV
        e_cen = (lower_e_edges + upper_e_edges) / 2

        integral_unit = 1 / (u.m**2 * u.s)

        if isinstance(source, PointSource):
            # For point sources the integral over the space angle is trivial

            dec = source.dec
            cosz = -np.sin(dec)  # ONLY FOR ICECUBE!

            integral = source.flux_model.spectral_shape.integral(
                lower_e_edges, upper_e_edges
            ).to(integral_unit)
            if cosz < min(self.effective_area.cosz_bin_edges) or cosz >= max(
                self.effective_area.cosz_bin_edges
            ):

                aeff = np.zeros(len(lower_e_edges)) << (u.m**2)

            else:
                aeff = (
                    self.effective_area.eff_area[
                        :, np.digitize(cosz, self.effective_area.cosz_bin_edges) - 1
                    ]
                    * u.m**2
                )
            if isinstance(self.energy_resolution, R2021EnergyResolution):
                self.energy_resolution.set_fit_params(dec.value)

            p_Edet = self.energy_resolution.prob_Edet_above_threshold(
                e_cen, self._min_det_energy, dec
            )
            p_Edet = np.nan_to_num(p_Edet)

        else:

            lower_cz_edges = self.effective_area.cosz_bin_edges[:-1]
            upper_cz_edges = self.effective_area.cosz_bin_edges[1:]

            # Switch upper and lower since zen -> dec induces a -1
            dec_lower = np.arccos(upper_cz_edges) * u.rad - np.pi / 2 * u.rad
            dec_upper = np.arccos(lower_cz_edges) * u.rad - np.pi / 2 * u.rad
            # make union with irf bins, at one point I might actually do this
            # if isinstance(self.energy_resolution, R2021EnergyResolution):
            #    dec_lower = np.union1d(dec_lower, self.energy_resolution._declination_bins[:-1] * u.rad)
            #    dec_upper = np.union1d(dec_lower, self.energy_resolution._declination_bins[1:] * u.rad)

            integral = source.flux_model.integral(
                lower_e_edges[:, np.newaxis] * (1 + z),
                upper_e_edges[:, np.newaxis] * (1 + z),
                dec_lower[np.newaxis, :],
                dec_upper[np.newaxis, :],
                0 * u.rad,
                2 * np.pi * u.rad,
            ).to(integral_unit)

            aeff = np.array(self.effective_area.eff_area, copy=True) << (u.m**2)

            if isinstance(self.energy_resolution, R2021EnergyResolution):

                p_Edet = np.zeros((dec_upper.size, e_cen.size))
                for c, (dec_l, dec_h) in enumerate(zip(dec_lower, dec_upper)):
                    dec = (dec_l + dec_h) / 2
                    self.energy_resolution.set_fit_params(dec.value)
                    p_Edet[c] = self.energy_resolution.prob_Edet_above_threshold(
                        e_cen, self._min_det_energy, dec
                    )

            else:
                p_Edet = self.energy_resolution.prob_Edet_above_threshold(
                    e_cen, self._min_det_energy
                )
            p_Edet = np.nan_to_num(p_Edet)

        return ((p_Edet * integral.T * aeff.T).T).sum()

    def _compute_exposure_integral(self):
        """
        Loop over sources and calculate the exposure integral.
        """

        self._integral_grid = []

        self._integral_fixed_vals = []

        for k, source in enumerate(self._sources.sources):
            if not self._source_parameter_map[source]:

                self._integral_fixed_vals.append(
                    self.calculate_rate(source)
                    / source.flux_model.total_flux_int.to(1 / (u.m**2 * u.s))
                )
                continue

            this_free_pars = self._source_parameter_map[source]
            this_par_grids = [self._par_grids[par_name] for par_name in this_free_pars]
            integral_grids_tmp = np.zeros(
                [self._n_grid_points] * len(this_par_grids)
            ) << (u.m**2)

            for i, grid_points in enumerate(product(*this_par_grids)):

                indices = np.unravel_index(i, integral_grids_tmp.shape)

                for par_name, par_value in zip(this_free_pars, grid_points):
                    par = Parameter.get_parameter(par_name)
                    par.value = par_value

                # To make units compatible with Stan model parametrisation
                integral_grids_tmp[indices] += self.calculate_rate(
                    source
                ) / source.flux_model.total_flux_int.to(1 / (u.m**2 * u.s))
                # essentially my calculation, but from the other end
                # normalisation of integrated particle flux is taken out 
                # because it is used as a transformed parameter
                # remainder is only dependent on gamma and detector

            # Reset free parameters to original values
            for par_name in this_free_pars:
                par = Parameter.get_parameter(par_name)
                original_values = self._original_param_values[par_name]

                if len(original_values) > 1:
                    par.value = original_values[k]
                else:
                    par.value = original_values[0]

            self._integral_grid.append(integral_grids_tmp)

    def _compute_energy_detection_factor(self):
        """
        Loop over sources and calculate Aeff as a function of arrival energy.
        """

        epsilon = 1e-5
        Emin = min(self.effective_area.tE_bin_edges)
        Emax = max(self.effective_area.tE_bin_edges)
        self.energy_grid = (
            10
            ** np.linspace(
                np.log10(Emin), np.log10(Emax - epsilon), self._n_grid_points
            )
            << u.GeV
        )

        self.pdet_grid = []

        for source in self._sources.sources:

            if isinstance(source, PointSource):

                unit_vector = icrs_to_uv(source.dec.value, source.ra.value)
                cosz = np.cos(np.pi - np.arccos(unit_vector[2]))
                cosz_bin = np.digitize(cosz, self.effective_area.cosz_bin_edges) - 1

                # Set to zero if outside cosz range
                if (cosz < min(self.effective_area.cosz_bin_edges)) or (
                    cosz >= max(self.effective_area.cosz_bin_edges)
                ):

                    pg = np.zeros_like(self.energy_grid)

                else:

                    pg = [
                        self.effective_area.eff_area[
                            np.digitize(E.value, self.effective_area.tE_bin_edges) - 1
                        ][cosz_bin]
                        for E in self.energy_grid
                    ]
                    pg = np.array(pg)

            if isinstance(source, DiffuseSource):

                aeff_vals = np.sum(self.effective_area.eff_area, axis=1)

                pg = [
                    aeff_vals[
                        np.digitize(E.value, self.effective_area.tE_bin_edges) - 1
                    ]
                    for E in self.energy_grid
                ]
                pg = np.array(pg)

            self.pdet_grid.append(pg)

        self.pdet_grid = np.array(self.pdet_grid) + 1e-10  # avoid log(0)

    def _compute_c_values(self):
        """
        For the rejection sampling implemented in the simulation, we need
        to find max(f/g) for the relevant energy range at different cosz,
        where:
        f, target function: aeff_factor * src_spectrum
        g, envelope function: bbpl envelope used for sampling

        src_spectrum should be transformed to detector frame
        and have a cutoff at the transformed boundaries
        most restrictive, i.e. (detector) union (source in detector frame)
        TODO: see if neccessary to do this or if cuts on simulation are done in stan file
        """

        Emin_src = self._min_src_energy.to_value(u.GeV)
        Emax_src = self._max_src_energy.to_value(u.GeV)

        cosz_bin_cens = (
            self.effective_area.cosz_bin_edges[:-1]
            + np.diff(self.effective_area.cosz_bin_edges) / 2
        )

        Eth = self.effective_area.rs_bbpl_params["threshold_energy"]
        gamma1 = self.effective_area.rs_bbpl_params["gamma1"]
        gamma2_scale = self.effective_area.rs_bbpl_params["gamma2_scale"]

        c_values = []
        for source in self._sources.sources:

            # should be Emin, Emax of source at detector
            Emin = Emin_src / (1 + source.redshift)
            Emax = Emax_src / (1 + source.redshift)
            E_range = 10 ** np.linspace(np.log10(Emin), np.log10(Emax))

            f_values_all = []
            for cosz in cosz_bin_cens:

                idx_cosz = np.digitize(cosz, self.effective_area.cosz_bin_edges) - 1
                aeff_values = []
                for E in E_range:

                    idx_E = np.digitize(E, self.effective_area.tE_bin_edges) - 1
                    aeff_values.append(self.effective_area.eff_area[idx_E][idx_cosz])

                if isinstance(source.flux_model, AtmosphericNuMuFlux):

                    dec = np.arcsin(-cosz)  # Only for IceCube
                    atmo_flux_integ_val = source.flux_model.total_flux_int.to(
                        1 / (u.m**2 * u.s)
                    ).value
                    f_values = (
                        source.flux_model(
                            E_range * u.GeV, dec * u.rad, 0 * u.rad
                        ).to_value(1 / (u.GeV * u.s * u.sr * u.m**2))
                        / atmo_flux_integ_val
                    ) * aeff_values
                    gamma2 = gamma2_scale - 3.6

                else:

                    f_values = (
                        source.flux_model.spectral_shape.pdf(
                            E_range * u.GeV, Emin * u.GeV, Emax * u.GeV, apply_lim=False
                        )
                        * aeff_values
                    )
                    gamma2 = (
                        gamma2_scale
                        - source.flux_model.spectral_shape.parameters["index"].value
                    )

                f_values_all.append(f_values)

            if Emin < Eth and Emax > Eth:

                g_values = bbpl_pdf(
                    E_range,
                    Emin,
                    Eth,
                    Emax,
                    gamma1,
                    gamma2,
                )

            elif Emin < Eth and Emax <= Eth:

                Eth_tmp = Emin + (Emax - Emin) / 2

                g_values = bbpl_pdf(
                    E_range,
                    Emin,
                    Eth_tmp,
                    Emax,
                    gamma1,
                    gamma1,
                )

            elif Emin >= Eth and Emax > Eth:

                Eth_tmp = Emin + (Emax - Emin) / 2
                g_values = bbpl_pdf(
                    E_range,
                    Emin,
                    Eth_tmp,
                    Emax,
                    gamma2,
                    gamma2,
                )

            c_values_src = [max(f_values / g_values) for f_values in f_values_all]
            c_values.append(c_values_src)

        self.c_values = c_values

    def __call__(self):
        """
        Compute the exposure integrals.
        """

        self._compute_exposure_integral()

        self._compute_energy_detection_factor()

        self._compute_c_values()


def bbpl_pdf(x, x0, x1, x2, gamma1, gamma2):
    """
    Bounded broken power law PDF.
    Used as envelope in rejection sampling and
    therefore in _compute_c_values().
    """

    x = np.atleast_1d(x)

    output = np.empty_like(x)

    I1 = (x1 ** (gamma1 + 1.0) - x0 ** (gamma1 + 1.0)) / (gamma1 + 1.0)
    I2 = (
        x1 ** (gamma1 - gamma2)
        * (x2 ** (gamma2 + 1.0) - x1 ** (gamma2 + 1.0))
        / (gamma2 + 1.0)
    )

    N = 1.0 / (I1 + I2)

    mask1 = x <= x1
    mask2 = x > x1

    output[mask1] = N * x[mask1] ** gamma1

    output[mask2] = N * x1 ** (gamma1 - gamma2) * x[mask2] ** gamma2

    return output
