"""
Module for the precomputation of quantities
to be passed to Stan models.
"""
from collections import defaultdict
from itertools import product

import astropy.units as u
import numpy as np

from .source.simple_source import SourceList, PointSource, DiffuseSource, icrs_to_uv
from .source.parameter import ParScale, Parameter
from .backend.stan_generator import StanGenerator


m_to_cm = 100  # cm


class ExposureIntegral:
    """
    Handles calculation of the exposure integral, eps_k.
    This is the convolution of the source spectrum and the
    effective area, multiplied by the observation time.
    """

    @u.quantity_input
    def __init__(
        self,
        source_list: SourceList,
        detector_model,
        min_det_energy: u.GeV,
        min_src_energy: u.GeV,
        max_src_energy: u.GeV,
        n_grid_points: int = 50,
    ):
        """
        Handles calculation of the exposure integral, eps_k.
        This is the convolution of the source spectrum and the
        effective area, multiplied by the observation time.

        :param source_list: An instance of SourceList.
        :param DetectorModel: An uninstantiated DetectorModel class.
        :param min_det_energy: The energy threshold of our detected sample in GeV.
        :param: min_src_energy: Minimum source energy in GeV.
        :param: max_src_energy: Maximum source energy in GeV.
        """

        self._source_list = source_list
        self._min_det_energy = min_det_energy
        self._min_src_energy = min_src_energy
        self._max_src_energy = max_src_energy
        self._n_grid_points = n_grid_points

        # Instantiate the given Detector class to access values
        with StanGenerator() as cg:
            dm = detector_model()
            self._effective_area = dm.effective_area
            self._energy_resolution = dm.energy_resolution

        self._parameter_source_map = defaultdict(list)
        self._source_parameter_map = defaultdict(list)
        self._original_param_values = defaultdict(list)

        for source in source_list:
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

    @property
    def source_list(self):
        return self._source_list

    @source_list.setter
    def source_list(self, value):
        if not isinstance(value, SourceList):
            raise ValueError(str(value) + " is not a recognised source list")

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

        lower_e_edges = self.effective_area._tE_bin_edges[:-1] << u.GeV
        upper_e_edges = self.effective_area._tE_bin_edges[1:] << u.GeV
        e_cen = (lower_e_edges + upper_e_edges) / 2

        if isinstance(source, PointSource):
            # For point sources the integral over the space angle is trivial

            dec = source.dec
            cosz = -np.sin(dec)  # ONLY FOR IC!

            integral = source.flux_model.spectral_shape.integral(
                lower_e_edges, upper_e_edges
            )

            if (
                cosz < self.effective_area._cosz_bin_edges[0]
                or cosz > self._effective_area._cosz_bin_edges[-1]
            ):
                aeff = np.zeros(len(lower_e_edges)) << (u.m ** 2)

            else:
                aeff = (
                    self.effective_area._eff_area[
                        :, np.digitize(cosz, self.effective_area._cosz_bin_edges) - 1
                    ]
                    * u.m ** 2
                )

        else:

            lower_cz_edges = self.effective_area._cosz_bin_edges[:-1]
            upper_cz_edges = self.effective_area._cosz_bin_edges[1:]

            # Switch upper and lower since zen -> dec induces a -1
            dec_lower = np.arccos(upper_cz_edges) * u.rad - np.pi / 2 * u.rad
            dec_upper = np.arccos(lower_cz_edges) * u.rad - np.pi / 2 * u.rad

            integral = source.flux_model.integral(
                lower_e_edges[:, np.newaxis],
                upper_e_edges[:, np.newaxis],
                dec_lower[np.newaxis, :],
                dec_upper[np.newaxis, :],
                0 * u.rad,
                2 * np.pi * u.rad,
            )

            aeff = np.array(self.effective_area._eff_area, copy=True) << (u.m ** 2)

        p_Edet = self.energy_resolution.prob_Edet_above_threshold(
            e_cen, self._min_det_energy
        )

        return ((p_Edet * integral.T * aeff.T * source.redshift_factor(z)).T).sum()

    def _compute_exposure_integral(self):
        """
        Loop over sources and calculate the exposure integral.
        """

        self._integral_grid = []

        self._integral_fixed_vals = []

        for k, source in enumerate(self.source_list.sources):
            if not self._source_parameter_map[source]:

                self._integral_fixed_vals.append(
                    self.calculate_rate(source)
                    / source.flux_model.total_flux_int.to(1 / (u.m ** 2 * u.s))
                )
                continue

            this_free_pars = self._source_parameter_map[source]
            this_par_grids = [self._par_grids[par_name] for par_name in this_free_pars]
            integral_grids_tmp = np.zeros(
                [self._n_grid_points] * len(this_par_grids)
            ) << (1 / u.s)

            for i, grid_points in enumerate(product(*this_par_grids)):

                indices = np.unravel_index(i, integral_grids_tmp.shape)

                for par_name, par_value in zip(this_free_pars, grid_points):
                    par = Parameter.get_parameter(par_name)
                    par.value = par_value

                integral_grids_tmp[indices] += self.calculate_rate(source)

            # To make units compatible with Stan model parametrisation
            self._integral_grid.append(
                integral_grids_tmp
                / source.flux_model.total_flux_int.to(1 / (u.m ** 2 * u.s))
            )

            # Reset free parameters to original values
            for par_name in this_free_pars:
                par = Parameter.get_parameter(par_name)
                par.value = self._original_param_values[par_name][k]

    def _compute_energy_detection_factor(self):
        """
        Loop over sources and calculate Aeff as a function of arrival energy.
        """

        self.energy_grid = (
            10
            ** np.linspace(
                np.log10(self._min_src_energy.value),
                np.log10(self._max_src_energy.value),
            )
            << u.GeV
        )
        self.pdet_grid = []

        for k, source in enumerate(self.source_list.sources):

            if isinstance(source, PointSource):

                unit_vector = icrs_to_uv(source.dec.value, source.ra.value)
                cosz = np.cos(np.pi - np.arccos(unit_vector[2]))
                cosz_bin = np.digitize(cosz, self.effective_area._cosz_bin_edges) - 1

                if cosz > 0.1:
                    pg = [0.0 for E in self.energy_grid]
                else:
                    pg = [
                        self.effective_area._eff_area[
                            np.digitize(E.value, self.effective_area._tE_bin_edges) - 1
                        ][cosz_bin]
                        for E in self.energy_grid
                    ]

                # TODO: Should this be scaled?
                self.pdet_grid.append(np.array(pg) / max(pg))

            if isinstance(source, DiffuseSource):

                aeff_vals = np.sum(self.effective_area._eff_area, axis=1)

                pg = [
                    aeff_vals[
                        np.digitize(E.value, self.effective_area._tE_bin_edges) - 1
                    ]
                    for E in self.energy_grid
                ]
                self.pdet_grid.append(np.array(pg) / max(pg))

    def __call__(self):
        """
        Compute the exposure integrals.
        """

        self._compute_exposure_integral()
        self._compute_energy_detection_factor()
