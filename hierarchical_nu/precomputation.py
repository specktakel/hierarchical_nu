"""
Module for the precomputation of quantities
to be passed to Stan models.
"""

from collections import defaultdict
from itertools import product
import logging
from typing import Union

import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp
import numpy as np

from .source.source import (
    Sources,
    PointSource,
    icrs_to_uv,
    BackgroundSource,
)
from .source.atmospheric_flux import AtmosphericNuMuFlux
from .source.flux_model import PGammaSpectrum
from .source.parameter import ParScale, Parameter
from .backend.stan_generator import StanGenerator
from .detector.detector_model import (
    LogNormEnergyResolution,
    GridInterpolationEnergyResolution,
)
from .detector.r2021_bg_llh import R2021BackgroundLLH
from .utils.roi import CircularROI, ROIList
from .detector.icecube import (
    EventType,
    CAS,
)
from .utils.fitting_tools import TopDownSegmentation

from tqdm.autonotebook import tqdm

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
        detector_model: EventType,
        n_grid_points: int = 50,
        show_progress: bool = False,
        bg_llh: Union[None, R2021BackgroundLLH] = None,
    ):
        """
        Handles calculation of the exposure integral.
        This is the convolution of the source spectrum and the
        effective area, multiplied by the observation time.

        :param sources: An instance of Sources.
        :param detector_model: An instance of EventType from the Refrigerator.
        :param n_grid_points: number of grid points for each parameter at which exposure is calculated.
        :param show_progress: set to True if progress bars should be displayed.
        :param bg_llh: If data-driven background likelihood is to be used, pass according instance of `R2021BackgroundLLH`, else `None`
        """

        self._show_progress = show_progress
        self._detector_model = detector_model
        self._sources = sources
        self._n_grid_points = n_grid_points
        self._bg_llh = bg_llh

        # Use Emin_det if available, otherwise use per event_type
        try:
            self._min_det_energy = Parameter.get_parameter("Emin_det").value

        except ValueError:
            self._min_det_energy = Parameter.get_parameter(
                f"Emin_det_{self._detector_model.P}"
            ).value

        # Silence log output
        logger = logging.getLogger("hierarchical_nu.backend.code_generator")
        logger.propagate = False

        # Instantiate the given Detector class to access values
        with StanGenerator():
            dm = detector_model.model()
            self._effective_area = dm.effective_area
            self._energy_resolution = dm.energy_resolution
            self._dm = dm

        self._parameter_source_map = defaultdict(list)
        self._source_parameter_map = defaultdict(list)
        self._original_param_values = defaultdict(list)

        for source in sources:
            c = 0
            for par in source.parameters.values():
                if not par.fixed:
                    self._parameter_source_map[par.name].append(source)
                    self._source_parameter_map[source].append(par.name)
                    self._original_param_values[par.name].append(par.value)
                    c += 1
                if c > 2:
                    raise NotImplementedError(
                        "Max two free parameters per source are currently implemented."
                    )

        self._par_grids = {}
        self._par_units = {}
        for par_name in list(self._parameter_source_map.keys()):
            par = Parameter.get_parameter(par_name)
            try:
                units = par.value.unit
                self._par_units[par_name] = units
                pmin, pmax = par.par_range
                par_range = (pmin.to_value(units), pmax.to_value(units))
                if not np.all(np.isfinite(par_range)):
                    raise ValueError(
                        "Parameter {} has non-finite bounds".format(par.name)
                    )
            except:
                par_range = par.par_range
                if not np.all(np.isfinite(par_range)):
                    raise ValueError(
                        "Parameter {} has non-finite bounds".format(par.name)
                    )

            if par.scale == ParScale.lin:
                grid = np.linspace(*par_range, num=self._n_grid_points)
            elif par.scale == ParScale.log:
                grid = np.logspace(*np.log10(par_range), num=self._n_grid_points)
            elif par.scale == ParScale.cos:
                grid = np.arccos(
                    np.linspace(*np.cos(par_range), num=self._n_grid_points)
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

    def calculate_rate(self, source, Ebins=None):
        # Emin determined as `Parameter` instance is accounted for
        # in the spectral shapes of the individual sources
        if Ebins is None:
            lower_e_edges = self.effective_area.tE_bin_edges[:-1].copy() << u.GeV
            upper_e_edges = self.effective_area.tE_bin_edges[1:].copy() << u.GeV

        else:
            lower_e_edges = Ebins[:-1] << u.GeV
            upper_e_edges = Ebins[1:] << u.GeV

        E_min = lower_e_edges[0]
        E_max = upper_e_edges[-1]

        log_E_space = np.linspace(np.log10(E_min.value), np.log10(E_max.value), 200)
        log_E_c = log_E_space[:-1] + np.diff(log_E_space) / 2
        E_c = np.power(10, log_E_c) * u.GeV
        d_E = np.diff(np.power(10, log_E_space)) * u.GeV

        if isinstance(source, PointSource):
            # For point sources the integral over the space angle is trivial

            # Assume that the source is inside the ROI
            # TODO add check at some point

            dec = source.dec.to(u.rad)
            cosz = -np.sin(dec.to_value(u.rad))  # ONLY FOR ICECUBE!

            flux_vals = source.flux_model.spectral_shape(E_c)
            if cosz < min(self.effective_area.cosz_bin_edges) or cosz >= max(
                self.effective_area.cosz_bin_edges
            ):
                aeff_vals = np.zeros(E_c.shape) << (u.m**2)

            else:
                aeff_vals = self.effective_area.eff_area_spline(
                    np.vstack(
                        (np.log10(E_c.to_value(u.GeV)), np.full(E_c.shape, cosz))
                    ).T
                ) << (u.m**2)

            if isinstance(self.energy_resolution, GridInterpolationEnergyResolution):
                p_Edet = self.energy_resolution.prob_Edet_above_threshold(
                    E_c,
                    self._min_det_energy,
                    np.full(E_c.shape, dec.to_value(u.rad)) * u.rad,
                    use_interpolation=True,
                )

            elif isinstance(self.energy_resolution, LogNormEnergyResolution):
                p_Edet = self.energy_resolution.prob_Edet_above_threshold(
                    E_c,
                    self._min_det_energy,
                    np.full(E_c.shape, dec.to_value(u.rad)) * u.rad,
                    use_lognorm=True,
                )

            else:
                p_Edet = self.energy_resolution.prob_Edet_above_threshold(
                    E_c, self._min_det_energy
                )
            p_Edet = np.nan_to_num(p_Edet)

            output = np.sum(aeff_vals * flux_vals * p_Edet * d_E)

        else:
            try:
                roi = ROIList.STACK[0]
            except IndexError:
                raise ValueError("An ROI is needed at this point.")

            # Setup coordinate grids at which to evaulate effective area and flux
            NSIDE = 256
            NPIX = hp.nside2npix(NSIDE)
            # Surface element
            d_omega = 4 * np.pi / NPIX * u.sr

            ra, dec = hp.pix2ang(
                nside=NSIDE, ipix=list(range(NPIX)), nest=True, lonlat=True
            )

            coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            mask = []
            for roi in ROIList.STACK:
                # Create mask with pixels inside the ROI
                if isinstance(roi, CircularROI):
                    mask.append(
                        roi.center.separation(coords).deg * u.deg < roi.radius.to(u.deg)
                    )
                else:
                    RA_min = roi.RA_min.to_value(u.rad)
                    RA_max = roi.RA_max.to_value(u.rad)
                    if RA_min > RA_max:
                        mask.append(
                            (
                                (coords.ra.rad >= roi.RA_min.to_value(u.rad))
                                | (coords.ra.rad <= roi.RA_max.to_value(u.rad))
                            )
                            & (coords.dec.rad >= roi.DEC_min.to_value(u.rad))
                            & (coords.dec.rad <= roi.DEC_max.to_value(u.rad))
                        )
                    else:
                        mask.append(
                            (coords.ra.rad >= roi.RA_min.to_value(u.rad))
                            & (coords.ra.rad <= roi.RA_max.to_value(u.rad))
                            & (coords.dec.rad >= roi.DEC_min.to_value(u.rad))
                            & (coords.dec.rad <= roi.DEC_max.to_value(u.rad))
                        )

            # Assume that indexing is faster than calculating stuff
            # only calculate at the unique declinations and sort values afterwards
            # As `dec` contains all points we can weigh each unique declination according to its
            # frequency over all points on the sky

            idxs = np.logical_or.reduce(mask)
            pixels = coords[idxs]
            ra = pixels.ra.rad * u.rad
            dec = pixels.dec.rad * u.rad
            dec_c, counts = np.unique(dec, return_counts=True)
            cosz_c = -np.sin(dec_c.to_value(u.rad))

            # Create common grids
            log_E_grid, cosz_grid = np.meshgrid(log_E_c, cosz_c)
            E_grid, dec_grid = np.meshgrid(E_c, dec_c)

            if self._bg_llh is not None:
                # for each slice in sin(dec) calculate integral over energy
                output = 0
                llh = self._bg_llh
                for d, c in zip(dec_c, counts):
                    # For each declination
                    #       digitize sin(dec)
                    #       calculate pdf integral over detected energy range, i.e. Emin_det to infinity
                    #       multiply with pdf(sin(dec)) -> integral calculated
                    output += (
                        llh.energy_integral(
                            np.sin(d.to_value(u.rad)),
                            np.log10(self._min_det_energy.to_value(u.GeV)),
                            12.0,  # just some arbitrary but high enough value
                        )
                        * llh.prob_omega(np.sin(d.to_value(u.rad)))
                        * c
                        * d_omega.to_value(u.sr)
                    )

            else:
                # Evaluate effective area and flux
                aeff_vals = (
                    self._effective_area.eff_area_spline(
                        np.vstack((log_E_grid.flatten(), cosz_grid.flatten())).T
                    ).reshape(log_E_grid.shape)
                    * u.m**2
                )

                flux_vals = source.flux_model(E_grid, dec_grid, 0 * u.rad)

                if isinstance(
                    self.energy_resolution, GridInterpolationEnergyResolution
                ):
                    p_Edet = self.energy_resolution.prob_Edet_above_threshold(
                        E_grid.flatten(),
                        self._min_det_energy,
                        dec_grid.flatten(),
                        use_interpolation=True,
                    ).reshape(E_grid.shape)

                elif isinstance(self.energy_resolution, LogNormEnergyResolution):
                    p_Edet = self.energy_resolution.prob_Edet_above_threshold(
                        E_grid.flatten(),
                        self._min_det_energy,
                        dec_grid.flatten(),
                        use_lognorm=True,
                    ).reshape(E_grid.shape)

                else:
                    p_Edet = self.energy_resolution.prob_Edet_above_threshold(
                        E_c, self._min_det_energy
                    )
                    p_Edet = np.repeat(np.expand_dims(p_Edet, 0), d_omega.size, axis=0)

                # Inflate d_omega if needed
                weights = np.repeat(np.expand_dims(counts, 1), p_Edet.shape[1], axis=1)
                p_Edet = np.nan_to_num(p_Edet)

                output = np.sum(
                    np.sum(
                        aeff_vals * flux_vals * p_Edet * d_omega * weights,
                        axis=0,
                    )
                    * d_E,
                    axis=0,
                )

        return output

    def _compute_exposure_integral(self):
        """
        Loop over sources and calculate the exposure integral.
        """

        self._integral_grid = []

        self._integral_fixed_vals = []

        with tqdm(total=self._sources.N, disable=not self._show_progress) as pbar:
            for k, source in enumerate(self._sources.sources):
                pbar.set_description(f"Source {k}")

                if not self._source_parameter_map[source]:
                    if (
                        isinstance(source.flux_model, AtmosphericNuMuFlux)
                        and self._detector_model == CAS
                    ):
                        self._integral_fixed_vals.append(0.0 * u.m**2)
                    elif self._bg_llh is not None:
                        self._integral_fixed_vals.append(self.calculate_rate(source))
                    else:
                        self._integral_fixed_vals.append(
                            self.calculate_rate(source)
                            / source.flux_model.total_flux_int.to(1 / (u.m**2 * u.s))
                        )
                    pbar.update(1)
                    continue

                this_free_pars = self._source_parameter_map[source]
                this_par_grids = [
                    self._par_grids[par_name] for par_name in this_free_pars
                ]
                integral_grids_tmp = np.zeros(
                    [self._n_grid_points] * len(this_par_grids)
                ) << (u.m**2)

                with tqdm(
                    total=integral_grids_tmp.size, disable=not self._show_progress
                ) as pbar_parameter:
                    for i, grid_points in enumerate(product(*this_par_grids)):
                        pbar_parameter.set_description(f"Parameter value {i}")
                        indices = np.unravel_index(i, integral_grids_tmp.shape)

                        for par_name, par_value in zip(this_free_pars, grid_points):
                            par = Parameter.get_parameter(par_name)
                            try:
                                units = self._par_units[par_name]
                            except KeyError:
                                units = 1.0
                            par.value = par_value * units

                        # To make units compatible with Stan model parametrisation
                        integral_grids_tmp[indices] += self.calculate_rate(
                            source
                        ) / source.flux_model.total_flux_int.to(1 / (u.m**2 * u.s))
                        pbar_parameter.update(1)

                # Reset free parameters to original values
                for par_name in this_free_pars:
                    par = Parameter.get_parameter(par_name)
                    original_values = self._original_param_values[par_name]

                    if len(original_values) > 1:
                        par.value = original_values[k]
                    else:
                        par.value = original_values[0]
                pbar.update(1)
                self._integral_grid.append(integral_grids_tmp)

    def _slice_aeff_for_point_sources(self):
        """
        Return a slice of the effective area at each point source's declination
        """

        logelow = np.log10(self.effective_area.tE_bin_edges[:-1].copy())
        logehigh = np.log10(self._effective_area.tE_bin_edges[1:].copy())

        ec = np.power(10, (logelow + logehigh) / 2) << u.GeV

        self.pdet_grid = []
        self.pdet_grid.append(ec)
        for source in self._sources:
            if isinstance(source, PointSource):
                unit_vector = icrs_to_uv(source.dec.value, source.ra.value)
                cosz = np.cos(np.pi - np.arccos(unit_vector[2]))
                cosz_bin = np.digitize(cosz, self.effective_area.cosz_bin_edges) - 1

                aeff_slice = self.effective_area.eff_area[:, cosz_bin].copy() << u.m**2

                self.pdet_grid.append(aeff_slice)

    def _compute_c_values(self, inplace: bool = False):
        """
        For the rejection sampling implemented in the simulation, we need
        to find max(f/g) for the relevant energy range at different cosz,
        where:
        f, target function: aeff_factor * src_spectrum
        g, envelope function: bbpl envelope used for sampling

        :param inplace: set to True if only source components with variable spectral shape
            shall be re-calculated
        """

        cosz_bin_cens = (
            self.effective_area.cosz_bin_edges[:-1]
            + np.diff(self.effective_area.cosz_bin_edges) / 2
        )

        envelope_container = []

        for c, source in enumerate(self._sources.sources):
            if isinstance(source, BackgroundSource):
                continue
            # Energy bounds in flux model are already redshift-corrected
            # and live in the detector frame

            try:
                Emin = source.flux_model.spectral_shape._lower_energy.to_value(u.GeV)
                Emax = source.flux_model.spectral_shape._upper_energy.to_value(u.GeV)
            except AttributeError:
                Emin = source.flux_model._lower_energy.to_value(u.GeV)
                Emax = source.flux_model._upper_energy.to_value(u.GeV)

            E_range = np.geomspace(Emin, Emax, 1_000)
            if isinstance(source, PointSource):
                # Point source has one declination/cosz,
                # no loop over cosz necessary
                cosz = source.cosz
                aeff_values = self.effective_area.eff_area_spline(
                    np.vstack((np.log10(E_range), np.full(E_range.shape, cosz))).T
                )
                f_values = (
                    source.flux_model.spectral_shape.pdf(
                        E_range * u.GeV, Emin * u.GeV, Emax * u.GeV, apply_lim=False
                    )
                    * aeff_values
                )

                log_break = np.log10(E_range[f_values.argmax()]) + 1.6
                segments = TopDownSegmentation(f_values, E_range, log_break=log_break)
                segments.generate_segments()
                if inplace:
                    self._envelope_container[c] = segments
                else:
                    envelope_container.append(segments)

            else:
                # For diffuse sources, we need to find an envelope envelopping
                # the maximum values along energy, marginalising over the declination.
                # Calculate f-values on an energy x cosz grid, then take maximum
                # and create the envelope function.

                if inplace and isinstance(source.flux_model, AtmosphericNuMuFlux):
                    continue
                f_values_all = []

                for cosz in cosz_bin_cens:
                    aeff_values = self.effective_area.eff_area_spline(
                        np.vstack((np.log10(E_range), np.full(E_range.shape, cosz))).T,
                    )

                    dec = np.arcsin(-cosz)  # Only for IceCube

                    if isinstance(source.flux_model, AtmosphericNuMuFlux):
                        atmo_flux_integ_val = source.flux_model.total_flux_int.to_value(
                            1 / (u.m**2 * u.s)
                        )
                        f_values = (
                            source.flux_model(
                                E_range.copy() * u.GeV, dec * u.rad, 0 * u.rad
                            ).to_value(1 / (u.GeV * u.s * u.sr * u.m**2))
                            / atmo_flux_integ_val
                        ) * aeff_values
                    else:
                        f_values = (
                            source.flux_model.spectral_shape.pdf(
                                E_range * u.GeV,
                                Emin * u.GeV,
                                Emax * u.GeV,
                                apply_lim=False,
                            )
                            * aeff_values
                        )

                    f_values_all.append(f_values)
                # Make array
                # Take max along energy axis, and use result for creating the envelope
                f_values = np.array(f_values_all).max(axis=0)
                log_break = np.log10(E_range[f_values.argmax()]) + 1.6
                segments = TopDownSegmentation(f_values, E_range, log_break=log_break)
                segments.generate_segments()
                if inplace:
                    self._envelope_container[c] = segments
                else:
                    envelope_container.append(segments)

        if not inplace:
            self._envelope_container = envelope_container

    def __call__(self):
        """
        Compute the exposure integrals.
        """

        self._compute_exposure_integral()

        self._slice_aeff_for_point_sources()

        self._compute_c_values()


'''
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
'''
