"""
Module for the precomputation of quantities
to be passed to Stan models.
"""
from collections import defaultdict
from itertools import product

import astropy.units as u
import numpy as np

from .source.simple_source import SourceList, PointSource, DiffuseSource
from .source.parameter import ParScale, Parameter
from .backend.stan_generator import StanGenerator



m_to_cm = 100  # cm


class ExposureIntegral:
    """
    Handles calculation of the exposure integral, eps_k.
    This is the convolution of the source spectrum and the
    effective area, multiplied by the observation time.
    """

    def __init__(
        self,
        source_list,
        effective_area,
        observation_time,
        minimum_energy,
        n_grid_points=50,
    ):
        """
        Handles calculation of the exposure integral, eps_k.
        This is the convolution of the source spectrum and the
        effective area, multiplied by the observation time.

        :param source_list: An instance of SourceList.
        :param effective_area: An uninstantiated EffectiveArea class.
        :param observation_time: Observation time in years.
        :param minimum_energy: The minimum energy to integrate over in GeV.
        """

        self._source_list = source_list
        self._observation_time = observation_time
        self._minimum_energy = minimum_energy
        self._n_grid_points = n_grid_points

        # Instantiate the given effective area class to access values
        with StanGenerator() as cg:
            self._effective_area = effective_area()


        self._parameter_source_map = defaultdict(list)
        self._source_parameter_map = defaultdict(list)

        for source in source_list:
            for par in source.parameters.values():
                if not par.fixed:
                    self._parameter_source_map[par.name].append(source)
                    self._source_parameter_map[source].append(par.name)

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
                grid = np.arccos(np.linspace(*np.cos(par.par_range), num=self._n_grid_points))
            else:
                raise NotImplementedError("This scale ({}) is not yet supported".format(par.scale))

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
    def observation_time(self):
        return self._observation_time

    @property
    def par_grids(self):
        return self._par_grids

    @property
    def integral_grid(self):
        return self._integral_grid

    def calculate_rate(self, source):
        z = source.redshift
        if isinstance(source, PointSource):
            # For point sources the integral over the space angle is trivial

            dec = source.dec
            cosz = -np.sin(dec)  # ONLY FOR IC!

            lower_edges = self.effective_area._tE_bin_edges[:-1] << u.GeV
            upper_edges = self.effective_area._tE_bin_edges[1:] << u.GeV

            integral = source.flux_model.spectral_shape.integral(lower_edges, upper_edges)

            if (
                cosz < self.effective_area._cosz_bin_edges[0] or
                cosz > self._effective_area._cosz_bin_edges[-1]
            ):
                aeff = np.zeros(len(lower_edges)) << (u.m**2)

            else:
                aeff = self.effective_area._eff_area[
                    :,
                    np.digitize(cosz, self.effective_area._cosz_bin_edges) - 1
                ] * u.m**2

            aeff[upper_edges < self._minimum_energy * u.GeV] = 0

        else:

            lower_e_edges = self.effective_area._tE_bin_edges[:-1] << u.GeV
            upper_e_edges = self.effective_area._tE_bin_edges[1:] << u.GeV

            lower_cz_edges = self.effective_area._cosz_bin_edges[:-1]
            upper_cz_edges = self.effective_area._cosz_bin_edges[1:]

            # Switch upper and lower since zen -> dec induces a -1
            dec_lower = np.arccos(upper_cz_edges) * u.rad - np.pi / 2 * u.rad
            dec_upper = np.arccos(lower_cz_edges) * u.rad - np.pi / 2 * u.rad

            integral = source.flux_model.integral(
                lower_e_edges[:, np.newaxis], upper_e_edges[:, np.newaxis],
                dec_lower[np.newaxis, :], dec_upper[np.newaxis, :],
                0 * u.rad, 2 * np.pi * u.rad
            )

            aeff = np.array(self.effective_area._eff_area, copy=True) << (u.m**2)
            aeff[upper_e_edges < self._minimum_energy * u.GeV] = 0

        return (integral * aeff * source.redshift_factor(z)).sum()

    def _compute_exposure_integral(self):
        """
        Loop over sources and calculate the exposure integral.
        """

        self._integral_grid = []

        for source in self.source_list.sources:
            if not self._source_parameter_map[source]:
                continue

            this_free_pars = self._source_parameter_map[source]
            this_par_grids = [self._par_grids[par_name] for par_name in this_free_pars]
            integral_grids_tmp = np.zeros([self._n_grid_points] * len(this_par_grids)) << (1 / u.s)

            for i, grid_points in enumerate(product(*this_par_grids)):

                indices = np.unravel_index(i, integral_grids_tmp.shape)

                for par_name, par_value in zip(this_free_pars, grid_points):
                    par = Parameter.get_parameter(par_name)
                    par.value = par_value

                integral_grids_tmp[indices] += self.calculate_rate(source)

            self._integral_grid.append(integral_grids_tmp)

    def __call__(self):
        """
        Compute the exposure integrals.
        """

        self._compute_exposure_integral()
