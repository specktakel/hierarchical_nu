"""
Module for the precomputation of quantities 
to be passed to Stan models.
"""

import numpy as np

from source.source import SourceList, PointSource, DiffuseSource
from backend.stan_generator import StanGenerator


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

        self._alpha_grid = np.logspace(
            np.log(1.1), np.log(4.0), self._n_grid_points, base=np.e
        )

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
    def alpha_grid(self):

        return self._alpha_grid

    @property
    def integral_grid(self):

        return self._integral_grid

    def _power_law_integral(self, Elow, Ehigh, alpha):
        """
        Integral of (E/Emin)^-alpha from Elow to Ehigh.
        """

        norm = 1 / (np.power(self._minimum_energy, -alpha) * (alpha - 1))

        integ = np.power(Elow, 1 - alpha) - np.power(Ehigh, 1 - alpha)

        return norm * integ

    def _compute_exposure_integral(self):
        """
        Loop over sources and calculate the exposure integral.
        """

        self._integral_grid = []

        for source in self.source_list.sources:

            z = source.redshift

            integral_grid_tmp = np.zeros(len(self.alpha_grid))

            for i, alpha in enumerate(self.alpha_grid):

                if isinstance(source, PointSource):

                    dec = np.deg2rad(source.position.dec.value)

                    cosz = -np.sin(dec)

                    j = 0
                    for Em, EM in zip(
                        self.effective_area._tE_bin_edges[:-1],
                        self.effective_area._tE_bin_edges[1:],
                    ):

                        integ = self._power_law_integral(Em, EM, alpha)

                        if (
                            cosz < self.effective_area._cosz_bin_edges[0]
                            or cosz > self._effective_area._cosz_bin_edges[-1]
                            or EM < self._minimum_energy
                        ):

                            aeff = 0

                        else:

                            aeff = self.effective_area._eff_area[j][
                                np.digitize(cosz, self.effective_area._cosz_bin_edges)
                                - 1
                            ]

                        integral_grid_tmp[i] += (
                            integ * aeff * np.power(1 + z, 1 - alpha)
                        )  # GeV m^2

                        j += 1

                if isinstance(source, DiffuseSource):

                    j = 0
                    for Em, EM in zip(
                        self.effective_area._tE_bin_edges[:-1],
                        self.effective_area._tE_bin_edges[1:],
                    ):

                        k = 0
                        for czm, czM in zip(
                            self.effective_area._cosz_bin_edges[:-1],
                            self.effective_area._cosz_bin_edges[1:],
                        ):

                            E_integ = self._power_law_integral(Em, EM, alpha)

                            ang_integ = (czM - czm) * 2 * np.pi

                            if EM < self._minimum_energy:

                                aeff = 0

                            else:

                                aeff = self.effective_area._eff_area[j][k]

                            integral_grid_tmp[i] += (
                                E_integ
                                * (ang_integ / (4 * np.pi))
                                * aeff
                                * np.power(1 + z, 1 - alpha)
                            )  # GeV m^2

                            k += 1

                        j += 1

            self._integral_grid.append(integral_grid_tmp)

    def __call__(self):
        """
        Compute the exposure integrals.
        """

        self._compute_exposure_integral()
