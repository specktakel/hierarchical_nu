import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from .power_law import BoundedPowerLaw
from .parameter import Parameter

"""
Module for simple flux models used in
neutrino detection calculations
"""


class SpectralShape(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._parameters: Dict[str, Parameter] = {}

    @abstractmethod
    def __call__(self, energy: float):
        pass

    @abstractmethod
    def integral(self, bounds: Tuple[float, float]):
        pass

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def redshift_factor(self, redshift: float):
        """
        Factor that appears when evaluating the spectrum in the local frame
        """
        pass


class FluxModel(ABC):
    """
    Abstract base class for flux models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._parameters: Dict[str, Parameter] = {}

    @abstractmethod
    def __call__(self, energy: float, dec: float, ra: float):
        pass

    @abstractmethod
    def total_flux(self, energy: float):
        pass

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def redshift_factor(self, z: float):
        pass

    @abstractmethod
    def integral(
            self,
            energy_bounds: Tuple[float, float],
            dec_bounds: Tuple[float, float],
            ra_bounds: Tuple[float, float]):
        pass


class PointSourceFluxModel(FluxModel):
    def __init__(
            self,
            spectral_shape: SpectralShape,
            coord: Tuple[float, float],
            *args, **kwargs):
        super().__init__(self)
        self._spectral_shape = spectral_shape
        self._parameters = spectral_shape.parameters
        self.coord = coord

    def __call__(self, energy: float, dec: float, ra: float):
        if (dec == self.coord[0]) and (ra == self.coord[1]):
            return self._spectral_shape(energy) / (4 * np.pi)
        return 0

    @staticmethod
    def _is_in_bounds(value: float, bounds: Tuple[float, float]):
        return bounds[0] <= value <= bounds[1]

    @property
    def spectral_shape(self):
        return self._spectral_shape

    def integral(
            self,
            energy_bounds: Tuple[float, float],
            dec_bounds: Tuple[float, float],
            ra_bounds: Tuple[float, float]):
        if not self._is_in_bounds(self.coord[0], dec_bounds):
            return 0
        if not self._is_in_bounds(self.coord[1], ra_bounds):
            return 0

        ra_int = ra_bounds[1] - ra_bounds[0]
        dec_int = np.sin(dec_bounds[1]) - np.sin(dec_bounds[0])

        return self._spectral_shape.integral(energy_bounds) * ra_int * dec_int / (4 * np.pi)

    def total_flux(self, energy: float):
        return self._spectral_shape(energy)

    def redshift_factor(self, z: float):
        return self._spectral_shape.redshift_factor(z)


class IsotropicDiffuseBG(FluxModel):
    def __init__(self, spectral_shape: SpectralShape, *args, **kwargs):
        super().__init__(self)
        self._spectral_shape = spectral_shape
        self._parameters = spectral_shape.parameters

    def __call__(self, energy: float, dec: float, ra: float):
        return self._spectral_shape(energy) / (4 * np.pi)

    def total_flux(self, energy: float):
        return self._spectral_shape(energy)

    def redshift_factor(self, z: float):
        return self._spectral_shape.redshift_factor(z)

    def integral(
            self,
            energy_bounds: Tuple[float, float],
            dec_bounds: Tuple[float, float],
            ra_bounds: Tuple[float, float]):

        ra_int = ra_bounds[1] - ra_bounds[0]
        dec_int = np.sin(dec_bounds[1]) - np.sin(dec_bounds[0])

        return self._spectral_shape.integral(energy_bounds) * ra_int * dec_int / (4 * np.pi)


class PowerLawSpectrum(SpectralShape):
    """
    Power law shape
    """

    def __init__(self, normalisation, normalisation_energy, index,
                 lower_energy=1e2, upper_energy=np.inf, *args, **kwargs):
        """
        Power law flux models.

        normalisation: float
            Flux normalisation [GeV^-1 m^-2 s^-1]
        normalisation_energy: float
            Energy at which flux is normalised [GeV].
        index: float
            Spectral index of the power law.
        lower_energy: float
            Lower energy bound [GeV].
        upper_energy: float
            Upper enegry bound [GeV], unbounded by default.
        """

        super().__init__()
        self._normalisation_energy = normalisation_energy
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy
        self._parameters = {
            "norm": normalisation,
            "index": index
        }

    def set_parameter(self, par_name: str, par_value: float):
        if par_name not in self._parameters:
            raise ValueError("Parameter name {} not found".format(par_name))
        par = self._parameters[par_name]
        if not (par.par_range[0] <= self._par_value <= par.par_range[1]):
            raise ValueError("Parameter {} is out of bounds".format(par_name))

        par.value = par_value

    def redshift_factor(self, z: float):
        index = self._parameters["index"].value
        return np.power(1 + z, 1 - index)

    def __call__(self, energy: float):
        """
        dN/dEdAdt.
        """
        norm = self._parameters["norm"].value
        index = self._parameters["index"].value

        if (energy < self._lower_energy) or (energy > self._upper_energy):
            return np.nan
        else:
            return norm * np.power(energy / self._normalisation_energy, -index)

    def integral(self, bounds: Tuple[float, float]):
        r"""
        \int spectrum dE over finite energy bounds.

        Arguments:
            lower: float
                [GeV]
            upper: float
                [GeV]
        """
        norm = self._parameters["norm"].value
        index = self._parameters["index"].value
        lower, upper = bounds
        int_norm = (norm / (np.power(self._normalisation_energy, -index) * (1 - index)))

        return int_norm * (np.power(upper, 1 - index) - np.power(lower, 1 - index))

    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        :param N: Number of samples.
        """

        self.power_law = BoundedPowerLaw(self._index, self._lower_energy, self._upper_energy)

        return self.power_law.samples(N)


#     def _rejection_sample(self, min_energy):
#         """
#         Sample energies from the power law.
#         Uses rejection sampling.

#         :param min_energy: Minimum energy to sample from [GeV].
#         """

#         dist_upper_lim = self.spectrum(min_energy)

#         accepted = False

#         while not accepted:

#             energy = np.random.uniform(min_energy, 1e3*min_energy)
#             dist = np.random.uniform(0, dist_upper_lim)

#             if dist < self.spectrum(energy):

#                 accepted = True

#         return energy
