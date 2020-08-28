from abc import ABC, abstractmethod
from typing import Dict, Tuple

import astropy.units as u
import numpy as np

from .power_law import BoundedPowerLaw
from .parameter import Parameter
from ..backend.stan_generator import UserDefinedFunction
from ..backend.operations import FunctionCall
from ..backend.variable_definitions import ForwardVariableDef
from ..backend.expression import StringExpression, ReturnStatement

"""
Module for simple flux models used in
neutrino detection calculations
"""


class SpectralShape(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._parameters: Dict[str, Parameter] = {}

    @u.quantity_input
    @abstractmethod
    def __call__(self, energy: u.GeV) -> 1 / (u.GeV * u.m ** 2 * u.s):
        pass

    @u.quantity_input
    @abstractmethod
    def integral(self, lower: u.GeV, upper: u.GeV) -> 1 / (u.m ** 2 * u.s):
        pass

    @property
    @u.quantity_input
    @abstractmethod
    def total_flux_density(self) -> u.erg / u.s / u.m ** 2:
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

    @classmethod
    @abstractmethod
    def make_stan_sampling_func(cls):
        pass


class FluxModel(ABC):
    """
    Abstract base class for flux models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._parameters: Dict[str, Parameter] = {}

    @u.quantity_input
    @abstractmethod
    def __call__(
        self, energy: u.GeV, dec: u.rad, ra: u.rad
    ) -> 1 / (u.GeV * u.s * u.m ** 2 * u.sr):
        pass

    @u.quantity_input
    @abstractmethod
    def total_flux(self, energy: u.GeV) -> 1 / (u.m ** 2 * u.s * u.GeV):
        pass

    @property
    @u.quantity_input
    @abstractmethod
    def total_flux_int(self) -> 1 / (u.m ** 2 * u.s):
        pass

    @property
    @abstractmethod
    def energy_bounds(self):
        pass

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def redshift_factor(self, z: float):
        pass

    @u.quantity_input
    @abstractmethod
    def integral(
        self,
        e_low: u.GeV,
        e_up: u.GeV,
        dec_low: u.rad,
        dec_up: u.rad,
        ra_low: u.rad,
        ra_up: u.rad,
    ) -> 1 / (u.m ** 2 * u.s):
        pass

    @property
    @u.quantity_input
    @abstractmethod
    def total_flux_density(self) -> u.erg / u.s / u.m ** 2:
        pass

    @abstractmethod
    def make_stan_sampling_func(self, f_name: str) -> UserDefinedFunction:
        pass


class PointSourceFluxModel(FluxModel):
    @u.quantity_input
    def __init__(
        self, spectral_shape: SpectralShape, dec: u.rad, ra: u.rad, *args, **kwargs
    ):
        super().__init__(self)
        self._spectral_shape = spectral_shape
        self._parameters = spectral_shape.parameters
        self.dec = dec
        self.ra = ra

    @u.quantity_input
    def __call__(
        self, energy: u.GeV, dec: u.rad, ra: u.rad
    ) -> 1 / (u.GeV * u.s * u.m ** 2 * u.sr):
        if (dec == self.dec) and (ra == self.ra):
            return self._spectral_shape(energy) / (4 * np.pi * u.sr)
        return 0

    @staticmethod
    def _is_in_bounds(value: float, bounds: Tuple[float, float]):
        return bounds[0] <= value <= bounds[1]

    @property
    def spectral_shape(self):
        return self._spectral_shape

    @property
    def energy_bounds(self):
        return self._spectral_shape.energy_bounds

    @u.quantity_input
    def integral(
        self,
        e_low: u.GeV,
        e_up: u.GeV,
        dec_low: u.rad,
        dec_up: u.rad,
        ra_low: u.rad,
        ra_up: u.rad,
    ) -> 1 / (u.m ** 2 * u.s):
        if not self._is_in_bounds(self.dec, (dec_low, dec_up)):
            return 0
        if not self._is_in_bounds(self.ra, ra_low, ra_up):
            return 0

        ra_int = ra_up - ra_low
        dec_int = (np.sin(dec_up) - np.sin(dec_low)) * u.rad

        int = (
            self._spectral_shape.integral(e_low, e_up)
            * ra_int
            * dec_int
            / (4 * np.pi * u.sr)
        )
        return int

    @u.quantity_input
    def total_flux(self, energy: u.GeV) -> 1 / (u.m ** 2 * u.s * u.GeV):
        return self._spectral_shape(energy)

    @property
    @u.quantity_input
    def total_flux_int(self) -> 1 / (u.m ** 2 * u.s):
        return self._spectral_shape.integral(*self._spectral_shape.energy_bounds)

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m ** 2:
        return self._spectral_shape.total_flux_density

    def redshift_factor(self, z: float):
        return self._spectral_shape.redshift_factor(z)

    def make_stan_sampling_func(self, f_name: str) -> UserDefinedFunction:
        shape_rng = self._spectral_shape.make_stan_sampling_func()
        func = UserDefinedFunction(
            f_name,
            ["alpha", "e_low", "e_up", "dec", "ra"],
            ["real", "real", "real"],
            "vector",
        )

        with func:
            ret_vec = ForwardVariableDef("ret_vec", "vector[3]")
            ret_vec[1] << shape_rng("alpha", "e_low", "e_up")
            ret_vec[2] << StringExpression(["dec"])
            ret_vec[3] << StringExpression(["ra"])

            ReturnStatement([ret_vec])

        return func


class IsotropicDiffuseBG(FluxModel):
    def __init__(self, spectral_shape: SpectralShape, *args, **kwargs):
        super().__init__(self)
        self._spectral_shape = spectral_shape
        self._parameters = spectral_shape.parameters

    @u.quantity_input
    def __call__(
        self, energy: u.GeV, dec: u.rad, ra: u.rad
    ) -> 1 / (u.GeV * u.s * u.m ** 2 * u.sr):
        return self._spectral_shape(energy) / (4 * np.pi * u.sr)

    @u.quantity_input
    def total_flux(self, energy: u.GeV) -> 1 / (u.m ** 2 * u.s * u.GeV):
        return self._spectral_shape(energy)

    @property
    @u.quantity_input
    def total_flux_int(self) -> 1 / (u.m ** 2 * u.s):
        return self._spectral_shape.integral(*self._spectral_shape.energy_bounds)

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m ** 2:
        return self._spectral_shape.total_flux_density

    def redshift_factor(self, z: float):
        return self._spectral_shape.redshift_factor(z)

    @property
    def spectral_shape(self):
        return self._spectral_shape

    @property
    def energy_bounds(self):
        return self._spectral_shape.energy_bounds

    @u.quantity_input
    def integral(
        self,
        e_low: u.GeV,
        e_up: u.GeV,
        dec_low: u.rad,
        dec_up: u.rad,
        ra_low: u.rad,
        ra_up: u.rad,
    ) -> 1 / (u.m ** 2 * u.s):

        ra_int = ra_up - ra_low
        dec_int = (np.sin(dec_up) - np.sin(dec_low)) * u.rad

        return (
            self._spectral_shape.integral(e_low, e_up)
            * ra_int
            * dec_int
            / (4 * np.pi * u.sr)
        )

    def make_stan_sampling_func(self, f_name: str) -> UserDefinedFunction:
        shape_rng = self._spectral_shape.make_stan_sampling_func(f_name + "_shape_rng")
        func = UserDefinedFunction(
            f_name, ["alpha", "e_low", "e_up"], ["real", "real", "real"], "vector"
        )

        with func:
            pi = FunctionCall([], "pi")
            ret_vec = ForwardVariableDef("ret_vec", "vector[3]")
            ret_vec[1] << shape_rng("alpha", "e_low", "e_up")
            ret_vec[2] << FunctionCall(
                [FunctionCall([-1, 1], "uniform_rng")], "acos"
            ) - (pi / 2)
            ret_vec[3] << FunctionCall([0, 2 * pi], "uniform_rng")

            ReturnStatement([ret_vec])

        return func


class PowerLawSpectrum(SpectralShape):
    """
    Power law shape
    """

    @u.quantity_input
    def __init__(
        self,
        normalisation: Parameter,
        normalisation_energy: u.GeV,
        index: Parameter,
        lower_energy: u.GeV = 1e2 * u.GeV,
        upper_energy: u.GeV = np.inf * u.GeV,
        *args,
        **kwargs
    ):
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
            Upper energy bound [GeV], unbounded by default.
        """

        super().__init__()
        self._normalisation_energy = normalisation_energy
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy
        self._parameters = {"norm": normalisation, "index": index}

    @property
    def energy_bounds(self):
        return (self._lower_energy, self._upper_energy)

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

    @u.quantity_input
    def __call__(self, energy: u.GeV) -> 1 / (u.GeV * u.m ** 2 * u.s):
        """
        dN/dEdAdt.
        """
        norm = self._parameters["norm"].value
        index = self._parameters["index"].value

        if (energy < self._lower_energy) or (energy > self._upper_energy):
            return 0.0
        else:
            return norm * np.power(energy / self._normalisation_energy, -index)

    @u.quantity_input
    def integral(self, lower: u.GeV, upper: u.GeV) -> 1 / (u.m ** 2 * u.s):
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

        # Check edge cases
        lower[
            ((lower < self._lower_energy) & (upper > self._lower_energy))
        ] = self._lower_energy
        upper[
            ((lower < self._upper_energy) & (upper > self._upper_energy))
        ] = self._upper_energy

        if index == 1:
            # special case
            int_norm = norm / (np.power(self._normalisation_energy, -index))
            output = int_norm * (np.log(upper / lower))
        else:

            # Pull out the units here because astropy screwes this up sometimes
            int_norm = norm / (
                np.power(self._normalisation_energy / u.GeV, -index) * (1 - index)
            )
            output = (
                int_norm
                * (
                    np.power(upper / u.GeV, 1 - index)
                    - np.power(lower / u.GeV, 1 - index)
                )
                * u.GeV
            )

        # Correct if outside bounds
        output[(upper <= self._lower_energy)] = 0.0 * 1 / (u.m ** 2 * u.s)
        output[(lower >= self._upper_energy)] = 0.0 * 1 / (u.m ** 2 * u.s)

        return output

    def _integral(self, lower, upper):
        norm = self._parameters["norm"].value.value
        index = self._parameters["index"].value

        e0 = self._normalisation_energy.value

        if index == 1:
            # special case
            int_norm = norm / (np.power(e0, -index))
            return int_norm * (np.log(upper / lower))

        # Pull out the units here because astropy screwes this up sometimes
        int_norm = norm / (np.power(e0, -index) * (1 - index))
        return int_norm * (np.power(upper, 1 - index) - np.power(lower, 1 - index))

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m ** 2:
        norm = self._parameters["norm"].value
        index = self._parameters["index"].value  # diff flux * energy
        lower, upper = self._lower_energy, self._upper_energy

        if index == 2:
            # special case
            int_norm = norm / (np.power(self._normalisation_energy, -index))
            return int_norm * (np.log(upper / lower))

        # Pull out the units here because astropy screwes this up sometimes
        int_norm = norm / (
            np.power(self._normalisation_energy / u.GeV, -index) * (2 - index)
        )
        return (
            int_norm
            * (np.power(upper / u.GeV, 2 - index) - np.power(lower / u.GeV, 2 - index))
            * u.GeV ** 2
        )

    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        :param N: Number of samples.
        """

        self.power_law = BoundedPowerLaw(
            self._index, self._lower_energy, self._upper_energy
        )

        return self.power_law.samples(N)

    @classmethod
    def make_stan_sampling_func(cls, f_name) -> UserDefinedFunction:
        func = UserDefinedFunction(
            f_name, ["alpha", "e_low", "e_up"], ["real", "real", "real"], "real"
        )

        with func:
            uni_sample = ForwardVariableDef("uni_sample", "real")
            norm = ForwardVariableDef("norm", "real")
            alpha = StringExpression(["alpha"])
            e_low = StringExpression(["e_low"])
            e_up = StringExpression(["e_up"])

            norm << (1 - alpha) / (e_up ** (1 - alpha) - e_low ** (1 - alpha))

            uni_sample << FunctionCall([0, 1], "uniform_rng")
            ReturnStatement(
                [
                    (uni_sample * (1 - alpha) / norm + e_low ** (1 - alpha))
                    ** (1 / (1 - alpha))
                ]
            )

        return func

    @classmethod
    def make_stan_flux_conv_func(cls, f_name) -> UserDefinedFunction:
        """
        Factor to convert from total_flux_density to total_flux_int. 
        """

        func = UserDefinedFunction(
            f_name, ["alpha", "e_low", "e_up"], ["real", "real", "real"], "real"
        )

        with func:
            alpha = StringExpression(["alpha"])
            e_low = StringExpression(["e_low"])
            e_up = StringExpression(["e_up"])

            ReturnStatement(
                [
                    (e_up ** (1 - alpha) - e_low ** (1 - alpha))
                    / (e_up ** (2 - alpha) - e_low ** (2 - alpha))
                ]
            )

        return func
