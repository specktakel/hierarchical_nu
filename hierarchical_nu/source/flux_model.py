from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import astropy.units as u
import numpy as np

from .power_law import BoundedPowerLaw
from .parameter import Parameter
from ..backend.stan_generator import (
    UserDefinedFunction,
    IfBlockContext,
    ElseBlockContext,
    ElseIfBlockContext,
)
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
    def __call__(self, energy: u.GeV) -> 1 / (u.GeV * u.m**2 * u.s):
        pass

    @u.quantity_input
    @abstractmethod
    def integral(self, lower: u.GeV, upper: u.GeV) -> 1 / (u.m**2 * u.s):
        pass

    @property
    @u.quantity_input
    @abstractmethod
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        pass

    @property
    def parameters(self):
        return self._parameters

    @classmethod
    @abstractmethod
    def make_stan_sampling_func(cls):
        pass

    @classmethod
    @abstractmethod
    def make_stan_lpdf_func(cls):
        pass

    @classmethod
    @abstractmethod
    def make_stan_flux_conv_func(cls, f_name) -> UserDefinedFunction:
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
    def total_flux(self, energy: u.GeV) -> 1 / (u.m**2 * u.s * u.GeV):
        pass

    @property
    @u.quantity_input
    @abstractmethod
    def total_flux_int(self) -> 1 / (u.m**2 * u.s):
        pass

    @property
    @abstractmethod
    def energy_bounds(self):
        pass

    @property
    def parameters(self):
        return self._parameters

    @property
    @u.quantity_input
    @abstractmethod
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
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
    def total_flux(self, energy: u.GeV) -> 1 / (u.m**2 * u.s * u.GeV):
        return self._spectral_shape(energy)

    @property
    @u.quantity_input
    def total_flux_int(self) -> 1 / (u.m**2 * u.s):
        return self._spectral_shape.integral(*self._spectral_shape.energy_bounds)

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        return self._spectral_shape.total_flux_density

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
    ) -> 1 / (u.GeV * u.s * u.m**2 * u.sr):
        return self._spectral_shape(energy) / (4.0 * np.pi * u.sr)

    @u.quantity_input
    def total_flux(self, energy: u.GeV) -> 1 / (u.m**2 * u.s * u.GeV):
        return self._spectral_shape(energy)

    @property
    @u.quantity_input
    def total_flux_int(self) -> 1 / (u.m**2 * u.s):
        return self._spectral_shape.integral(*self._spectral_shape.energy_bounds)

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        return self._spectral_shape.total_flux_density

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
    ) -> 1 / (u.m**2 * u.s):
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
        Lives in the detector frame.

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

    @u.quantity_input
    def __call__(self, energy: u.GeV) -> 1 / (u.GeV * u.m**2 * u.s):
        """
        dN/(dEdAdt).
        """
        norm = self._parameters["norm"].value
        index = self._parameters["index"].value
        if isinstance(energy, np.ndarray):
            output = np.zeros_like(energy.value) * norm
            mask = np.nonzero(
                ((energy <= self._upper_energy) & (energy >= self._lower_energy))
            )
            output[mask] = norm * np.power(
                energy[mask] / self._normalisation_energy, -index
            )
            return output
        if (energy < self._lower_energy) or (energy > self._upper_energy):
            return 0.0 * norm
        else:
            return norm * np.power(energy / self._normalisation_energy, -index)

    @u.quantity_input
    def integral(self, lower: u.GeV, upper: u.GeV) -> 1 / (u.m**2 * u.s):
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
        output[(upper <= self._lower_energy)] = 0.0 * 1 / (u.m**2 * u.s)
        output[(lower >= self._upper_energy)] = 0.0 * 1 / (u.m**2 * u.s)

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
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        # Pull out the units here because astropy screws this up sometimes
        norm = self._parameters["norm"].value.to_value(1 / (u.GeV * u.m**2 * u.s))
        index = self._parameters["index"].value
        lower, upper = self._lower_energy.to_value(u.GeV), self._upper_energy.to_value(
            u.GeV
        )
        return_units = u.GeV / (u.m**2 * u.s)

        # Special case to avoid NaNs
        if index == 2:
            int_norm = norm * np.power(
                self._normalisation_energy.to_value(u.GeV), index
            )
            return int_norm * (np.log(upper / lower)) * return_units

        int_norm = (
            norm
            * np.power(self._normalisation_energy.to_value(u.GeV), index)
            / (2 - index)
        )
        return (
            int_norm
            * (np.power(upper, 2 - index) - np.power(lower, 2 - index))
            * return_units
        )

    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        :param N: Number of samples.
        """

        index = self._parameters["index"].value
        self.power_law = BoundedPowerLaw(
            index,
            self._lower_energy.to_value(u.GeV),
            self._upper_energy.to_value(u.GeV),
        )

        return self.power_law.samples(N)

    @u.quantity_input
    def pdf(self, E: u.GeV, Emin: u.GeV, Emax: u.GeV, apply_lim: bool = True):
        """
        Return PDF.
        """

        E_input = E.to_value(u.GeV)
        Emin_input = Emin.to_value(u.GeV)
        Emax_input = Emax.to_value(u.GeV)
        index = self._parameters["index"].value

        self.power_law = BoundedPowerLaw(
            index,
            Emin_input,
            Emax_input,
        )

        return self.power_law.pdf(E_input, apply_lim=apply_lim)

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
    def make_stan_lpdf_func(cls, f_name) -> UserDefinedFunction:
        func = UserDefinedFunction(
            f_name,
            ["E", "alpha", "e_low", "e_up"],
            ["real", "real", "real", "real"],
            "real",
        )

        with func:
            alpha = StringExpression(["alpha"])
            e_low = StringExpression(["e_low"])
            e_up = StringExpression(["e_up"])
            E = StringExpression(["E"])

            N = ForwardVariableDef("N", "real")
            p = ForwardVariableDef("p", "real")

            with IfBlockContext([E, ">", e_up]):
                ReturnStatement(["negative_infinity()"])
            with ElseIfBlockContext([E, "<", e_low]):
                ReturnStatement(["negative_infinity()"])

            with IfBlockContext([StringExpression([alpha, " == ", 1.0])]):
                N << 1.0 / (FunctionCall([e_up], "log") - FunctionCall([e_low], "log"))
            with ElseBlockContext():
                N << (1.0 - alpha) / (e_up ** (1.0 - alpha) - e_low ** (1.0 - alpha))

            p << N * FunctionCall([E, alpha * -1], "pow")

            ReturnStatement([FunctionCall([p], "log")])

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
            f1 = ForwardVariableDef("f1", "real")
            f2 = ForwardVariableDef("f2", "real")

            alpha = StringExpression(["alpha"])
            e_low = StringExpression(["e_low"])
            e_up = StringExpression(["e_up"])

            with IfBlockContext([StringExpression([alpha, " == ", 1.0])]):
                f1 << FunctionCall([e_up], "log") - FunctionCall([e_low], "log")
            with ElseBlockContext():
                f1 << (1 / (1 - alpha)) * (e_up ** (1 - alpha) - e_low ** (1 - alpha))

            with IfBlockContext([StringExpression([alpha, " == ", 2.0])]):
                f2 << FunctionCall([e_up], "log") - FunctionCall([e_low], "log")
            with ElseBlockContext():
                f2 << (1 / (2 - alpha)) * (e_up ** (2 - alpha) - e_low ** (2 - alpha))

            ReturnStatement([f1 / f2])

        return func


class TwiceBrokenPowerLaw(SpectralShape):
    """
    Power law shape
    """

    _index0 = -10.0
    _index2 = 10.0

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
        Lives in the detector frame.

        normalisation: float
            Flux normalisation [GeV^-1 m^-2 s^-1]
        normalisation_energy: float
            Energy at which flux is normalised [GeV].
        index: float
            Spectral index of the power law, is defined s.t. x^(-index) is used
        lower_energy: float
            Lower energy bound [GeV].
        upper_energy: float
            Upper energy bound [GeV], unbounded by default.
        """

        super().__init__()
        print("This is workin progress, be careful")
        self._e0 = Parameter.get_parameter("Emin").value
        self._e3 = Parameter.get_parameter("Emax").value
        self._normalisation_energy = normalisation_energy
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy
        self._parameters = {"norm": normalisation, "index": index}

    @property
    def I0(self):
        return self._piecewise_integral(self._e0, self._lower_energy, self._index0)

    @property
    def I1(self):
        return (
            self._piecewise_integral(
                self._lower_energy, self._upper_energy, self._parameters["index"].value
            )
            * self.N1
        )

    @property
    def I2(self):
        return (
            self._piecewise_integral(self._upper_energy, self._e3, self._index2)
            * self.N2
        )

    @property
    def N0(self):
        return 1.0

    @property
    def N1(self):
        """
        Normalisation factor, defined s.t. the broken power law has no jumps but only kinks
        """

        return np.power(
            self._lower_energy / self._normalisation_energy,
            self._parameters["index"].value - self._index0,
        )

    @property
    def N2(self):
        return self.N1 * np.power(
            self._upper_energy / self._normalisation_energy,
            self._index2 - self._parameters["index"].value,
        )

    @property
    def Itot(self):
        return self.I0 + self.I1 + self.I2

    @u.quantity_input
    def _piecewise_integral(self, x1: u.GeV, x2: u.GeV, gamma: float):
        """
        Returns the dimensionless integral x^(-gamma) dx
        """

        if x2 <= x1:
            return 0.0
        if gamma != 1.0:
            return (
                (np.power(x2 / u.GeV, 1.0 - gamma) - np.power(x1 / u.GeV, 1.0 - gamma))
                / (1.0 - gamma)
                / np.power(self._normalisation_energy / u.GeV, -gamma)
            )
        else:
            return (
                np.log((x2 / u.GeV) / (x1 / u.GeV)) * self._normalisation_energy / u.GeV
            )

    @property
    def energy_bounds(self):
        return (self._e0, self._e3)

    def set_parameter(self, par_name: str, par_value: float):
        if par_name not in self._parameters:
            raise ValueError("Parameter name {} not found".format(par_name))
        par = self._parameters[par_name]
        if not (par.par_range[0] <= self._par_value <= par.par_range[1]):
            raise ValueError("Parameter {} is out of bounds".format(par_name))

        par.value = par_value

    @u.quantity_input
    def __call__(self, energy: u.GeV) -> 1 / (u.GeV * u.m**2 * u.s):
        """
        dN/(dEdAdt).
        """
        norm = self._parameters["norm"].value
        index = self._parameters["index"].value
        index0 = self._index0
        index2 = self._index2
        norm_energy = self._normalisation_energy

        if energy.size > 1:
            output = np.zeros(energy.shape) << 1 / u.GeV / u.m**2 / u.s
            # Go through all parts of the broken power law
            lower = np.nonzero((energy <= self._lower_energy) & (energy >= self._e0))
            output[lower] = (
                np.power(energy[lower] / norm_energy, -index0) / self.norm_norm * norm
            )
            middle = np.nonzero(
                (energy <= self._upper_energy) & (energy > self._lower_energy)
            )
            output[middle] = (
                np.power(energy[middle] / norm_energy, -index)
                * self.N1
                / self.norm_norm
                * norm
            )
            upper = np.nonzero((energy <= self._e3) & (energy > self._upper_energy))
            output[upper] = (
                np.power(energy[upper] / norm_energy, -index2)
                * self.N2
                / self.norm_norm
                * norm
            )
            return output
        else:
            if (energy >= self._e0) and (energy < self._lower_energy):
                return np.power(energy / norm_energy, -index0) / self.norm_norm * norm
            elif energy >= self._lower_energy and energy <= self._upper_energy:
                return (
                    np.power(energy / norm_energy, -index)
                    * self.N1
                    / self.norm_norm
                    * norm
                )
            elif energy > self._upper_energy and energy <= self._e3:
                return (
                    np.power(energy / norm_energy, -index2)
                    * self.N2
                    / self.norm_norm
                    * norm
                )
            else:
                return 0 * norm

    @property
    def norm_norm(self):
        """
        Please propose a better name for this method
        Find the part of the broken power law in which normalisation_energy lies
        and return its Nx factor.
        """

        norm_energy = self._normalisation_energy

        # Go through all parts of the broken power law
        if (norm_energy >= self._e0) and (norm_energy < self._lower_energy):
            return 1.0
        elif norm_energy >= self._lower_energy and norm_energy <= self._upper_energy:
            return self.N1
        elif norm_energy > self._upper_energy and norm_energy <= self._e3:
            return self.N2
        else:
            raise ValueError("Norm energy is outside of the spectrum's energy range")

    @u.quantity_input
    def integral(self, lower: u.GeV, upper: u.GeV) -> 1 / (u.m**2 * u.s):
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

        indices = [self._index0, index, self._index2]
        energy_bounds = [self._e0, self._lower_energy, self._upper_energy, self._e3]

        norms = [self.N0, self.N1, self.N2]
        if lower.size > 1:
            squeeze = False
        else:
            squeeze = True
        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)

        # Check edge cases
        lower[((lower < self._e0) & (upper > self._e0))] = self._e0
        upper[((lower < self._e3) & (upper > self._e3))] = self._e3

        output = np.zeros(upper.shape) << 1 / u.m**2 / u.s

        for c, (_l, _u) in enumerate(zip(lower, upper)):
            l_idx = np.digitize(_l, energy_bounds) - 1
            u_idx = np.digitize(_u, energy_bounds, right=True) - 1

            if l_idx == u_idx:
                output[c] = (
                    self._piecewise_integral(_l, _u, indices[l_idx])
                    * norms[l_idx]
                    * norm
                    * u.GeV
                )
            else:
                for i in range(l_idx, u_idx + 1):
                    if i == l_idx:
                        output[c] += (
                            self._piecewise_integral(
                                _l, energy_bounds[i + 1], indices[i]
                            )
                            * norms[i]
                            * norm
                            * u.GeV
                        )
                    elif i == u_idx:
                        output[c] += (
                            self._piecewise_integral(energy_bounds[i], _u, indices[i])
                            * norms[i]
                            * norm
                            * u.GeV
                        )
                    else:
                        output[c] += (
                            self._piecewise_integral(
                                energy_bounds[i], energy_bounds[i + 1], indices[i]
                            )
                            * norms[i]
                            * norm
                            * u.GeV
                        )

        output = output / self.norm_norm
        # Correct if outside bounds
        output[(upper < self._e0)] = 0.0 * 1 / (u.m**2 * u.s)
        output[(lower > self._e3)] = 0.0 * 1 / (u.m**2 * u.s)

        if squeeze:
            output = output[0]
        return output

    def _integral(self, lower, upper):
        raise NotImplementedError
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
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        norm = self._parameters["norm"].value
        index = self._parameters["index"].value

        indices = [self._index0, index, self._index2]
        energy_bounds = [self._e0, self._lower_energy, self._upper_energy, self._e3]

        norms = [self.N0, self.N1, self.N2]

        lower = np.atleast_1d(self._e0)
        upper = np.atleast_1d(self._e3)

        output = np.zeros(upper.shape) << u.GeV / u.m**2 / u.s

        for c, (_l, _u) in enumerate(zip(lower, upper)):
            l_idx = np.digitize(_l, energy_bounds) - 1
            u_idx = np.digitize(_u, energy_bounds, right=True) - 1

            if l_idx == u_idx:
                output[c] = (
                    self._piecewise_integral(_l, _u, indices[l_idx] - 1.0)
                    * norms[l_idx]
                    * norm
                    * u.GeV
                    * self._normalisation_energy
                )
            else:
                for i in range(l_idx, u_idx + 1):
                    if i == l_idx:
                        output[c] += (
                            self._piecewise_integral(
                                _l, energy_bounds[i + 1], indices[i] - 1.0
                            )
                            * norms[i]
                            * norm
                            * u.GeV
                            * self._normalisation_energy
                        )
                    elif i == u_idx:
                        output[c] += (
                            self._piecewise_integral(
                                energy_bounds[i], _u, indices[i] - 1.0
                            )
                            * norms[i]
                            * norm
                            * u.GeV
                            * self._normalisation_energy
                        )
                    else:
                        output[c] += (
                            self._piecewise_integral(
                                energy_bounds[i], energy_bounds[i + 1], indices[i] - 1.0
                            )
                            * norms[i]
                            * norm
                            * u.GeV
                            * self._normalisation_energy
                        )

        output = output / self.norm_norm

        output = output.to(u.erg / u.m**2 / u.s)

        return output.squeeze()

    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        :param N: Number of samples.
        """

        raise NotImplementedError
        index = self._parameters["index"].value
        self.power_law = BoundedPowerLaw(
            index,
            self._lower_energy.to_value(u.GeV),
            self._upper_energy.to_value(u.GeV),
        )

        return self.power_law.samples(N)

    @u.quantity_input
    def pdf(self, E: u.GeV, Emin: u.GeV, Emax: u.GeV, *args, **kwargs):
        """
        Return PDF.
        """

        integral = self.integral(Emin, Emax)
        value = self.__call__(E)
        return value / integral * u.GeV

    @classmethod
    def make_stan_sampling_func(cls, f_name) -> UserDefinedFunction:
        raise NotImplementedError

    @classmethod
    def make_stan_lpdf_func(cls, f_name) -> UserDefinedFunction:
        func = UserDefinedFunction(
            f_name,
            ["E", "alpha1", "alpha2", "alpha3", "e1", "e2", "e3", "e4"],
            ["real", "real", "real", "real", "real", "real", "real", "real"],
            "real",
        )

        with func:
            alpha1 = StringExpression(["alpha1"])
            alpha2 = StringExpression(["alpha2"])
            alpha3 = StringExpression(["alpha3"])
            e1 = StringExpression(["e1"])
            e2 = StringExpression(["e2"])
            e3 = StringExpression(["e3"])
            e4 = StringExpression(["e4"])
            E = StringExpression(["E"])

            I1 = ForwardVariableDef("I1", "real")
            I1 << (
                FunctionCall([e2, 1.0 - alpha1], "pow")
                - FunctionCall([e1, 1.0 - alpha1], "pow")
            ) / (1.0 - alpha1)
            N2 = ForwardVariableDef("N2", "real")
            N2 << FunctionCall([e2, alpha2 - alpha1], "pow")
            I2 = ForwardVariableDef("I2", "real")
            with IfBlockContext([alpha2, "==", 1.0]):
                I2 << FunctionCall([e3 / e2], "log") * N2
            with ElseBlockContext():
                (
                    I2
                    << (
                        FunctionCall([e3, 1.0 - alpha2], "pow")
                        - FunctionCall([e2, 1.0 - alpha2], "pow")
                    )
                    / (1.0 - alpha2)
                    * N2
                )
            N3 = ForwardVariableDef("N3", "real")
            N3 << FunctionCall([e3, alpha3 - alpha2], "pow") * N2
            I3 = ForwardVariableDef("I3", "real")
            (
                I3
                << (
                    FunctionCall([e4, 1.0 - alpha3], "pow")
                    - FunctionCall([e3, 1.0 - alpha3], "pow")
                )
                / (1.0 - alpha3)
                * N3
            )
            I = ForwardVariableDef("I", "real")
            I << I1 + I2 + I3
            p = ForwardVariableDef("p", "real")
            with IfBlockContext([E, "<", e2]):
                p << FunctionCall([E, alpha1 * -1], "pow")
            with ElseIfBlockContext([E, "<", e3]):
                p << FunctionCall([E, alpha2 * -1], "pow") * N2
            with ElseBlockContext():
                p << FunctionCall([E, alpha3 * -1], "pow") * N3

            ReturnStatement([FunctionCall([p / I], "log")])

        return func

    @classmethod
    def make_stan_flux_conv_func(cls, f_name) -> UserDefinedFunction:
        """
        Factor to convert from total_flux_density to total_flux_int.
        Keep for now and disregard the flanks.
        """

        func = UserDefinedFunction(
            f_name,
            ["alpha1", "alpha2", "alpha3", "e1", "e2", "e3", "e4"],
            ["real", "real", "real", "real", "real", "real", "real"],
            "real",
        )

        with func:
            alpha1 = StringExpression(["alpha1"])
            alpha2 = StringExpression(["alpha2"])
            alpha3 = StringExpression(["alpha3"])
            e1 = StringExpression(["e1"])
            e2 = StringExpression(["e2"])
            e3 = StringExpression(["e3"])
            e4 = StringExpression(["e4"])
            f1 = ForwardVariableDef("f1", "real")
            f2 = ForwardVariableDef("f2", "real")

            I1 = ForwardVariableDef("I1", "real")
            I2 = ForwardVariableDef("I2", "real")
            I3 = ForwardVariableDef("I3", "real")

            N2 = ForwardVariableDef("N2", "real")
            N3 = ForwardVariableDef("N3", "real")

            I1 << (
                FunctionCall([e2, 1.0 - alpha1], "pow")
                - FunctionCall([e1, 1.0 - alpha1], "pow")
            ) / (1.0 - alpha1)
            N2 << FunctionCall([e2, alpha2 - alpha1], "pow")
            with IfBlockContext([alpha2, "==", 1.0]):
                I2 << FunctionCall([e3 / e2], "log") * N2
            with ElseBlockContext():
                (
                    I2
                    << (
                        FunctionCall([e3, 1.0 - alpha2], "pow")
                        - FunctionCall([e2, 1.0 - alpha2], "pow")
                    )
                    / (1.0 - alpha2)
                    * N2
                )
            N3 << FunctionCall([e3, alpha3 - alpha2], "pow") * N2
            (
                I3
                << (
                    FunctionCall([e4, 1.0 - alpha3], "pow")
                    - FunctionCall([e3, 1.0 - alpha3], "pow")
                )
                / (1.0 - alpha3)
                * N3
            )
            f1 << I1 + I2 + I3

            I1 << (
                FunctionCall([e2, 2.0 - alpha1], "pow")
                - FunctionCall([e1, 2.0 - alpha1], "pow")
            ) / (2.0 - alpha1)
            with IfBlockContext([alpha2, "==", 1.0]):
                I2 << FunctionCall([e3 / e2], "log") * N2
            with ElseBlockContext():
                (
                    I2
                    << (
                        FunctionCall([e3, 2.0 - alpha2], "pow")
                        - FunctionCall([e2, 2.0 - alpha2], "pow")
                    )
                    / (2.0 - alpha2)
                    * N2
                )
            (
                I3
                << (
                    FunctionCall([e4, 2.0 - alpha3], "pow")
                    - FunctionCall([e3, 2.0 - alpha3], "pow")
                )
                / (2.0 - alpha3)
                * N3
            )
            f2 << I1 + I2 + I3
            ReturnStatement([f1 / f2])

        return func


@u.quantity_input
def integral_power_law(
    gamma: float, n: Union[float, int], x1: u.GeV, x2: u.GeV, x0: u.GeV = 1e5 * u.GeV
):
    """
    Implements expectation value (up to normalisation
    of the distribution) over power law/pareto distribution of the form
    <x^n> = \int_{x1}^{x2} (x/x_0)^{-\gamma} x^n dx

    Is used for flux conversion from energy to numbers.

    Should in the end replace flux_conv_
    """

    if gamma != n + 1.0:
        return (
            np.power(x0, gamma)
            * (np.power(x2, n + 1 - gamma) - np.power(x1, n + 1 - gamma))
            / (n + 1 - gamma)
        )
    else:
        return np.power(x0, n + 1) * np.log(x2 / x1)


def flux_conv_(alpha, e_low, e_up):
    if alpha == 1.0:
        f1 = np.log(e_up) - np.log(e_low)
    else:
        f1 = 1 / (1 - alpha) * (np.power(e_up, 1 - alpha) - np.power(e_low, 1 - alpha))

    if alpha == 2.0:
        f2 = np.log(e_up) - np.log(e_low)
    else:
        f2 = 1 / (2 - alpha) * (np.power(e_up, 2 - alpha) - np.power(e_low, 2 - alpha))

    return f1 / f2
