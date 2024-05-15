from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import astropy.units as u
import numpy as np
from scipy.integrate import quad

from .power_law import BoundedPowerLaw
from .parameter import Parameter
from ..backend.stan_generator import (
    UserDefinedFunction,
    IfBlockContext,
    ElseBlockContext,
    ElseIfBlockContext,
)
from ..backend.operations import FunctionCall
from ..backend.variable_definitions import (
    ForwardVariableDef,
    InstantVariableDef,
    ForwardArrayDef,
)
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
        lower[((lower < self._lower_energy) & (upper > self._lower_energy))] = (
            self._lower_energy
        )
        upper[((lower < self._upper_energy) & (upper > self._upper_energy))] = (
            self._upper_energy
        )

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
    def make_stan_sampling_lpdf_func(cls, f_name) -> UserDefinedFunction:
        return cls.make_stan_lpdf_func(f_name)

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


class TwiceBrokenPowerLaw(PowerLawSpectrum, SpectralShape):
    """
    Twice broken power law.
    Adds very steep flanks outside of lower and upper energy to restrict
    the energy range to a defined range while keeping the numerics of stan satisfied.
    Inherits flux integrals etc. from PowerLawSpectrum, meaning that the flanks are not included
    in any of those calculations.
    """

    _index0 = -15.0
    _index2 = 15.0

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

        super(SpectralShape, self).__init__()
        self._normalisation_energy = normalisation_energy
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy
        self._parameters = {"norm": normalisation, "index": index}

    @classmethod
    def make_stan_lpdf_func(cls, f_name) -> UserDefinedFunction:
        func = UserDefinedFunction(
            f_name,
            [
                "E",
                "alpha2",
                "e2",
                "e3",
            ],
            ["real", "real", "real", "real"],
            "real",
        )

        with func:
            alpha1 = InstantVariableDef("alpha1", "real", [cls._index0])
            alpha2 = StringExpression(["alpha2"])
            alpha3 = InstantVariableDef("alpha3", "real", [cls._index2])
            e2 = StringExpression(["e2"])
            e3 = StringExpression(["e3"])
            E = StringExpression(["E"])

            I = ForwardVariableDef("I2", "real")
            with IfBlockContext([alpha2, "==", 1.0]):
                I << FunctionCall([e3 / e2], "log")
            with ElseBlockContext():
                (
                    I
                    << (
                        FunctionCall([e3, 1.0 - alpha2], "pow")
                        - FunctionCall([e2, 1.0 - alpha2], "pow")
                    )
                    / (1.0 - alpha2)
                )
            p = ForwardVariableDef("p", "real")
            with IfBlockContext([E, "<", e2]):
                (
                    p
                    << FunctionCall([E / e2, alpha1 * -1], "pow")
                    * FunctionCall([e2, alpha2 * -1], "pow")
                    / I
                )
            with ElseIfBlockContext([E, "<", e3]):
                p << FunctionCall([E, alpha2 * -1], "pow") / I
            with ElseBlockContext():
                (
                    p
                    << FunctionCall([E / e3, alpha3 * -1], "pow")
                    * FunctionCall([e3, alpha2 * -1], "pow")
                    / I
                )

            ReturnStatement([FunctionCall([p], "log")])

        return func


class LogParabolaSpectrum(SpectralShape):
    @u.quantity_input
    def __init__(
        self,
        normalisation: Parameter,
        normalisation_energy: u.GeV,
        alpha: Parameter,
        beta: Parameter,
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
        alpha: Parameter
            Slope parameter of spectral shape
        beta: Parameter
            Curvature parameter of spectral shape
        lower_energy: float
            Lower energy bound [GeV].
        upper_energy: float
            Upper energy bound [GeV], unbounded by default.
        """

        super().__init__()
        self._normalisation = normalisation
        self._normalisation_energy = normalisation_energy
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy
        self._parameters = {"norm": normalisation, "alpha": alpha, "beta": beta}

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
        alpha = self.parameters["alpha"].value
        beta = self.parameters["beta"].value
        E = energy.to_value(u.GeV)
        E0 = self._normalisation_energy.to_value(u.GeV)
        norm = self.parameters["norm"].value
        if isinstance(energy, np.ndarray):
            output = np.zeros_like(energy.value) * norm
            mask = np.nonzero(
                ((energy <= self._upper_energy) & (energy >= self._lower_energy))
            )
            output[mask] = norm * np.power(
                E[mask] / E0, -alpha - beta * np.log(E[mask] / E0)
            )
            return output
        if (energy < self._lower_energy) or (energy > self._upper_energy):
            return 0.0 * norm
        else:
            return norm * np.power(E / E0, -alpha - beta * np.log(E / E0))

    @u.quantity_input
    def integral(self, lower: u.GeV, upper: u.GeV) -> 1 / (u.m**2 * u.s):
        # Calculate numerically in log space
        # Transformed integrand reads
        # exp((1-alpha) * x - beta * x**2
        alpha = self.parameters["alpha"].value
        beta = self.parameters["beta"].value
        E0 = self._normalisation_energy.to_value(u.GeV)
        norm = self.parameters["norm"].value

        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        xl = np.log(lower.to_value(u.GeV) / E0)
        xh = np.log(upper.to_value(u.GeV) / E0)

        # Check edge cases
        lower[((lower < self._lower_energy) & (upper > self._lower_energy))] = (
            self._lower_energy
        )
        upper[((lower < self._upper_energy) & (upper > self._upper_energy))] = (
            self._upper_energy
        )

        def integrand(x):
            return np.exp((1 - alpha) * x - beta * np.power(x, 2))

        results = np.zeros_like(xl)
        for c in range(xl.size):
            results[c] = quad(integrand, xl[c], xh[c])[0]
        return results * norm * E0 * u.GeV

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        # Calculate numerically in log space
        # Transformed integrand reads
        # exp((2-alpha) * x - beta * x**2
        alpha = self.parameters["alpha"].value
        beta = self.parameters["beta"].value
        E0 = self._normalisation_energy.to_value(u.GeV)
        norm = self.parameters["norm"].value

        xl = np.log(self._lower_energy.to_value(u.GeV) / E0)
        xh = np.log(self._upper_energy.to_value(u.GeV) / E0)

        def integrand(x):
            return np.exp((2 - alpha) * x - beta * np.power(x, 2))

        result = quad(integrand, xl, xh)[0]
        return result * norm * np.power(E0, 2) * u.GeV**2

    @property
    def parameters(self):
        return self._parameters

    @classmethod
    def make_stan_sampling_func(cls):
        pass

    @classmethod
    def make_stan_utility_func(cls):
        # Needs to be passed to integrate_1d
        # is defined in logspace for faster integration
        lp = UserDefinedFunction(
            "logparabola_dN_dx",
            ["x", "xc", "theta", "x_r", "x_i"],
            ["real", "real", "array[] real", "data array[] real", "data array[] int"],
            "real",
        )
        with lp:
            x = StringExpression(["x"])
            a = InstantVariableDef("a", "real", ["theta[1]"])
            b = InstantVariableDef("b", "real", ["theta[2]"])
            ReturnStatement(
                [FunctionCall([(1.0 - a) * x - b * FunctionCall([x, 2], "pow")], "exp")]
            )
        # Same here
        lp = UserDefinedFunction(
            "logparabola_dN_dx",
            ["x", "xc", "theta", "x_r", "x_i"],
            ["real", "real", "array[] real", "data array[] real", "data array[] int"],
            "real",
        )
        with lp:
            x = StringExpression(["x"])
            a = InstantVariableDef("a", "real", ["theta[1]"])
            b = InstantVariableDef("b", "real", ["theta[2]"])
            ReturnStatement(
                [FunctionCall([(2.0 - a) * x - b * FunctionCall([x, 2], "pow")], "exp")]
            )

    @classmethod
    def make_stan_lpdf_func(cls, f_name) -> UserDefinedFunction:
        func = UserDefinedFunction(
            f_name,
            ["E", "theta", "x_r", "x_i"],
            ["real", "array[] real", "data array[] real", "data array[] int"],
            "real",
        )

        with func:
            # Use this packed definition to please integrate_1d
            theta = StringExpression(["theta"])

            # Unpack variables
            E0 = InstantVariableDef("E0", "real", ["x_r[1]"])
            e_low = InstantVariableDef("e_low", "real", ["x_r[2]"])
            e_up = InstantVariableDef("e_up", "real", ["x_r[3]"])
            E = StringExpression(["E"])

            N = ForwardVariableDef("N", "real")
            p = ForwardVariableDef("p", "real")

            with IfBlockContext([E, ">", e_up]):
                ReturnStatement(["negative_infinity()"])
            with ElseIfBlockContext([E, "<", e_low]):
                ReturnStatement(["negative_infinity()"])

            logEL_E0 = InstantVariableDef("logELE0", "real", ["log(e_low/E0)"])
            logEU_E0 = InstantVariableDef("logEUE0", "real", ["log(e_up/E0)"])
            logE_E0 = InstantVariableDef("logEE0", "real", ["log(E/E0)"])

            N << FunctionCall(
                [
                    FunctionCall(
                        [
                            "logparabola_dN_dx",
                            logEL_E0,
                            logEU_E0,
                            theta,
                            "x_r",
                            "x_i",
                        ],
                        "integrate_1d",
                    )
                ],
                "log",
            )
            p << logE_E0 * (-theta[1] - theta[2] * logE_E0)
            ReturnStatement([p - N])

    @classmethod
    def make_stan_flux_conv_func(cls, f_name) -> UserDefinedFunction:
        func = UserDefinedFunction(
            f_name,
            ["E0", "alpha", "beta", "e_low", "e_up"],
            ["real", "real", "real", "real", "real"],
            "real",
        )

        with func:
            alpha = StringExpression(["alpha"])
            beta = StringExpression(["beta"])

            f1 = ForwardVariableDef("f1", "real")
            f2 = ForwardVariableDef("f2", "real")

            theta = ForwardArrayDef("theta", "real", ["[2]"])
            x_i = ForwardArrayDef("x_i", "int", ["[0]"])
            x_r = ForwardArrayDef("x_r", "real", ["[0]"])
            theta[1] << alpha
            theta[2] << beta
            logEL_E0 = InstantVariableDef("logELE0", "real", ["log(e_low/E0)"])
            logEU_E0 = InstantVariableDef("logEUE0", "real", ["log(e_up/E0)"])

            f1 << FunctionCall(
                [
                    FunctionCall(
                        [
                            "logparabola_dN_dx",
                            logEL_E0,
                            logEU_E0,
                            theta,
                            x_r,
                            x_i,
                        ],
                        "integrate_1d",
                    )
                ],
                "log",
            )

            f2 << FunctionCall(
                [
                    FunctionCall(
                        [
                            "logparabola_x_dN_dx",
                            logEL_E0,
                            logEU_E0,
                            theta,
                            x_r,
                            x_i,
                        ],
                        "integrate_1d",
                    )
                ],
                "log",
            )

            ReturnStatement([f1 / f2])


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
