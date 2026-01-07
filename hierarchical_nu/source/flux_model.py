from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import astropy.units as u
import numpy as np
from scipy.integrate import quad
from scipy.special import erf

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
)
from ..backend.expression import StringExpression, ReturnStatement

"""
Module for simple flux models used in
neutrino detection calculations
"""


class SpectralShape(ABC):
    """
    Abstract base class for spectral shapes
    """

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

    def set_parameter(self, par_name: str, par_value: float):
        if par_name not in self._parameters:
            raise ValueError("Parameter name {} not found".format(par_name))
        par = self._parameters[par_name]
        if not (par.par_range[0] <= par_value <= par.par_range[1]):
            raise ValueError("Parameter {} is out of bounds".format(par_name))

        # Copy fixed or not, release param, set value and restore initial state
        fixed = par.fixed
        par.fixed = False
        par.value = par_value
        par.fixed = fixed

    @property
    def energy_bounds(self):
        return (self._lower_energy, self._upper_energy)

    @property
    def name(self):
        return self._name


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
        """
        Define point source flux model at arbitrary position on the sky
        :param spectral_shape: energy spectral shape
        :param dec: source declination
        :param ra: source right ascension
        """

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
        """
        Isotropic flux model, i.e. for astrophysical diffuse flux
        :param spectral_shape: energy spectral shape
        """
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

    _name = "power-law"

    @u.quantity_input
    def __init__(
        self,
        normalisation: Parameter,
        normalisation_energy: u.GeV,
        index: Parameter,
        lower_energy: u.GeV = 1e2 * u.GeV,
        upper_energy: u.GeV = np.inf * u.GeV,
        *args,
        **kwargs,
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

    @u.quantity_input
    def __call__(self, energy: u.GeV) -> 1 / (u.GeV * u.m**2 * u.s):
        """
        dN/(dEdAdt).
        """
        norm = self._parameters["norm"].value
        index = self._parameters["index"].value

        return_units = 1 / u.GeV / u.m**2 / u.s
        if energy.shape != ():
            output = np.zeros_like(energy.value) * norm
            mask = np.nonzero(
                ((energy <= self._upper_energy) & (energy >= self._lower_energy))
            )
            output[mask] = (
                norm * np.power(energy[mask] / self._normalisation_energy, -index)
            ).to(return_units)
            return output
        if (energy < self._lower_energy) or (energy > self._upper_energy):
            return 0.0 * norm.to(return_units)
        else:
            return (norm * np.power(energy / self._normalisation_energy, -index)).to(
                return_units
            )

    @classmethod
    @u.quantity_input
    def _flux(
        cls, norm: 1 / u.TeV / u.m**2 / u.s, E: u.GeV, alpha: float, Enorm: u.GeV
    ) -> 1 / (u.TeV / u.m**2 / u.s):
        return norm * np.power(E / Enorm, -alpha)

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
    def make_stan_sampling_func(cls, f_name, *args, **kwargs) -> UserDefinedFunction:
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
    def make_stan_lpdf_func(cls, f_name, *args, **kwargs) -> UserDefinedFunction:
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
    def make_stan_flux_conv_func(cls, f_name, *args, **kwargs) -> UserDefinedFunction:
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

    @classmethod
    def make_stan_diff_flux_conv_func(
        cls, f_name, *args, **kwargs
    ) -> UserDefinedFunction:
        """
        Factor to convert from differential norm to integrated norm,
        i.e. integrate dN/dE dE over energy range
        """

        func = UserDefinedFunction(
            f_name,
            ["norm_energy", "alpha", "e_low", "e_up"],
            ["real", "real", "real", "real"],
            "real",
        )

        with func:
            integral = ForwardVariableDef("integral", "real")
            alpha = StringExpression(["alpha"])
            e_low = StringExpression(["e_low"])
            e_up = StringExpression(["e_up"])
            norm_energy = StringExpression(["norm_energy"])

            with IfBlockContext([StringExpression([alpha, " == ", 1.0])]):
                integral << FunctionCall([e_up], "log") - FunctionCall([e_low], "log")
            with ElseBlockContext():
                integral << (1 / (1 - alpha)) * (
                    e_up ** (1 - alpha) - e_low ** (1 - alpha)
                )

            ReturnStatement([integral * norm_energy**alpha])
        return func

    @staticmethod
    def flux_conv_(**kwargs):
        alpha = kwargs["alpha"]
        e_low = kwargs["e_low"]
        e_up = kwargs["e_up"]

        if alpha == 1.0:
            f1 = np.log(e_up) - np.log(e_low)
        else:
            f1 = (
                1
                / (1 - alpha)
                * (np.power(e_up, 1 - alpha) - np.power(e_low, 1 - alpha))
            )

        if alpha == 2.0:
            f2 = np.log(e_up) - np.log(e_low)
        else:
            f2 = (
                1
                / (2 - alpha)
                * (np.power(e_up, 2 - alpha) - np.power(e_low, 2 - alpha))
            )

        return f1 / f2


class TwiceBrokenPowerLaw(PowerLawSpectrum, SpectralShape):
    """
    Twice broken power law.
    Adds very steep flanks outside of lower and upper energy to restrict
    the energy range to a defined range while keeping the numerics of stan satisfied.
    Inherits flux integrals etc. from PowerLawSpectrum, meaning that the flanks are not included
    in any of those calculations.
    """

    # Indices at the flanks
    _index0 = -15.0
    _index2 = 15.0
    _name = "twice-broken-power-law"

    @u.quantity_input
    def __init__(
        self,
        normalisation: Parameter,
        normalisation_energy: u.GeV,
        index: Parameter,
        lower_energy: u.GeV = 1e2 * u.GeV,
        upper_energy: u.GeV = np.inf * u.GeV,
        *args,
        **kwargs,
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
    def make_stan_lpdf_func(cls, f_name, *args, **kwargs) -> UserDefinedFunction:
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
    """Logparabola spectral model"""

    _name = "logparabola"

    @u.quantity_input
    def __init__(
        self,
        normalisation: Parameter,
        normalisation_energy: Parameter,
        alpha: Parameter,
        beta: Parameter,
        lower_energy: u.GeV = 1e2 * u.GeV,
        upper_energy: u.GeV = np.inf * u.GeV,
        *args,
        **kwargs,
    ):
        """
        Logparabola flux model.
        Lives in the detector frame.

        normalisation: float
            Flux normalisation [GeV^-1 m^-2 s^-1]
        normalisation_energy: float or Parameter
            Energy at which flux is normalised [GeV] and local index is alpha.
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
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy
        self._parameters = {
            "norm": normalisation,
            "index": alpha,
            "beta": beta,
            "norm_energy": normalisation_energy,
        }

    @u.quantity_input
    def __call__(self, energy: u.GeV) -> 1 / (u.GeV * u.m**2 * u.s):
        alpha = self.parameters["index"].value
        beta = self.parameters["beta"].value
        E = energy.to_value(u.GeV)
        E0 = self.parameters["norm_energy"].value.to_value(u.GeV)
        norm = self.parameters["norm"].value
        return_units = 1 / (u.GeV * u.m**2 * u.s)

        if energy.shape != ():
            output = np.zeros_like(energy.value) * norm
            mask = np.nonzero(
                ((energy <= self._upper_energy) & (energy >= self._lower_energy))
            )
            output[mask] = norm * np.power(
                E[mask] / E0, -alpha - beta * np.log(E[mask] / E0)
            )
            return output.to(return_units)
        if (energy < self._lower_energy) or (energy > self._upper_energy):
            return (0.0 * norm).to(return_units)
        else:
            return (norm * np.power(E / E0, -alpha - beta * np.log(E / E0))).to(
                return_units
            )

    @classmethod
    @u.quantity_input
    def _flux(
        cls,
        norm: 1 / u.TeV / u.m**2 / u.s,
        E: u.GeV,
        alpha: float,
        beta: float,
        Enorm: u.GeV,
    ) -> 1 / (u.TeV / u.m**2 / u.s):
        return norm * np.power(E / Enorm, -alpha - beta * np.log(E / Enorm))

    @u.quantity_input
    def integral(self, lower: u.GeV, upper: u.GeV) -> 1 / (u.m**2 * u.s):
        # Calculate numerically in log space
        # Transformed integrand reads
        # exp((1-alpha) * x - beta * x**2
        alpha = self.parameters["index"].value
        beta = self.parameters["beta"].value
        E0 = self.parameters["norm_energy"].value.to_value(u.GeV)
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

        results = np.zeros(xl.shape) << 1 / u.m**2 / u.s
        for c in range(xl.size):
            results[c] = (
                quad(self._dN_dx, xl[c], xh[c], (alpha, beta))[0] * norm * E0 * u.GeV
            )

        if results.size == 1:
            return results[0]
        return results

    @classmethod
    def _dN_dE(cls, E, E0, alpha, beta):
        # Unnormalised pdf
        return np.power(E / E0, -alpha - beta * np.log(E / E0))

    @classmethod
    def _dN_dx(cls, x, alpha, beta):
        return np.exp((1.0 - alpha) * x - beta * np.power(x, 2))

    @classmethod
    def _x_dN_dx(cls, x, alpha, beta):
        return np.exp((2.0 - alpha) * x - beta * np.power(x, 2))

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        # Calculate numerically in log space
        # Transformed integrand reads
        # exp((2-alpha) * x - beta * x**2
        alpha = self.parameters["index"].value
        beta = self.parameters["beta"].value
        E0 = self.parameters["norm_energy"].value.to_value(u.GeV)
        norm = self.parameters["norm"].value

        xl = np.log(self._lower_energy.to_value(u.GeV) / E0)
        xh = np.log(self._upper_energy.to_value(u.GeV) / E0)

        result = quad(self._x_dN_dx, xl, xh, (alpha, beta))[0]
        return result * norm * np.power(E0, 2) * u.GeV**2

    def flux_conv(self):
        # Calculate (\int dN / dE / dA /dt dE)/(\int E dN / dE / dA / dt dE)
        alpha = self.parameters["index"].value
        beta = self.parameters["beta"].value
        E0 = self.parameters["norm_energy"].value.to_value(u.GeV)

        xl = np.log(self._lower_energy.to_value(u.GeV) / E0)
        xh = np.log(self._upper_energy.to_value(u.GeV) / E0)

        f1 = quad(self._dN_dx, xl, xh, (alpha, beta))[0]
        f2 = quad(self._x_dN_dx, xl, xh, (alpha, beta))[0]

        return f1 / f2 / E0

    @classmethod
    def flux_conv_(cls, **kwargs):
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        e_up = kwargs["e_up"]
        e_low = kwargs["e_low"]
        e_0 = kwargs["e_0"]
        xl = np.log(e_low / e_0)
        xh = np.log(e_up / e_0)

        f1 = quad(cls._dN_dx, xl, xh, (alpha, beta))[0]
        f2 = quad(cls._x_dN_dx, xl, xh, (alpha, beta))[0]

        return f1 / f2 / e_0

    @property
    def parameters(self):
        return self._parameters

    @u.quantity_input
    def pdf(self, E: u.GeV, Emin: u.GeV, Emax: u.GeV, apply_lim: bool = True):
        """
        Return PDF.
        """

        E_input = np.atleast_1d(E.to_value(u.GeV))
        Emin_input = Emin.to_value(u.GeV)
        Emax_input = Emax.to_value(u.GeV)
        alpha = self._parameters["index"].value
        beta = self._parameters["beta"].value
        E0 = self.parameters["norm_energy"].value.to_value(u.GeV)

        norm = (
            quad(
                self._dN_dx,
                np.log(Emin_input / E0),
                np.log(Emax_input / E0),
                (alpha, beta),
            )[0]
            * E0
        )

        pdf = self._dN_dE(E_input, E0, alpha, beta) / norm
        if apply_lim:
            pdf[E_input < Emin_input] = 0.0
            pdf[E_input > Emax_input] = 0.0
        if pdf.size == 1:
            return pdf[0]
        return pdf

    @classmethod
    def make_stan_sampling_lpdf_func(cls, f_name) -> UserDefinedFunction:
        return cls.make_stan_lpdf_func(f_name, False, False, False)

    @classmethod
    def make_stan_sampling_func(cls, f_name, *args, **kwargs):
        # no inverse transform sampling for you!
        raise NotImplementedError

    @classmethod
    def make_stan_utility_func(cls, fit_index: bool, fit_beta: bool, fit_Enorm: bool):
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
            c_f = 1
            c_d = 1
            if fit_index:
                a = InstantVariableDef("a", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                a = InstantVariableDef("a", "real", [f"x_r[{c_d}]"])
                c_d += 1

            if fit_beta:
                b = InstantVariableDef("b", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                b = InstantVariableDef("b", "real", [f"x_r[{c_d}]"])
                c_d += 1

            ReturnStatement(
                [FunctionCall([(1.0 - a) * x - b * FunctionCall([x, 2], "pow")], "exp")]
            )
        # Same here
        lp = UserDefinedFunction(
            "logparabola_x_dN_dx",
            ["x", "xc", "theta", "x_r", "x_i"],
            ["real", "real", "array[] real", "data array[] real", "data array[] int"],
            "real",
        )
        with lp:
            x = StringExpression(["x"])
            c_f = 1
            c_d = 1
            if fit_index:
                a = InstantVariableDef("a", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                a = InstantVariableDef("a", "real", [f"x_r[{c_d}]"])
                c_d += 1

            if fit_beta:
                b = InstantVariableDef("b", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                b = InstantVariableDef("b", "real", [f"x_r[{c_d}]"])
                c_d += 1

            ReturnStatement(
                [FunctionCall([(2.0 - a) * x - b * FunctionCall([x, 2], "pow")], "exp")]
            )

    @classmethod
    def make_stan_lpdf_func(
        cls, f_name, fit_index: bool, fit_beta: bool, fit_Enorm: bool
    ) -> UserDefinedFunction:
        """
        If fit_beta==True, signature is theta=[alpha, beta], x_r=[E0, Emin, Emax]
        else theta=[alpha, E0], x_r=[beta, Emin, Emax]
        """
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
            c_f = 1
            c_d = 1
            if fit_index:
                a = InstantVariableDef("a", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                a = InstantVariableDef("a", "real", [f"x_r[{c_d}]"])
                c_d += 1

            if fit_beta:
                b = InstantVariableDef("b", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                b = InstantVariableDef("b", "real", [f"x_r[{c_d}]"])
                c_d += 1

            if fit_Enorm:
                E0 = InstantVariableDef("E0", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                E0 = InstantVariableDef("E0", "real", [f"x_r[{c_d}]"])
                c_d += 1

            e_low = InstantVariableDef("e_low", "real", [f"x_r[{c_d}]"])
            c_d += 1
            e_up = InstantVariableDef("e_up", "real", [f"x_r[{c_d}]"])

            E = StringExpression(["E"])

            N = ForwardVariableDef("N", "real")
            p = ForwardVariableDef("p", "real")

            with IfBlockContext([E, ">", e_up]):
                ReturnStatement(["negative_infinity()"])
            with ElseIfBlockContext([E, "<", e_low]):
                ReturnStatement(["negative_infinity()"])

            logEL_E0 = InstantVariableDef("logELE0", "real", ["log(e_low/E0)"])
            logEU_E0 = InstantVariableDef("logEUE0", "real", ["log(e_up/E0)"])
            E_E0 = InstantVariableDef("EE0", "real", ["E/E0"])
            logE_E0 = InstantVariableDef("logEE0", "real", ["log(EE0)"])

            (
                N
                << FunctionCall(
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
                * E0
            )
            p << FunctionCall([E_E0, -a - b * logE_E0], "pow")
            ReturnStatement([FunctionCall([p / N], "log")])

        return func

    @classmethod
    def make_stan_flux_conv_func(
        cls,
        f_name,
        fit_index: bool,
        fit_beta: bool,
        fit_Enorm: bool,
    ) -> UserDefinedFunction:

        func = UserDefinedFunction(
            f_name,
            ["theta", "x_r", "x_i"],
            ["array[] real", "data array[] real", "data array[] int"],
            "real",
        )

        with func:
            theta = StringExpression(["theta"])

            # Unpack variables
            c_f = 1
            c_d = 1
            if fit_index:
                a = InstantVariableDef("a", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                a = InstantVariableDef("a", "real", [f"x_r[{c_d}]"])
                c_d += 1

            if fit_beta:
                b = InstantVariableDef("b", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                b = InstantVariableDef("b", "real", [f"x_r[{c_d}]"])
                c_d += 1

            if fit_Enorm:
                E0 = InstantVariableDef("E0", "real", [f"theta[{c_f}]"])
                c_f += 1
            else:
                E0 = InstantVariableDef("E0", "real", [f"x_r[{c_d}]"])
                c_d += 1

            e_low = InstantVariableDef("e_low", "real", [f"x_r[{c_d}]"])
            c_d += 1
            e_up = InstantVariableDef("e_up", "real", [f"x_r[{c_d}]"])

            f1 = ForwardVariableDef("f1", "real")
            f2 = ForwardVariableDef("f2", "real")

            logEL_E0 = InstantVariableDef("logELE0", "real", ["log(e_low/E0)"])
            logEU_E0 = InstantVariableDef("logEUE0", "real", ["log(e_up/E0)"])

            f1 << FunctionCall(
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
            # Additional factor of E0 due to further transformation of E->log(E/E0)->x
            (
                f2
                << FunctionCall(
                    [
                        "logparabola_x_dN_dx",
                        logEL_E0,
                        logEU_E0,
                        theta,
                        "x_r",
                        "x_i",
                    ],
                    "integrate_1d",
                )
                * E0
            )

            ReturnStatement([f1 / f2])

        return func


class PGammaSpectrum(SpectralShape):
    """
    PGamma spectral shape as derived by https://www.aanda.org/articles/aa/full_html/2024/09/aa50592-24/aa50592-24.html
    Approximated by a flat power law and a logparabola branch at high energies
    """

    _src_index = 0.0
    _alpha = 0.0
    _beta = 0.7
    _name = "pgamma"

    @u.quantity_input
    def __init__(
        self,
        normalisation: Parameter,
        normalisation_energy: Parameter,
        lower_energy: u.GeV = 1e2 * u.GeV,
        upper_energy: u.GeV = np.inf * u.GeV,
        *args,
        **kwargs,
    ):
        """
        :param normalisation: flux normalisation in 1 / energy / area / time
        :param normalisation_energy: energy at which flux is normalised and energy
        at which definition changes from flat to logparabola
        :param lower_energy: lower energy bound of spectrum
        :param upper_energy: upper energy bound of spectrum
        """

        super().__init__()
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy
        # Create hidden parameters requiring less changes to the code generator
        index = Parameter(
            self._src_index, "src_index", fixed=True, par_range=(-1.0, 1.0)
        )
        beta = Parameter(self._beta, "beta_index", fixed=True, par_range=(0.0, 1.0))
        self._parameters = {
            "index": index,
            "beta": beta,
            "norm": normalisation,
            "norm_energy": normalisation_energy,
        }

    @classmethod
    @u.quantity_input
    def _logp(
        cls,
        E: u.GeV,
        E_0: u.GeV,
        N_0: u.Quantity[1 / u.GeV / u.m**2 / u.s],
        alpha: float,
        beta: float,
    ) -> u.Quantity[1 / u.GeV / u.m**2 / u.s]:
        return N_0 * np.power(E / E_0, -alpha - beta * np.log(E / E_0))

    @u.quantity_input
    def __call__(self, E: u.GeV) -> u.Quantity[1 / u.GeV / u.m**2 / u.s]:
        E0 = self.parameters["norm_energy"].value
        norm = self.parameters["norm"].value

        if E.shape != ():
            output = np.zeros_like(E.value) * norm
            output[E >= E0] = self._logp(E[E >= E0], E0, norm, self._alpha, self._beta)

            output[E < E0] = norm

            mask = np.nonzero(((E > self._upper_energy) ^ (E < self._lower_energy)))
            output[mask] = 0.0 * norm
            return output
        if (E < self._lower_energy) or (E > self._upper_energy):
            return 0.0 * norm

        if E >= E0:
            return self._logp(E, E0, norm, self._alpha, self._beta)
        return norm

    @u.quantity_input
    def integral(self, lower: u.GeV, upper: u.GeV) -> u.Quantity[1 / (u.m**2 * u.s)]:
        # Integral of logparabola part for fixed alpha=0 and beta=0.7 (or otherwise fixed)
        # can be solved in a closed form as
        # sqrt(pi) * e^(1/(4beta)) * E0 / (2 * sqrt(beta)) * erf((2beta*log(E/E0) - 1) / (2 * sqrt(beta)))
        E0 = self.parameters["norm_energy"].value
        norm = self.parameters["norm"].value

        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        xl = np.log((lower / E0).to_value(1))
        xh = np.log((upper / E0).to_value(1))

        # Check edge cases
        lower[((lower < self._lower_energy) & (upper > self._lower_energy))] = (
            self._lower_energy
        )
        upper[((lower < self._upper_energy) & (upper > self._upper_energy))] = (
            self._upper_energy
        )

        results = np.zeros(xl.shape) << 1 / u.m**2 / u.s
        for c in range(xl.size):
            # Check if domain of integration is completely within on or the other definition
            if xh[c] <= 0.0:
                results[c] = norm * (upper[c] - lower[c])
            elif xl[c] >= 0.0:
                results[c] = (
                    (np.sqrt(np.pi) * np.exp(1 / (4.0 * self._beta)) * E0)
                    / (2.0 * np.sqrt(self._beta))
                    * (
                        erf(
                            (2.0 * self._beta * xh[c] - 1.0)
                            / (2.0 * np.sqrt(self._beta))
                        )
                        - erf(
                            (2.0 * self._beta * xl[c] - 1.0)
                            / (2.0 * np.sqrt(self._beta))
                        )
                    )
                ) * norm
            else:
                # from Elow to Ebreak: PL, from Ebreak to Ehigh: logparabola
                pl = norm * (E0 - lower[c])
                logp = (
                    (np.sqrt(np.pi) * np.exp(1 / (4.0 * self._beta)) * E0)
                    / (2.0 * np.sqrt(self._beta))
                    * (
                        erf(
                            (2.0 * self._beta * xh[c] - 1.0)
                            / (2.0 * np.sqrt(self._beta))
                        )
                        - erf(-1.0 / (2.0 * np.sqrt(self._beta)))
                    )
                ) * norm
                results[c] = pl + logp

        if results.size == 1:
            return results[0]
        return results

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.Quantity[u.erg / u.s / u.m**2]:
        E0 = self.parameters["norm_energy"].value
        norm = self.parameters["norm"].value

        pl = norm * (E0**2 - self._lower_energy**2) / 2

        xh = np.log((self._upper_energy / E0).to_value(1))

        result = (
            (np.sqrt(np.pi) * np.exp(1 / self._beta) * E0**2)
            / (2.0 * np.sqrt(self._beta))
            * (
                erf((self._beta * xh - 1.0) / np.sqrt(self._beta))
                - erf(-1.0 / np.sqrt(self._beta))
            )
        ) * norm
        return result + pl

    def flux_conv(self):
        # Calculate (\int dN / dE / dA /dt dE)/(\int E dN / dE / dA / dt dE)

        return (
            self.integral(self._lower_energy, self._upper_energy)
            / self.total_flux_density
        ).to(1 / u.GeV)

    @classmethod
    def flux_conv_(cls, **kwargs):
        # Stitch together from power law and logparabola
        # NB: f1 and f2 have to be each stitched together, not per domain!
        # Misnormer, e_0 is break energy and norm energy

        e_0 = kwargs["e_0"]
        e_low = kwargs["e_low"]
        e_up = kwargs["e_up"]
        if not e_0 > e_low:
            e_0 = e_low

        if not e_0 < e_up:
            e_0 = e_up

        xl = np.log(e_low / e_0)
        xh = np.log(e_up / e_0)

        f1_pl = e_0 - e_low
        f1_logp = (
            (np.sqrt(np.pi) * np.exp(1 / (4.0 * cls._beta)) * e_0)
            / (2.0 * np.sqrt(cls._beta))
            * (
                erf((2.0 * cls._beta * xh - 1.0) / (2.0 * np.sqrt(cls._beta)))
                - erf(-1.0 / (2.0 * np.sqrt(cls._beta)))
            )
        )

        f2_pl = 1 / 2 * (e_0**2 - e_low**2)
        f2_logp = (
            (np.sqrt(np.pi) * np.exp(1 / cls._beta) * e_0**2)
            / (2.0 * np.sqrt(cls._beta))
            * (
                erf((cls._beta * xh - 1.0) / np.sqrt(cls._beta))
                - erf(-1.0 / np.sqrt(cls._beta))
            )
        )

        return (f1_pl + f1_logp) / (f2_pl + f2_logp)

    @property
    def parameters(self):
        return self._parameters

    @u.quantity_input
    def pdf(self, E: u.GeV, Emin: u.GeV, Emax: u.GeV, apply_lim: bool = True):
        """
        Return PDF.
        """

        E_input = np.atleast_1d(E)

        norm = self.integral(*self.energy_bounds)

        pdf = (self.__call__(E_input) / norm).to_value(1 / u.GeV)
        if apply_lim:
            pdf[E_input < Emin] = 0.0
            pdf[E_input > Emax] = 0.0
        if pdf.size == 1:
            return pdf[0]
        return pdf

    @classmethod
    def make_stan_sampling_lpdf_func(cls, f_name) -> UserDefinedFunction:
        return cls.make_stan_lpdf_func(f_name, False, False, False)

    @classmethod
    def make_stan_sampling_func(cls, f_name, *args, **kwargs):
        # no inverse transform sampling for you!
        raise NotImplementedError

    @classmethod
    def make_stan_utility_func(cls, fit_index: bool, fit_beta: bool, fit_Enorm: bool):
        pass

    @classmethod
    def make_stan_lpdf_func(
        cls, f_name, fit_index: bool, fit_beta: bool, fit_Enorm: bool
    ) -> UserDefinedFunction:

        func = UserDefinedFunction(
            f_name,
            ["E", "theta", "x_r", "x_i"],
            ["real", "array[] real", "data array[] real", "data array[] int"],
            "real",
        )

        with func:
            E = StringExpression(["E"])
            c = 1
            if fit_Enorm:
                E0 = InstantVariableDef("E0", "real", ["theta[1]"])
            else:
                E0 = InstantVariableDef("E0", "real", [f"x_r[{c}]"])
                c += 1
            b = InstantVariableDef("b", "real", [cls._beta])
            sqrt_b = InstantVariableDef("sqrt_b", "real", [np.sqrt(cls._beta)])
            sqrt_b_inv_half = InstantVariableDef(
                "sqrt_b_inv_half", "real", [0.5 / np.sqrt(cls._beta)]
            )
            e_low = InstantVariableDef("e_low", "real", [f"x_r[{c}]"])
            c += 1
            e_up = InstantVariableDef("e_up", "real", [f"x_r[{c}]"])
            logEL_E0 = InstantVariableDef("logELE0", "real", ["log(e_low/E0)"])
            logEU_E0 = InstantVariableDef("logEUE0", "real", ["log(e_up/E0)"])
            E_E0 = InstantVariableDef("EE0", "real", ["E/E0"])
            logE_E0 = InstantVariableDef("logEE0", "real", ["log(EE0)"])
            prefactor = InstantVariableDef(
                "prefactor",
                "real",
                [
                    np.sqrt(np.pi)
                    * np.exp(1.0 / (4 * cls._beta))
                    / (2.0 * np.sqrt(cls._beta))
                ],
            )

            N = ForwardVariableDef("N", "real")
            N_logp = ForwardVariableDef("N_logp", "real")
            N_pl = ForwardVariableDef("N_pl", "real")
            p = ForwardVariableDef("p", "real")

            with IfBlockContext([E, ">", e_up]):
                ReturnStatement(["negative_infinity()"])
            with ElseIfBlockContext([E, "<", e_low]):
                ReturnStatement(["negative_infinity()"])

            N_logp << StringExpression(
                ["prefactor * E0 * (erf(sqrt_b*logEUE0-0.5) - erf(-sqrt_b_inv_half))"]
            )
            N_pl << E0 - e_low

            N << N_logp + N_pl
            # Decide in which part of the spectrum the energy E falls
            # return accordingly lpdf taken from that definition
            with IfBlockContext([E, " >= ", E0]):
                p << FunctionCall([E_E0, -b * logE_E0], "pow")
            with ElseBlockContext():
                p << 1
            ReturnStatement([FunctionCall([p / N], "log")])

        return func

    @classmethod
    def make_stan_flux_conv_func(
        cls,
        f_name,
        fit_index: bool,
        fit_beta: bool,
        fit_Enorm: bool,
    ) -> UserDefinedFunction:

        func = UserDefinedFunction(
            f_name,
            ["theta", "x_r", "x_i"],
            ["array[] real", "data array[] real", "data array[] int"],
            "real",
        )

        with func:

            c_d = 1
            # theta = StringExpression(["theta"])
            if fit_Enorm:
                E0 = InstantVariableDef("E0", "real", ["theta[1]"])
            else:
                E0 = InstantVariableDef("E0", "real", [f"x_r[{c_d}]"])
                c_d += 1
            b = InstantVariableDef("b", "real", [cls._beta])
            sqrt_b = InstantVariableDef("sqrt_b", "real", [np.sqrt(cls._beta)])
            sqrt_b_inv_half = InstantVariableDef(
                "sqrt_b_inv_half", "real", [0.5 / np.sqrt(cls._beta)]
            )
            e_low = InstantVariableDef("e_low", "real", [f"x_r[{c_d}]"])
            c_d += 1
            e_up = InstantVariableDef("e_up", "real", [f"x_r[{c_d}]"])
            logEL_E0 = InstantVariableDef("logELE0", "real", ["log(e_low/E0)"])
            logEU_E0 = InstantVariableDef("logEUE0", "real", ["log(e_up/E0)"])
            f1_prefactor = InstantVariableDef(
                "f1_prefactor",
                "real",
                [
                    np.sqrt(np.pi)
                    * np.exp(1.0 / (4 * cls._beta))
                    / (2.0 * np.sqrt(cls._beta))
                ],
            )
            f2_prefactor = InstantVariableDef(
                "f2_prefactor",
                "real",
                [np.sqrt(np.pi) * np.exp(1.0 / cls._beta) / (2.0 * np.sqrt(cls._beta))],
            )

            N_pl = ForwardVariableDef("N_pl", "real")

            f1 = ForwardVariableDef("f1", "real")
            f2 = ForwardVariableDef("f2", "real")

            f1 << StringExpression(
                [
                    "f1_prefactor * E0 * (erf(sqrt_b*logEUE0-0.5) - erf(-sqrt_b_inv_half))"
                ]
            )
            N_pl << E0 - e_low
            f1 << f1 + N_pl
            # Additional factor of E0 due to further transformation of E->log(E/E0)->x

            f2 << StringExpression(
                ["f2_prefactor * E0^2 * (erf((b*logEUE0-1.)/sqrt_b) - erf(-1./sqrt_b))"]
            )

            N_pl << 0.5 * (E0**2 - e_low**2)

            f2 << f2 + N_pl
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
