"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod
from typing import Union, List, Iterable
import numpy as np

from stan_generator import StanCodeBit, TListStrStanCodeBit


class StanExpression(metaclass=ABCMeta):
    """
    Generic Stan expression

    The expression can depend on inputs, such that it's possible to
    chain StanExpressions in a graph like manner
    """

    def __init__(self, inputs: List["TStanable"]):
        self.inputs = inputs

    @abstractmethod
    def to_stan(self) -> StanCodeBit:
        """
        Converts the expression into a StanCodeBit
        """
        pass


class PyMCExpression(metaclass=ABCMeta):
    def __init__(self, inputs):
        self.inputs = inputs

    @abstractmethod
    def to_pymc(self):
        pass


# Define type union for stanable types
TStanable = Union[StanExpression, str, float]


class Parametrization(StanExpression,
                      PyMCExpression,
                      metaclass=ABCMeta):
    """
    Base class for parametrizations.

    Parametrizations are functions of a given input variable.
    These can be splines, distributions, ...
    Comes with a converter to stan code / pymc3 variables.
    """

    def __init__(self, inputs: List[TStanable]):
        self._inputs = inputs

    @abstractmethod
    def to_stan(self) -> StanCodeBit:
        """
        Convert the parametrization to Stan
        """
        pass

    @abstractmethod
    def to_pymc(self):
        """
        Convert the parametrizaton to PyMC3
        """
        pass


def stanify(var: TStanable) -> StanCodeBit:
    """Return call to to_stan function if possible"""
    if isinstance(var, StanExpression):
        return var.to_stan()
    code_bit = StanCodeBit()
    code_bit.add_code([str(var)])
    return code_bit


class LogParametrization(Parametrization):
    """log with customizable base"""
    def __init__(self, inputs: TStanable, base: float = 10):
        Parametrization.__init__(self, [inputs])
        self._base = base

    def to_stan(self) -> StanCodeBit:
        """See base class"""

        x_eval_stan = stanify(self._inputs[0])
        stan_code: TListStrStanCodeBit = []
        if self._base != 10:
            base = str(self._base)
            stan_code += ["log10(", x_eval_stan, "}) / log10(", base, ")"]
        else:
            stan_code += ["log10(", x_eval_stan, "})"]

        stan_code_bit = StanCodeBit()
        stan_code_bit.add_code(stan_code)

        return stan_code_bit

    def to_pymc(self):
        pass


TArrayOrNumericIterable = Union[np.ndarray, Iterable[float]]


class PolynomialParametrization(Parametrization):

    def __init__(
            self,
            inputs: TStanable,
            coefficients: TArrayOrNumericIterable) -> None:

        Parametrization.__init__(self, [inputs])
        self._coeffs = coefficients

    def to_stan(self) -> StanCodeBit:
        """See base class"""

        x_eval_stan = stanify(self._inputs[0])

        # TODO: Make sure that eval_poly1d is part of some util lib.
        # Or maybe add a hook for loading ?

        coeffs_stan = "[" + ",".join([str(coeff)
                                      for coeff in self._coeffs]) + "]"

        stan_code: TListStrStanCodeBit = [
            "eval_poly1d(",
            x_eval_stan,
            "), ",
            coeffs_stan]

        stan_code_bit = StanCodeBit()
        stan_code_bit.add_code(stan_code)
        return stan_code_bit

    def to_pymc(self):
        pass


class LognormalParametrization(Parametrization):

    def __init__(self, inputs: TStanable, mu: float, sigma: float):
        Parametrization.__init__(self, [inputs])
        self._mu = mu
        self._sigma = sigma

    def __call__(self, x):
        pass

    def to_stan(self) -> StanCodeBit:
        mu_stan = stanify(self._mu)
        sigma_stan = stanify(self._sigma)
        x_obs_stan = stanify(self._inputs[0])

        stan_code: TListStrStanCodeBit = []
        stan_code += ["lognormal_lpdf(", x_obs_stan, " | ", mu_stan, ", ",
                      sigma_stan, ")"]

        stan_code_bit = StanCodeBit()
        stan_code_bit.add_code(stan_code)

        return stan_code_bit

    def to_pymc(self):
        pass


class EffectiveArea(metaclass=ABCMeta):
    """
    Implements baseclass for effective areas.


    Every implementation of an effective area has to define a setup method,
    that will take care of downloading required files, creating splines, etc.

    The effective areas can depend on multiple quantities (ie. energy,
    direction, time, ..)
    """

    """
    Parameters on which the effective area depends.
    Overwrite when subclassing
    """
    PARAMETERS = None

    def __call__(self, **kwargs):
        """
        Return the effective area for variables given in kwargs
        """
        if (set(self.PARAMETERS) - kwargs.keys()):
            raise ValueError("Not all required parameters passed to call")
        else:
            self._calc_effective_area(kwargs)

    @abstractmethod
    def _calc_effective_area(
            self,
            param_dict: dict) -> float:
        pass

    @abstractmethod
    def setup(self) -> None:
        """
        Download and or build all the required input data for calculating
        the effective area
        """
        pass


class Resolution(metaclass=ABCMeta):
    """Base class for parametrizing resolutions"""

    PARAMETERS = None

    def __call__(self, **kwargs):
        """
        Return the resolution for variables given in kwargs
        """
        return self._calc_resolution(kwargs)

    @abstractmethod
    def _calc_resolution(
            self,
            param_dict: dict):
        pass

    @abstractmethod
    def setup(self):
        """
        Download and or build all the required input data for calculating
        the resolution
        """
        pass


class NorthernTracksEffectiveArea(EffectiveArea):
    """
    Effective area for the two-year Northern Tracks release:
    https://icecube.wisc.edu/science/data/HE_NuMu_diffuse

    """

    PARAMETERS = ["true_energy", "true_cos_zen"]

    def _calc_effective_area(
            self,
            param_dict: dict):
        pass

    def setup(self):
        """See base class"""


class NorthernTracksEnergyResolution(Resolution):
    PARAMETERS = ["true_energy"]  # neglect zenith dependence

    def _calc_resolution(
            self,
            param_dict: dict):
        pass


class NorthernTracksAngularResolution(Resolution):
    PARAMETERS = ["true"]


class DetectorModel(metaclass=ABCMeta):

    @property
    def effective_area(self):
        return self._get_effective_area()

    @abstractmethod
    def _get_effective_area(self):
        return self.__get_effective_area

    @property
    def energy_resolution(self):
        return self._get_energy_resolution()

    @abstractmethod
    def _get_energy_resolution(self):
        return self._energy_resolution

    @property
    def angular_resolution(self):
        return self._get_angular_resolution()

    @abstractmethod
    def _get_angular_resolution(self):
        self._angular_resolution


if __name__ == "__main__":

    invar = "E_true"
    log_e_eval = LogParametrization(invar)
    test_poly_coeffs = [1, 1, 1, 1]
    param = PolynomialParametrization(log_e_eval, test_poly_coeffs)

    invar = "E_reco"
    lognorm = LognormalParametrization(invar, param, param)
    print(lognorm.to_stan())
