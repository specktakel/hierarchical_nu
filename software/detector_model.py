"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod
from typing import Type, Union
import numpy as np

TStanable = Union[Type[Parametrization], str]

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
            param_dict: dict) ->float:
        pass

    @abstractmethod
    def setup(self) -> None:
        """
        Download and or build all the required input data for calculating
        the effective area
        """
        pass


class Parametrization(metaclass=ABCMeta):
    """
    Base class for parametrizations.

    Comes with a converter to stan code
    """

    @abstractmethod
    def to_stan(self) -> str:
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


def stanify(var: TStanable) -> str:
    if issubclass(var, Parametrization):
        return var.to_stan()
    return var


class LogParametrization(Parametrization):
    def __init__(self, base: float):
        self._base = base


    def to_stan(self, x_eval: TStanable) -> str:
        """See base class"""

        x_eval_stan = stanify(x_eval)
        base_stan = "log({x_eval}) / log({base});"


class PolynomialParametrization(Parametrization):

    def __init__(self, coefficients: np.ndarray) -> None:
        self._coeffs = coefficients

    def to_stan(self, x_eval: TStanable) -> str:
        """See base class"""

        x_eval_stan = stanify(x_eval)

        # TODO: Make sure that eval_poly1d is part of some util lib.
        # Or maybe add a hook for loading ?

        coeffs_stan = "[" + [str(coeff)+"," if i != len(self._coeffs)
                             else str(coeff)
                             for i, coeff in enumerate(self._coeffs)] + "]"

        stan_code = "eval_poly1d({x_eval}, {coeffs});"
        stan_code = stan_code.format(x_eval_stan, coeffs_stan)

        return stan_code


class LognormalParametrization:

    def __init__(self, mu, sigma):

        self._mu = mu
        self._sigma = sigma

    def __call__(self, x):
        pass

    def to_stan(self, x_obs):
        if issubclass(self._mu, Parametrization):
            mu_stan = self._mu.to_stan()
        else:
            mu_stan = str(self._mu)

        if issubclass(self._sigma, Parametrization):
            sigma_stan = self._sigma.to_stan()
        else:
            sigma_stan = str(self._sigma)

        if issubclass(x_obs, Parametrization):
            x_obs_stan = x_obs.to_stan()
        else:
            x_obs_stan = str(x_obs)

        stan_code = (
            """lognormal_lpdf({x_obs_name} | {mu_stan}, {sigma_stan})""")
        stan_code.format(x_obs_name=x_obs_stan, mu_stan=mu_stan,
                         sigma_stan=sigma_stan)


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
