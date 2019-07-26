"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod


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

