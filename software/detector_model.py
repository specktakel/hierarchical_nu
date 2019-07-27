"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod
from typing import Union, List
import os
import pandas as pd
import numpy as np

from cache import Cache
from parameterizations import (
    Parameterization,
    TStanable,
    VMFParameterization,
    PolynomialParameterization,
    TruncatedParameterization)


Cache.set_cache_dir(".cache")


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
    PARAMETERS: Union[None, List] = None

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


class Resolution(Parameterization, metaclass=ABCMeta):
    """Base class for parametrizing resolutions"""

    PARAMETERS: Union[None, List] = None

    def __init__(self, inputs: List[TStanable]):
        Parameterization.__init__(self, inputs)

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


class NorthernTracksAngularResolution(VMFParameterization, Resolution):  # type: ignore
    """
    Angular resolution for Northern Tracks Sample

    Data from https://arxiv.org/pdf/1811.07979.pdf
    Fits a polynomial to the median angular resolution converted to
    `kappa` parameter of a VMF distribution

    Attributes:
        poly_params: Coefficients of the polynomial
        e_min: Lower energy bound of the polynomial
        e_max: Upper energy bound of the polynomial

    """
    PARAMETERS = ["true"]
    DATA_PATH = "NorthernTracksAngularRes.csv"
    CACHE_FNAME = "angular_reso_tracks.npz"

    def __init__(self, inputs: List[TStanable]) -> None:
        """

        Args:
            inputs: List[TStanable]
                First item is true energy, second item is true
                position
        """
        Resolution.__init__(self, inputs)
        self.poly_params: Union[None, np.ndarray] = None
        self.e_min: Union[None, float] = None
        self.e_max: Union[None, float] = None

        self.setup()

        VMFParameterization.__init__(self, inputs, self._kappa)

    def _calc_resolution(self):
        pass

    def setup(self) -> None:
        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                data = np.load(fr)
                self.poly_params = data["poly_params"]
                self.e_min = float(data["e_min"])
                self.e_max = float(data["e_max"])
        else:
            if not os.path.exists(self.DATA_PATH):
                raise RuntimeError(self.DATA_PATH, "is not a valid path")

            data = pd.read_csv(
                self.DATA_PATH,
                sep=";",
                decimal=",",
                header=None,
                names=["energy", "resolution"],
                )

            # Kappa parameter of VMF distribution
            data["kappa"] = 1.38 / np.radians(data.resolution)**2

            self.poly_params = np.polyfit(data.energy, data.kappa, 5)
            self.e_min = float(data.energy.min())
            self.e_max = float(data.energy.max())

            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                np.savez(
                    fr,
                    poly_params=self.poly_params,
                    e_min=10**data.energy.min(),
                    e_max=10**data.energy.max())

        # Clip true energy
        clipped_e = TruncatedParameterization(
            self._inputs[0],
            self.e_min,
            self.e_max)

        self._kappa = PolynomialParameterization(
            clipped_e,
            self.poly_params)


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

    e_true = "E_true"
    pos_true = "pos_true"
    ntp = NorthernTracksAngularResolution([e_true, pos_true])

    print(ntp.to_stan())
