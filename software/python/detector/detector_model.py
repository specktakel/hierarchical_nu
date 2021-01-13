"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod
from typing import Sequence

from ..cache import Cache
from ..backend import (
    Expression,
    TExpression,
    TListTExpression,
    UserDefinedFunction,
    DistributionMode,
)

import logging

logger = logging.getLogger(__name__)
Cache.set_cache_dir(".cache")


class EffectiveArea(UserDefinedFunction, metaclass=ABCMeta):
    """
    Implements baseclass for effective areas.

    Every implementation of an effective area has to define a setup method,
    that will take care of downloading required files, creating parametrizations etc.

    The effective areas can depend on multiple quantities (ie. energy,
    direction, time, ..)
    """

    @abstractmethod
    def setup(self) -> None:
        """
        Download and or build all the required input data for calculating
        the effective area. Alternatively load from necessary info from
        cache if already stored.

        Setup must provide the info to fill the properties listed below,
        and be called in the initialisation.
        """

        pass

    @property
    def eff_area(self):
        """
        2D histogram of effective area values, with
        Etrue on axis 0 and cosz on the axis 1.
        """

        return self._eff_area

    @property
    def tE_bin_edges(self):
        """
        True energy bin edges corresponding the the
        histogram in eff_area.
        """

        return self._tE_bin_edges

    @property
    def cosz_bin_edges(self):
        """
        cos(zenith) bin edges corresponding to the
        histogram in eff_area.
        """

        return self._cosz_bin_edges


class Resolution(Expression, metaclass=ABCMeta):
    """
    Base class for parameterizing resolutions
    """

    def __init__(self, inputs: Sequence[TExpression], stan_code: TListTExpression):
        Expression.__init__(self, inputs, stan_code)

    def __call__(self, **kwargs):
        """
        Return the resolution for variables given in kwargs
        """
        return self._calc_resolution(kwargs)

    @abstractmethod
    def _calc_resolution(self, param_dict: dict):
        pass

    @abstractmethod
    def setup(self):
        """
        Download and or build all the required input data for calculating
        the resolution
        """
        pass


class DetectorModel(metaclass=ABCMeta):
    def __init__(self, mode: DistributionMode = DistributionMode.PDF):
        self._mode = mode

    @property
    def effective_area(self):
        return self._get_effective_area()

    @abstractmethod
    def _get_effective_area(self):
        return self.__get_effective_area()

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
