from typing import Tuple
from enum import Enum
import numpy as np

ParScale = Enum("ParScale", "lin log cos")


class Parameter:
    __par_registry = {}

    def __init__(
            self,
            value,
            name: str,
            fixed=False,
            par_range=(-np.inf, np.inf),
            scale=ParScale.lin):

        if name in Parameter.__par_registry:
            self.__dict__ = self.__par_registry[name].__dict__
        else:
            self._value = value
            self._fixed = fixed
            self._par_range = par_range
            self._scale = scale
            self._name = name
            Parameter.__par_registry[name] = self

    @classmethod
    def get_parameter(cls, par_name):
        if par_name not in cls.__par_registry:
            print(cls.__par_registry)
            raise ValueError("Parameter {} not found".format(par_name))
        return cls.__par_registry[par_name]

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, fixed: bool):
        self._fixed = fixed

    @property
    def par_range(self):
        return self._par_range

    @par_range.setter
    def par_range(self, par_range: Tuple[float, float]):
        self._par_range = par_range
