from typing import Tuple, Any
from enum import Enum
import numpy as np

ParScale = Enum("ParScale", "lin log cos")


class Parameter:
    """
    Parameter class

    Parameters with the same name share an internal state

    Parameters:
        value: Any
        name: str
            Parameter name. Parameters of the same name share an internal state.
        fixed: bool
            If set to true, value cannot be changed
        par_range: Optional[Tuple[float, float]]
            Parameter range. Will be used as check when setting the parameter value
        scale: ParScale
            Parameter scale
    """

    __par_registry = {}

    def __init__(
        self,
        value: Any = np.nan,
        name: str = "",
        fixed: bool = False,
        par_range=(-np.inf, np.inf),
        scale=ParScale.lin,
    ):
        # If name is registered, copy internal state
        if name in Parameter.__par_registry:
            self.__dict__ = self.__par_registry[name].__dict__
        else:
            self._value = value
            self._initial_val = value
            self._fixed = fixed
            self._par_range = par_range
            self._scale = scale
            self._name = name
            Parameter.__par_registry[name] = self

    @classmethod
    def get_parameter(cls, par_name):
        if par_name not in cls.__par_registry:
            # print(cls.__par_registry)
            raise ValueError("Parameter {} not found".format(par_name))
        return cls.__par_registry[par_name]

    @classmethod
    def remove_parameter(cls, par_name):
        if par_name not in cls.__par_registry:
            raise ValueError("Parameter {} not found".format(par_name))
        del cls.__par_registry[par_name]

    @classmethod
    def clear_registry(cls):
        """Clear the parameter registry"""
        cls.__par_registry = {}

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
        if self._par_range is not None:
            if not self._par_range[0] <= value <= self._par_range[1] and not np.isnan(
                value
            ):
                raise ValueError("Parameter {} out of bounds".format(self.name))
        if self.fixed:
            raise RuntimeError("Parameter {} is fixed".format(self.name))
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

    def __repr__(self):
        return "Parameter {} = {}, fixed = {}".format(self.name, self.value, self.fixed)

    def reset(self):
        """Reset value to initial val"""
        self.value = self._initial_val

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            raise ValueError

        if (
            self.value == other.value
            and np.all(self.par_range == other.par_range)
            and self.fixed == other.fixed
            and self.scale == other.scale
        ):
            return True
        else:
            return False
