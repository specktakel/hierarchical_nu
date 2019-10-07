from abc import abstractmethod
from typing import Iterable
from .expression import Expression, StanDefCode
from .stan_code import TListStrStanCodeBit
import numpy as np  # type:ignore

__all__ = ["VariableDef", "StanArray"]


class VariableDef(Expression):
    """
    Stan variable definition

    """

    def __init__(
            self,
            name: str):
        Expression.__init__(self, [])
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def def_code(self) -> StanDefCode:
        pass

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        return [self._name]

    def to_pymc(self):
        pass


class StanArray(VariableDef):
    """
    Stan real array definition

    Parameters:
        name: Variable name to use in stan code
    """

    def __init__(
            self,
            name: str,
            type_name: str,
            array_data: Iterable):
        VariableDef.__init__(self, name)
        self._array_data = np.asarray(array_data)
        self._type = type_name
        self.add_stan_hook(name, "var_def", [self.def_code.def_code])

    @property
    def def_code(self) -> StanDefCode:
        """
        See parent class
        """

        # Variable Definition
        stan_code = self._type + " " + self._name
        for shape_d in self._array_data.shape:
            stan_code += "[" + str(shape_d) + "]"

        # Fill array
        arraystr = np.array2string(
            self._array_data,
            threshold=np.inf,
            separator=",")
        arraystr = arraystr.replace("[", "{")
        arraystr = arraystr.replace("]", "}")
        stan_code += " = " + arraystr + "; \n"
        def_code = StanDefCode(self.name)
        def_code.add_def_code([stan_code])
        return def_code
