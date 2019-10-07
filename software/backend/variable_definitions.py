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

class ForwardVariableDef(VariableDef):
    """Define variable without assigning value"""

    def __init__(self, name: str, var_type: str) -> None:
        VariableDef.__init__(self, name)
        self._var_type = var_type
        self.add_stan_hook(name, "var_def", self.def_code)

    @property
    def def_code(self) -> StanDefCode:
        """See parent class"""
        stan_code = self._var_type + " " + self.name +";\n"
        def_code = StanDefCode(self.name)
        def_code.add_def_code([stan_code])
        return def_code


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
        self.add_stan_hook(name, "var_def", self.def_code)

    @property
    def def_code(self) -> StanDefCode:
        """
        See parent class
        """

        # Variable Definition
        stan_code = self._type

        shape_str = "[" + ",".join([str(shape_d)
                                    for shape_d in self._array_data.shape])
        shape_str += "]"
        if self._type == "vector":
            stan_code += shape_str + " " + self.name

        else:
            stan_code += " " + self.name + shape_str

        # Fill array
        arraystr = np.array2string(
            self._array_data,
            threshold=np.inf,
            separator=",")
        if self._type != "vector":
            arraystr = arraystr.replace("[", "{")
            arraystr = arraystr.replace("]", "}")
        stan_code += " = " + arraystr 
        if self._type == "vector":
            stan_code += "'" # FU Stan
        stan_code += "; \n"
        def_code = StanDefCode(self.name)
        def_code.add_def_code([stan_code])
        return def_code
