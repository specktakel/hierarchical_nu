from abc import abstractmethod
from typing import Iterable
from .expression import Expression, StanDefCode, TListTExpression
from .stan_code import TStrStanCodeBit
from .stan_generator import DefinitionContext
import logging
logger = logging.getLogger(__name__)
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

    def def_code(self) -> None:
        code = self._gen_def_code()
        with DefinitionContext() as _:
            def_code = StanDefCode()
            def_code.add_def_code(code)

    @abstractmethod
    def _gen_def_code(self) -> TListTExpression:
        pass

    @property
    def stan_code(self) -> TListTExpression:
        return [self._name]

    def to_pymc(self):
        pass


class ForwardVariableDef(VariableDef):
    """Define variable without assigning value"""

    def __init__(self, name: str, var_type: str) -> None:
        VariableDef.__init__(self, name)
        self._var_type = var_type
        self.def_code()
        #self.add_stan_hook(name, "var_def", self.def_code)

    def _gen_def_code(self) -> TListTExpression:
        """See parent class"""
        stan_code = self._var_type + " " + self.name + ";\n"

        return [stan_code]


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
        self.def_code()

    def _gen_def_code(self) -> TListTExpression:
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
            stan_code += "'"  # FU Stan
        stan_code += "; \n"

        return [stan_code]


if __name__ == "__main__":
    from .stan_generator import StanGenerator, GeneratedQuantitiesContext
    from .operations import AssignValue

    logging.basicConfig(level=logging.DEBUG)
    with StanGenerator() as cg:
        with GeneratedQuantitiesContext() as gq:
            val = ForwardVariableDef("a", "real")
            val = AssignValue([val], "b")


        print(cg.generate())
