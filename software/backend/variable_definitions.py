from abc import abstractmethod
from typing import Iterable
import numpy as np  # type:ignore
from .expression import (
    NamedExpression, StanDefCode, TListTExpression, TExpression,
    Expression)
from .stan_generator import DefinitionContext
import logging
import re
logger = logging.getLogger(__name__)

__all__ = ["VariableDef", "StanArray", "ForwardArrayDef",
           "ForwardVariableDef"]


class VariableDef(NamedExpression):
    """
    Stan variable definition

    """
    def __init__(
            self,
            name: str):
        NamedExpression.__init__(self, [], name)

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

    def _gen_def_code(self) -> TListTExpression:
        """See parent class"""
        """
        match = re.match(r"(real|int)\[(\d+)\]", self._var_type)
        if match is not None:
            # Refactor so that array size comes after name
            stan_code = match.groups()[0] + " "
            stan_code += self.name + "["+ match.groups()[1] + "]"
        else:
            stan_code = self._var_type + " " + self.name
        """
        return [self._var_type + " " + self.name]


class ForwardArrayDef(VariableDef):
    """Define an array of variables"""

    def __init__(
            self,
            name: str,
            var_type: str,
            array_dim: TListTExpression) -> None:
        VariableDef.__init__(self, name)
        self._var_type = var_type
        self._array_dim = array_dim
        for expr in self._array_dim:
            if isinstance(expr, Expression):
                expr.add_output(self)

        self.def_code()

    def _gen_def_code(self) -> TListTExpression:
        return [self._var_type + " " + self.name] + self._array_dim


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
        stan_code += ""

        return [stan_code]


if __name__ == "__main__":
    from .stan_generator import StanGenerator, GeneratedQuantitiesContext
    from .operations import AssignValue

    logging.basicConfig(level=logging.DEBUG)
    with StanGenerator() as cg:
        with GeneratedQuantitiesContext() as gq:
            val = ForwardVariableDef("a", "real")
            _ = AssignValue([val], "b")

        print(cg.generate())
