from abc import abstractmethod
from typing import Iterable, Union
import numpy as np  # type:ignore
from .expression import NamedExpression, TListTExpression, Expression, PlainStatement
from .stan_generator import DefinitionContext
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "VariableDef",
    "StanArray",
    "StanVector",
    "ForwardArrayDef",
    "ForwardVariableDef",
    "ForwardVectorDef",
    "InstantVariableDef",
]


class VariableDef(NamedExpression):
    """
    Stan variable definition

    """

    def __init__(self, name: str):

        NamedExpression.__init__(self, [], [name], name)

        # Add None output. This will supress the generation of the variable
        # name if the variable is not used in any expression
        self.add_output(None)
        code = self._gen_def_code()
        with DefinitionContext() as _:
            Expression([], code)

    @abstractmethod
    def _gen_def_code(self) -> TListTExpression:
        pass


class ForwardVariableDef(VariableDef):
    """Define variable without assigning value"""

    def __init__(self, name: str, var_type: str) -> None:
        self._var_type = var_type
        VariableDef.__init__(self, name)

    def _gen_def_code(self) -> TListTExpression:
        """See parent class"""

        return [self._var_type + " " + self.name]


class InstantVariableDef(VariableDef):
    """Define variable and immediately assign a value"""

    def __init__(self, name: str, var_type: str, value: TListTExpression) -> None:
        self._var_type = var_type
        self._value = value
        VariableDef.__init__(self, name)

    def _gen_def_code(self) -> TListTExpression:
        """See parent class"""

        return [self._var_type + " " + self.name + " = "] + self._value


class ParameterDef(ForwardVariableDef):
    """Define parameters"""

    def __init__(
        self, name: str, var_type: str, lower_bound=None, upper_bound=None
    ) -> None:

        if isinstance(lower_bound, float) or isinstance(lower_bound, int):
            self._lower_bound = "lower=" + str(lower_bound)
        elif isinstance(lower_bound, ForwardVariableDef):
            self._lower_bound = "lower=" + lower_bound.name
        else:
            self._lower_bound = lower_bound

        if isinstance(upper_bound, float) or isinstance(upper_bound, int):
            self._upper_bound = "upper=" + str(upper_bound)
        elif isinstance(upper_bound, ForwardVariableDef):
            self._upper_bound = "upper=" + upper_bound.name
        else:
            self._upper_bound = upper_bound

        ForwardVariableDef.__init__(self, name, var_type)

    def _gen_bound_str(self):

        if not self._lower_bound:
            bound = "<" + self._upper_bound + "> "
        if not self._upper_bound:
            bound = "<" + self._lower_bound + "> "
        else:
            bound = "<" + self._lower_bound + ", " + self._upper_bound + "> "

        return bound

    def _gen_def_code(self) -> TListTExpression:

        bound = self._gen_bound_str()

        return [self._var_type + bound + self.name]


class ForwardArrayDef(VariableDef):
    """Define an array of variables"""

    def __init__(self, name: str, var_type: str, array_dim: TListTExpression) -> None:

        self._var_type = var_type
        self._array_dim = array_dim
        VariableDef.__init__(self, name)

        for expr in self._array_dim:
            if isinstance(expr, Expression):
                expr.add_output(self)

    def _gen_def_code(self) -> TListTExpression:
        # return [self._var_type + " " + self.name] + self._array_dim
        return ["array"] + self._array_dim + [" " + self._var_type + " " + self.name]


class ForwardVectorDef(VariableDef):
    """Define a vector of variables"""

    def __init__(self, name: str, array_dim: TListTExpression) -> None:

        self._array_dim = array_dim
        VariableDef.__init__(self, name)

        for expr in self._array_dim:
            if isinstance(expr, Expression):
                expr.add_output(self)

    def _gen_def_code(self) -> TListTExpression:
        # return [self._var_type + " " + self.name] + self._array_dim
        return ["vector["] + self._array_dim + ["] " + self.name]


class ParameterVectorDef(ParameterDef):
    """Define a vector of parameters"""

    def __init__(
        self,
        name: str,
        var_type: str,
        array_dim: TListTExpression,
        lower_bound: float,
        upper_bound: float,
    ) -> None:

        self._array_dim = array_dim
        ParameterDef.__init__(self, name, var_type, lower_bound, upper_bound)

    def _gen_def_code(self) -> TListTExpression:

        bound = self._gen_bound_str()

        return [self._var_type + bound] + self._array_dim + [" " + self.name]


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
        array_data: Iterable,
        array_shape: Union[Iterable, None] = None,
    ):

        self._array_data = np.asarray(array_data)
        self._type = type_name
        self._array_shape = array_shape
        VariableDef.__init__(self, name)

    def _gen_def_code(self) -> TListTExpression:
        """
        See parent class
        """

        # Variable Definition
        stan_code = self._type
        if self._array_shape is None:
            shape_str = "[" + ",".join(
                [str(shape_d) for shape_d in self._array_data.shape]
            )
        else:
            shape_str = "[" + ",".join([str(shape_d) for shape_d in self._array_shape])
        shape_str += "]"
        if self._type in ["vector", "row_vector"]:
            stan_code += shape_str + " " + self.name

        else:
            stan_code = "array" + shape_str + " " + self._type + " " + self.name

        # Fill array
        arraystr = np.array2string(self._array_data, threshold=np.inf, separator=",")
        if "vector" not in self._type:
            arraystr = arraystr[1:-1]
            arraystr = arraystr.replace("[", "{")
            arraystr = arraystr.replace("]", "}")
            arraystr = "{" + arraystr + "}"
        stan_code += " = " + arraystr
        if self._type in ["vector"]:
            stan_code += "'"  # FU Stan, I feel your pain
        stan_code += ""

        return [stan_code]


class StanVector(VariableDef):
    """
    Stan vector definition

    Parameters:
        name: Variable name to use in stan code
    """

    def __init__(self, name: str, array_data: Iterable):

        self._array_data = np.asarray(array_data)
        self._type = "vector"
        VariableDef.__init__(self, name)

    def _gen_def_code(self) -> TListTExpression:
        """
        See parent class
        """

        # Variable Definition
        stan_code = self._type

        shape_str = "[" + str(self._array_data.size) + "]"

        stan_code += shape_str + " " + self.name

        # Fill array
        arraystr = np.array2string(self._array_data, threshold=np.inf, separator=",")
        stan_code += " = " + arraystr
        stan_code += "'"  # FU Stan
        stan_code += ""

        return [stan_code]


if __name__ == "__main__":
    from .stan_generator import StanGenerator, GeneratedQuantitiesContext

    logging.basicConfig(level=logging.DEBUG)
    with StanGenerator() as cg:
        with GeneratedQuantitiesContext() as gq:
            val = ForwardVariableDef("a", "real")
            val << "b"

        print(cg.generate())
