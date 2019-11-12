from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, List
import logging
from .stan_code import StanCodeBit, TListStrStanCodeBit
from .code_generator import Contextable

logger = logging.getLogger(__name__)

__all__ = ["Expression", "StanDefCode", "ReturnStatement",
           "NamedExpression", "TExpression"]


class Expression(Contextable, metaclass=ABCMeta):
    """
    Generic expression

    The expression can depend on inputs, such that it's possible to
    chain Expressions in a graph like manner.
    Comes with converters to PyMC3 and Stan Code.
    """

    def __init__(self, inputs: Sequence["TExpression"]):
        Contextable.__init__(self)
        self._inputs: List["TExpression"] = list(inputs)
        for input in self._inputs:
            if isinstance(input, Expression):
                input.add_output(self)
            else:
                logger.debug("Found non Expression of type: {} in input.".format(input))
        self._output: List["TExpression"] = []

    @property
    def output(self) -> List["TExpression"]:
        return self._output

    def add_output(self, output: "TExpression"):
        self._output.append(output)

    @property
    @abstractmethod
    def stan_code(self) -> "TListTExpression":
        """Main code of the expression"""
        pass

    def to_stan(self) -> StanCodeBit:
        """
        Converts the expression into a StanCodeBit
        """

        code_bit = StanCodeBit()

        converted_code: TListStrStanCodeBit = []

        for code in self.stan_code:
            if isinstance(code, Expression):
                converted_code.append(code.to_stan())
            else:
                if isinstance(code, (float, int)):
                    code = str(code)
                if not isinstance(code, (StanCodeBit, str)):
                    raise RuntimeError("Code has incompatible type: {}".format(type(code)))  # noqa: E501
                converted_code.append(code)

        code_bit.add_code(converted_code)
        return code_bit

    def to_pymc(self):
        pass


# Define type union for stanable types
TExpression = Union[Expression, str, float, int]
TListTExpression = List[TExpression]


class _GetItemExpression(Expression):

    @property
    def stan_code(self) -> TListTExpression:
        """See base class"""
        base_expression = self._inputs[0]
        key_expression = self._inputs[1]

        return [base_expression, "[", key_expression, "]"]


def getitem_func(self: Expression, key: TExpression):
    return _GetItemExpression([self, key])


setattr(_GetItemExpression, "__getitem__", getitem_func)
setattr(Expression, "__getitem__", getitem_func)


class StringExpression(Expression):
    @property
    def stan_code(self) -> TListTExpression:
        stan_code: TListTExpression = list(self._inputs)
        return stan_code


class NamedExpression(Expression):
    def __init__(self, inputs: Sequence[TExpression], name: str):
        Expression.__init__(self, inputs)
        self._name = name

    @property
    def name(self):
        return self._name


TNamedExpression = Union[NamedExpression, str, float, int]


class ReturnStatement(Expression):
    def __init__(self, inputs: Sequence[TExpression]):
        Expression.__init__(self, inputs)

    @property
    def stan_code(self) -> TListTExpression:
        stan_code: TListTExpression = ["return "]
        stan_code += self._inputs
        return stan_code


class StanDefCode(Expression):
    """
    Class representing a variable definition
    """

    def __init__(self) -> None:
        Expression.__init__(self, [])
        self._def_code: TListTExpression = []

    def add_def_code(self, code: TListTExpression):
        self._def_code += code

    @property
    def stan_code(self) -> TListTExpression:
        return self._def_code

    def to_pymc(self):
        pass
