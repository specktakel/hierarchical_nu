from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, List, Tuple
import logging
from .stan_code import StanCodeBit, TListStrStanCodeBit
from .code_generator import Contextable, ContextStack
from .baseclasses import NamedObject

logger = logging.getLogger(__name__)

__all__ = [
    "Expression",
    "ReturnStatement",
    "StringExpression",
    "NamedExpression",
    "TExpression",
    "TListTExpression",
]


class _BaseExpression(Contextable, metaclass=ABCMeta):
    """
    Generic expression

    The expression can depend on inputs, such that it's possible to
    chain Expressions in a graph like manner.
    Comes with converters to PyMC3 and Stan Code.
    """

    def __init__(
        self, inputs: Sequence["TExpression"], block_output: bool, end_delim=";\n"
    ):
        Contextable.__init__(self)
        self._end_delim = end_delim
        logger.debug("Input is of type: {}".format(type(inputs)))

        # If inputs is an expression, rather than a list of expression,
        # converting to a list will cause a loop of death. This is due
        # to the [] overload of Expression
        assert not isinstance(inputs, Expression)

        self._inputs: List["TExpression"] = list(inputs)
        for input in self._inputs:
            if isinstance(input, Expression):
                input.add_output(self)
            else:
                logger.debug(
                    "Found non Expression of type: {} in input.".format(input)
                )  # noqa: E501
        self._output: List["TExpression"] = []
        self._block_output = block_output

    @property
    def output(self) -> List["TExpression"]:
        return self._output

    def add_output(self, output: "TExpression"):
        if not self._block_output:
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

        code_bit = StanCodeBit(end_delim=self._end_delim)

        converted_code: TListStrStanCodeBit = []

        for code in self.stan_code:
            if isinstance(code, _BaseExpression):
                converted_code.append(code.to_stan())
            else:
                if isinstance(code, (float, int)):
                    code = str(code)
                if not isinstance(code, (StanCodeBit, str)):
                    msg = "Code has incompatible type: {}".format(type(code))
                    msg += "\n My code was: {}".format(self.stan_code)
                    raise RuntimeError(msg)  # noqa: E501
                converted_code.append(code)

        code_bit.add_code(converted_code)
        return code_bit

    def to_pymc(self):
        pass


# Define type union for stanable types
TExpression = Union[_BaseExpression, str, float, int, slice, Tuple]
TListTExpression = List[TExpression]


class Expression(_BaseExpression):
    def __init__(
        self,
        inputs: Sequence["TExpression"],
        stan_code: TListTExpression,
        block_output=False,
        end_delim=";\n",
    ):
        _BaseExpression.__init__(self, inputs, block_output, end_delim=end_delim)
        self._stan_code = stan_code

    @property
    def stan_code(self) -> TListTExpression:
        return self._stan_code

    def __getitem__(self, key):
        from .stan_generator import IndexingContext

        with IndexingContext(key) as idx:
            output: TListTExpression = [self, *idx]
        return StringExpression(output)

    """
    def __getitem__(self: _BaseExpression, key: TExpression):

        stan_code: TListTExpression = [self, "[", key, "]"]

        return Expression([self, key], stan_code)
    """

    def __lshift__(self: _BaseExpression, other: Union[TExpression, TListTExpression]):
        logger.debug("Assigning {} to {}".format(other, self))  # noqa: E501
        logger.debug("My code: {}".format(self.stan_code))  # noqa: E501
        if not isinstance(other, list):
            other = [other]
        stan_code: TListTExpression = [self, " = "]
        stan_code += other

        if self.output:
            # Output node is already connected to something
            logger.debug("Output is connected to: {}".format(self.output))  # noqa: E501

        inputs: TListTExpression = [self]
        inputs += other
        return Expression(inputs, stan_code)

    """
    def __iadd__(self: _BaseExpression, other: Union[TExpression, TListTExpression]):
        # Why is this not working? It does not print a line as __lshift__ with `+=` instead of `=`
        # Instead at next call of variable the expression is printed
        logger.debug("Assigning {} to {}".format(other, self))  # noqa: E501
        logger.debug("My code: {}".format(self.stan_code))  # noqa: E501
        if not isinstance(other, list):
            other = [other]
        stan_code: TListTExpression = [self, " += "]
        stan_code += other

        if self.output:
            # Output node is already connected to something
            logger.debug("Output is connected to: {}".format(self.output))  # noqa: E501

        inputs: TListTExpression = [self]
        inputs += other
        return Expression(inputs, stan_code)
    """

    def _make_operator_expression(self, other: TExpression, op_code, invert=False):
        stan_code: TListTExpression = []
        if invert:
            stan_code += ["(", other, op_code, self, ")"]
        else:
            stan_code += ["(", self, op_code, other, ")"]

        return Expression([self, other], stan_code)

    def __add__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "+")

    def __radd__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "+", True)

    def __mul__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "*")

    def __rmul__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "*", True)

    def __truediv__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "/")

    def __rtruediv__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "/", True)

    def __sub__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "-")

    def __rsub__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "-", True)

    def __pow__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "^")

    def __rpow__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "^", True)

    def __ne__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "!=")

    def __rne__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "!=", True)

    def __eq__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "==")

    def __req__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "==", True)

    def __mod__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "%")

    def __rmod__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "%", True)

    def __neg__(self: "Expression") -> "Expression":
        return self._make_operator_expression("", "-", True)

    """
    Comparisons are used internally to sort contexts, FIX
    def __lt__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "<")

    def __rlt__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "<", True)

    def __le__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "<=")

    def __rle__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, "<=", True)

    def __gt__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, ">")

    def __rgt__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, ">", True)

    def __ge__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, ">=")

    def __rge__(self: "Expression", other: TExpression) -> "Expression":
        return self._make_operator_expression(other, ">=", True)
    """


class StringExpression(Expression):
    def __init__(self, inputs: Sequence["TExpression"]):
        stan_code = list(inputs)
        Expression.__init__(self, inputs, stan_code)


class NamedExpression(Expression, NamedObject):
    def __init__(
        self, inputs: Sequence[TExpression], stan_code: TListTExpression, name: str
    ):
        Expression.__init__(self, inputs, stan_code)
        self._name = name


TNamedExpression = Union[NamedExpression, str, float, int]


class ReturnStatement(Expression):
    def __init__(self, inputs: Sequence[TExpression]):
        stan_code: TListTExpression = ["return "]
        stan_code += inputs
        Expression.__init__(self, inputs, stan_code)


class PlainStatement(Expression):
    def __init__(self, inputs: Sequence[TExpression]):
        stan_code = list(inputs)
        super().__init__(inputs, stan_code)
        self._end_delim = ""
