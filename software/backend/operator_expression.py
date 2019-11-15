from typing import Callable, Tuple, List
import operator

from .expression import Expression, TExpression, TListTExpression
from .pymc_generator import pymcify


class _OperatorExpression(Expression):
    """
    Implements basic arithmetic operations for Expressions

    Parameters:
        inputs: List[TExpression]
            Exactly two inputs for an operator (LHS, RHS)
        op_code: Tuple[str, str]
            Tuple of op code (e.g. `+`) and python operator name
            (e.g. `__add__`)


    """
    def __init__(self, inputs: List[TExpression],
                 op_code: Tuple[str, str]) -> None:
        Expression.__init__(self, inputs)
        self._op_code = op_code

    @property
    def stan_code(self) -> TListTExpression:
        """See base class"""
        in0_stan = self._inputs[0]
        in1_stan = self._inputs[1]

        stan_code:  TListTExpression = ["(", in0_stan, self._op_code[0],
                                        in1_stan, ")"]

        return stan_code

    def to_pymc(self):
        """
        See parent class
        """
        in0_pymc = pymcify(self._inputs[0])
        in1_pymc = pymcify(self._inputs[1])
        return getattr(operator, self._op_code[1])(in0_pymc, in1_pymc)
        pass

def make_op_func(op_code: Tuple[str, str], invert: bool = False) -> Callable:
    """
    Return a factory function that creates a operator expression

    Args:
        op_code: Tuple[str, str]
            Tuple of op code (e.g. `+`) and python operator
            name (e.g. `__add__`)
        invert: bool
            Set to true for right-hand operators
    """
    def op_func(self: TExpression, other: TExpression) -> _OperatorExpression:
        if invert:
            inputs = [other, self]
        else:
            inputs = [self, other]
        return _OperatorExpression(inputs, op_code)
    return op_func


def register_operators(cls, ops):
    """Register an operator overload for a class"""
    for op_type, (op_code, invert) in ops.items():
        func = make_op_func((op_code, op_type), invert)
        setattr(cls, op_type, func)

"""
Register standard arithmetic operators for both the Expression
baseclass and the _OperatorExpression class. This allows operator
manipulations of Expressions (and subclasses), as well as operator
manipulations of _OperatorExpressions
"""

ops = {
    "__add__": ("+", False),
    "__radd__": ("+", True),
    "__mul__": ("*", False),
    "__rmul__": ("*", True),
    "__div__": ("/", False),
    "__rdiv__": ("/", True),
    "__sub__": ("-", False),
    "__rsub__": ("-", True),
    "__pow__": ("**", False),
    "__rpow__": ("**", True),
    }

register_operators(_OperatorExpression, ops)
register_operators(Expression, ops)