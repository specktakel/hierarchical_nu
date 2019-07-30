"""Module for autogenerating Stan code"""
from typing import List, Union, Dict, Callable
from abc import ABCMeta, abstractmethod


class StanFunction:
    """
    Class representing a Stan function
    """

    def __init__(self, name: str, code: str) -> None:
        self._name = name
        self._code = code

    @property
    def name(self):
        return self._name

    @property
    def code(self):
        return self._code


# TODO: Add specific classes for other Stan blocks?!


class StanGenerator:
    """
    Class for autogenerating Stan code
    """

    def __init__(self):
        self._code_bits = []

    def add_code_bit(self, code_bit):
        self._code_bits.append(code_bit)

    def to_stan(self) -> str:
        functions: Dict[str, StanFunction] = {}

        main_code = ""
        # Loop through all collected code bits
        for code_bit in self._code_bits:
            sfunc = code_bit.functions

            # Check if we have to add any stan functions
            if sfunc:
                # This bit contains functions, add them
                for func in sfunc:
                    # Only add if not already added
                    if func.name not in functions:
                        functions[func.name] = func.code

            main_code += code_bit.code + "\n"

        # Generate function code
        func_code = "functions { \n"
        for fname, fcode in functions:
            func_code += fcode + "\n"
        func_code += "}\n"

        return func_code + main_code


# Declare a type for a List containing either str or StanCodeBits
TStrStanCodeBit = Union[str, "StanCodeBit"]
TListStrStanCodeBit = List[TStrStanCodeBit]


class StanCodeBit:
    """
    Class representing one or multiple Stan statements
    """

    def __init__(self):
        self._functions = []
        self._code: TListStrStanCodeBit = []

    @property
    def functions(self) -> List[StanFunction]:
        return self._functions

    @property
    def code(self) -> str:
        # We have to resolve recursions of CodeBits
        code_str = ""
        for code_bit in self._code:
            if isinstance(code_bit, StanCodeBit):
                # If a part of the code is still a CodeBit,
                # call its code method
                code_str += code_bit.code
            else:
                code_str += code_bit

        return code_str

    def add_function(self, function: StanFunction) -> None:
        """Add a StanFunction"""
        self._functions.append(function)

    def add_code(self, code: TListStrStanCodeBit) -> None:
        self._code += code

    def __repr__(self) -> str:

        code_gen = StanGenerator()
        code_gen.add_code_bit(self)

        return code_gen.to_stan()


class Expression(metaclass=ABCMeta):
    """
    Generic expression

    The expression can depend on inputs, such that it's possible to
    chain Expressions in a graph like manner.
    Comes with converters to PyMC3 and Stan Code.
    """

    def __init__(self, inputs: List["TExpression"]):
        self._inputs = inputs

    @abstractmethod
    def to_stan(self) -> StanCodeBit:
        """
        Converts the expression into a StanCodeBit
        """
        pass

    @abstractmethod
    def to_pymc(self):
        pass


# Define type union for stanable types
TExpression = Union[Expression, str, float]


def stanify(var: TExpression) -> StanCodeBit:
    """Call to_stan function if possible"""
    if isinstance(var, Expression):
        return var.to_stan()

    # Not an Expression, so cast to string
    code_bit = StanCodeBit()
    code_bit.add_code([str(var)])
    return code_bit


class _OperatorExpression(Expression):
    def __init__(self, inputs: List[TExpression], op_code: str) -> None:
        Expression.__init__(self, inputs)
        self._op_code = op_code

    def to_stan(self) -> StanCodeBit:
        in0_stan = stanify(self._inputs[0])
        in1_stan = stanify(self._inputs[1])

        stan_code:  TListStrStanCodeBit = ["(", in0_stan, self._op_code, in1_stan, ")"]

        code_bit = StanCodeBit()
        code_bit.add_code(stan_code)

        return code_bit

    def to_pymc(self):
        pass


def make_op_func(op_code: str, invert: bool = False) -> Callable:
    """
    Return a factory function that creates a operator expression

    Args:
        op_code: str
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
    """Register a operator overload for a class"""
    for op_type, (op_code, invert) in ops.items():
        func = make_op_func(op_code, invert)

        setattr(cls, op_type, func)


"""
Register standard arithmetic operators for both the Expression
baseclass and the _OperatorExpression class.This allows operator
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
