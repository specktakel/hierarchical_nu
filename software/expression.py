from abc import ABCMeta, abstractmethod
from typing import List, Union

"""
try:
    from typing import ForwardRef  # type: ignore
except ImportError:
    # python 3.6
    from typing import _ForwardRef as ForwardRef # type: ignore
"""

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
    def to_stan(self) -> "StanCodeBit":
        """
        Converts the expression into a StanCodeBit
        """
        pass

    @abstractmethod
    def to_pymc(self):
        pass


# Define type union for stanable types
TExpression = Union[Expression, str, float]
