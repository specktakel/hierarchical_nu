from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, List
import logging
from .typedefs import TStanHookDict
from .stan_code import StanCodeBit, TListStrStanCodeBit

logger = logging.getLogger(__name__)

__all__ = ["Expression", "TExpression", "StanFunction",
           ]


class StanFunction:
    """
    Class representing a Stan function

    Parameters:
        name: str
        code: TListStrStanCodeBit
    """

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._func_code: TListStrStanCodeBit = []

    def add_func_code(self, code: TListStrStanCodeBit):
        self._func_code += code

    @property
    def name(self) -> str:
        return self._name

    @property
    def func_code(self) -> StanCodeBit:
        new_code_bit = StanCodeBit()
        new_code_bit.add_code(self._func_code)
        return new_code_bit


class StanDefCode:
    def __init__(self, name: str) -> None:
        self._name = name
        self._def_code: TListStrStanCodeBit = []

    def add_def_code(self, code: TListStrStanCodeBit):
        self._def_code += code

    @property
    def name(self) -> str:
        return self._name

    @property
    def def_code(self) -> StanCodeBit:
        new_code_bit = StanCodeBit()
        new_code_bit.add_code(self._def_code)
        return new_code_bit


class Expression(metaclass=ABCMeta):
    """
    Generic expression

    The expression can depend on inputs, such that it's possible to
    chain Expressions in a graph like manner.
    Comes with converters to PyMC3 and Stan Code.
    """

    def __init__(self, inputs: Sequence["TExpression"]):
        self._inputs: List["TExpression"] = list(inputs)
        self._stan_hooks: TStanHookDict = {}

    @property
    @abstractmethod
    def stan_code(self) -> TListStrStanCodeBit:
        """Main code of the expression"""
        pass

    @property
    def stan_hooks(self) -> TStanHookDict:
        return self._stan_hooks

    def add_stan_hook(self, name: str, hook_type: str, code: TListStrStanCodeBit):
        """
        Add a stan hook

        These can be used to tell the generator to add code e.g. for variable
        definitions or functions.

        Parameters:
            name: str
            type: str
            code: str
        """
        if name in self._stan_hooks:
            logger.warning("Hook with name %s already exists. Skipping..", name)  # noqa: E501
        else:
            self._stan_hooks[name] = (hook_type, code)

    def to_stan(self) -> "StanCodeBit":
        """
        Converts the expression into a StanCodeBit
        """

        code_bit = StanCodeBit()
        code_bit.add_code(self.stan_code)
        for hook_name, (hook_type, hook_code) in self._stan_hooks.items():
            if hook_type == "function":
                func = StanFunction(hook_name)
                func.add_func_code(hook_code)
                code_bit.add_function(func)
            elif hook_type == "var_def":
                def_code = StanDefCode(hook_name)
                def_code.add_def_code(hook_code)
                code_bit.add_definition(def_code)
        return code_bit

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
