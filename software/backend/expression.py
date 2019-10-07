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

    def __init__(self, name: str, func_header: str) -> None:
        self._name: str = name
        self._func_code: List[TListStrStanCodeBit] = []
        self._func_header: str = func_header

    def add_func_code(self, code: TListStrStanCodeBit):
        self._func_code.append(code)

    @property
    def name(self) -> str:
        return self._name

    @property
    def func_header(self):
        return self._func_header

    @property
    def func_code(self) -> List[StanCodeBit]:
        code_bits: List[StanCodeBit] = []
        for fc in self._func_code:
            new_code_bit = StanCodeBit()
            new_code_bit.add_code(fc)
            code_bits.append(new_code_bit)
        return code_bits


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

    def add_stan_hook(
            self,
            name: str,
            hook_type: str,
            hook_obj: Union[StanFunction, StanDefCode]):
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
            self._stan_hooks[name] = (hook_type, hook_obj)

    def to_stan(self) -> "StanCodeBit":
        """
        Converts the expression into a StanCodeBit
        """

        code_bit = StanCodeBit()
        code_bit.add_code(self.stan_code)
        for hook_name, (hook_type, hook_code) in self._stan_hooks.items():
            if hook_type == "function":
                assert isinstance(hook_code, StanFunction)
                code_bit.add_function(hook_code)
            elif hook_type == "var_def":
                assert isinstance(hook_code, StanDefCode)
                code_bit.add_definition(hook_code)
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


class AssignValue(Expression):

    def __init__(self, inputs: Sequence[TExpression], output: TExpression):
        Expression.__init__(self, inputs)
        self._output = output

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        stan_code: TListStrStanCodeBit = [
            stanify(self._output), " = ", stanify(self._inputs[0])]
        return stan_code

    def to_pymc(self):
        pass
    
