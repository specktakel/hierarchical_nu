"""Module for autogenerating Stan code"""
from typing import List, Union, Dict
from expression import TExpression, Expression

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


def stanify(var: TExpression) -> StanCodeBit:
    """Call to_stan function if possible"""
    if isinstance(var, Expression):
        return var.to_stan()

    # Not an Expression, so cast to string
    code_bit = StanCodeBit()
    code_bit.add_code([str(var)])
    return code_bit
