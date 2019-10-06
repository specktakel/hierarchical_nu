"""Module for autogenerating Stan code"""
from typing import List, Union, Dict, TYPE_CHECKING
from .expression import TExpression, Expression

if TYPE_CHECKING:
    from .variable_definitions import VariableDef

__all__ = ["StanFunction", "StanGenerator", "stanify",
           "StanCodeBit", "TStrStanCodeBit", "TListStrStanCodeBit"]


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
        # def_codes: Dict[str, "VariableDef"] = {}
        def_codes: List[str] = []

        main_code = ""
        # Loop through all collected code bits
        for code_bit in self._code_bits:
            sfunc = code_bit.functions
            sdefs = code_bit.def_codes
            # Check if we have to add any stan functions
            if sfunc:
                # This bit contains functions, add them
                for func in sfunc:
                    # Only add if not already added
                    if func.name not in functions:
                        functions[func.name] = func.code

            # Check if we have to add any stan variable definitions
            if sdefs:
                # This bit contains variable defs, add them
                for sdef in sdefs:
                    """
                    # Only add if not already added
                    if sdef.name not in def_codes:
                        def_codes[sdef.name] = sdef.code
                    else:
                        raise RuntimeError("Variable with name %s already" /
                                           " defined")
                    """
                    def_codes.append(sdef)

            main_code += code_bit.code + "\n"

        # Generate function code
        func_code = "functions { \n"
        for fname, fcode in functions:
            func_code += fcode + "\n"
        func_code += "}\n"

        # Generate variable def code
        def_code = "\n".join(def_codes)

        return func_code + def_code + main_code


# Declare a type for a List containing either str or StanCodeBits
TStrStanCodeBit = Union[str, "StanCodeBit"]
TListStrStanCodeBit = List[TStrStanCodeBit]


class StanCodeBit:
    """
    Class representing one or multiple Stan statements
    """

    def __init__(self):
        self._functions = []
        self._def_codes = []
        self._code: List[str] = []

    @property
    def functions(self) -> List[StanFunction]:
        return self._functions

    @property
    def def_codes(self) -> List["VariableDef"]:
        return self._def_codes

    """
    def unravel(self) -> None:
        # Resolve recursions of CodeBits
        new_code: TListStrStanCodeBit = []
        for code_bit in self._code:
            if isinstance(code_bit, StanCodeBit):
                # If a part of the code is still a CodeBit,
                # call its code method
                new_code.append(code_bit.code)
                if code_bit.functions:
                    self._functions += code_bit.functions
                if code_bit.def_codes:
                    self._def_codes += code_bit.def_codes
            else:
                new_code.append(code_bit)
        self._code = new_code
    """

    @property
    def code(self) -> str:
        return "".join(self._code)

    def add_function(self, function: StanFunction) -> None:
        """Add a StanFunction"""
        self._functions.append(function)

    def add_def_code(self, def_code: "VariableDef") -> None:
        """Add a variable definition"""
        self._def_codes.append(def_code)

    def add_code(self, code: TListStrStanCodeBit) -> None:

        for code_bit in code:
            if isinstance(code_bit, StanCodeBit):
                # If a part of the code is still a CodeBit,
                # call its code method
                self._code.append(code_bit.code)
                if code_bit.functions:
                    self._functions += code_bit.functions
                if code_bit.def_codes:
                    self._def_codes += code_bit.def_codes
            else:
                self._code.append(code_bit)

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
