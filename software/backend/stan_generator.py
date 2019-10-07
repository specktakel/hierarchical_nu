"""Module for autogenerating Stan code"""
from typing import (List, Dict, TYPE_CHECKING, Tuple)
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .stan_code import StanCodeBit
__all__ = ["StanGenerator"]


class StanGenerator:
    """
    Class for autogenerating Stan code
    """

    def __init__(self):
        self._code_bits: List["StanCodeBit"] = []

    def add_code_bit(self, code_bit: "StanCodeBit") -> None:
        """Add a StanCodeBit"""
        self._code_bits.append(code_bit)

    def to_stan(self) -> Tuple[str, str]:
        """Convert added StanCodeBits to Stan code"""

        def loop_code_bits(
                code_bits: List["StanCodeBit"],
                functions: Dict[str, str],
                iteration=0):
            """
            Recursively loop through StanCodeBits
            """
            main_code = ""
            for code_bit in code_bits:
                definitions: Dict[str, List["StanCodeBit"]] = {}
                sfunc = code_bit.functions
                sdefs = code_bit.def_codes

                # Check if we have to add any stan variable definitions
                if sdefs:
                    # This bit contains variable defs, add them
                    for sdef in sdefs:
                        if sdef.name not in definitions:
                            # Only add if not already added
                            mc = loop_code_bits(
                                [sdef.def_code],
                                functions,
                                iteration+1)
                            definitions[sdef.name] = mc
                            main_code += mc
                            # if fc or defs:
                            #     logger.warn("Variable definition contains definition or function code")  # noqa: E501
                            # functions += fc
                            # definitions += defs
                            # main_code += mc

                # Check if we have to add any stan functions
                if sfunc:
                    # This bit contains functions, add them
                    for func in sfunc:
                        # Only add if not already added
                        if func.name not in functions:
                            function_code = func.func_header + "\n"
                            function_code += "{\n"
                            for func_code in func.func_code:
                                mc = loop_code_bits(
                                    [func_code],
                                    functions,
                                    iteration+1)
                                function_code += mc + "\n"
                            function_code += "}\n"

                            functions[func.name] = function_code
                        else:
                            logger.warn("Function %s already added", func.name)

                main_code += code_bit.code + "\n"
            return main_code

        # Loop through all collected code bits
        functions: Dict[str, str] = {}
        stan_code = loop_code_bits(self._code_bits, functions)

        # Add function codes
        func_code = ""
        for fname, fcode in functions.items():
            func_code += fcode + "\n"
        return func_code, stan_code
