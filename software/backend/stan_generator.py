"""Module for autogenerating Stan code"""
from typing import (List, TYPE_CHECKING)
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

    def to_stan(self) -> str:
        """Convert added StanCodeBits to Stan code"""

        functions: List[str] = []

        def loop_code_bits(code_bits: List["StanCodeBit"], iteration=0):
            """
            Recursively loop through StanCodeBits
            """
            main_code = ""
            functions = ""
            definitions = ""
            for code_bit in code_bits:
                sfunc = code_bit.functions
                sdefs = code_bit.def_codes

                # Check if we have to add any stan variable definitions
                if sdefs:
                    # This bit contains variable defs, add them
                    for sdef in sdefs:
                        fc, defs, mc += loop_code_bits([sdef.def_code], iteration+1)
                        if fc or defs:
                            logger.warn("Variable definition contains definition or function code")
                        # TODO: what about duplicates?

                # Check if we have to add any stan functions
                if sfunc:
                    # This bit contains functions, add them
                    for func in sfunc:
                        # Only add if not already added
                        if func.name not in functions:
                            main_code += loop_code_bits([func.func_code])
                            functions.append(func.name)

                            """
                            if fcode_bit.functions:
                                # This function defines functions, not supported
                                raise RuntimeError("Defining functions within functions currently not supported")  # noqa: E501
                            fsdefs = fcode_bit.def_codes
                            functions[func.name] = fsdefs + "\n" + fcode_bit.code + "\n"
                            """
                        else:
                            logger.warn("Function %s already added", func.name)

                main_code += "functions {\n"
                main_code += code_bit.code + "\n"
                main_code += "} \n"
            return main_code

        # Loop through all collected code bits
        stan_code = loop_code_bits(self._code_bits)

        return stan_code
