from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .expression import StanFunction, StanDefCode
# Declare a type for a List containing either str or StanCodeBits
TStrStanCodeBit = Union[str, "StanCodeBit"]
TListStrStanCodeBit = List[TStrStanCodeBit]


class StanCodeBit:
    """
    Class representing one or multiple Stan statements
    """

    def __init__(self):
        self._functions: List["StanFunction"] = []
        self._def_codes: List["StanDefCode"] = []
        self._code: List[str] = []

    @property
    def code(self) -> str:
        return "".join(self._code)

    def add_function(self, function: "StanFunction") -> None:
        """Add a StanFunction"""
        self._functions.append(function)

    def add_definition(self, def_code: "StanDefCode") -> None:
        """Add a variable definition"""
        self._def_codes.append(def_code)

    def add_code(self, code: TListStrStanCodeBit) -> None:
        for code_bit in code:
            if isinstance(code_bit, StanCodeBit):
                self._code.append(code_bit.code)
            else:
                self._code.append(code_bit)

    def __repr__(self) -> str:
        raise NotImplementedError

