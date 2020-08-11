from typing import Union, List

# Declare a type for a List containing either str or StanCodeBits
TStrStanCodeBit = Union[str, "StanCodeBit"]
TListStrStanCodeBit = List[TStrStanCodeBit]


class StanCodeBit:
    """
    Class representing one or multiple Stan statements
    """

    def __init__(self, end_delim=";\n"):
        self._code: List[str] = []
        self._end_delim = end_delim

    @property
    def code(self) -> str:
        return "".join(self._code)
    
    @property
    def end_delim(self) -> str:
        return self._end_delim

    def add_code(self, code: TListStrStanCodeBit) -> None:
        for code_bit in code:
            if isinstance(code_bit, StanCodeBit):
                self._code.append(code_bit.code)
            else:
                self._code.append(code_bit)

    def __repr__(self) -> str:
        raise NotImplementedError
