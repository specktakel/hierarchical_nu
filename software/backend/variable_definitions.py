from abc import abstractmethod
from .expression import Expression
from .stan_generator import StanCodeBit
import numpy as np  # type:ignore

__all__ = ["VariableDef", "StanArray"]


class VariableDef(Expression):
    """
    Stan variable definition

    """

    def __init__(
            self,
            name: str):
        Expression.__init__(self, [])
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    @abstractmethod
    def def_code(self) -> str:
        pass

    def to_stan(self) -> StanCodeBit:
        code_bit = StanCodeBit()
        code_bit.add_code([self._name])
        code_bit.add_def_code(self.def_code)
        return code_bit

    def to_pymc(self):
        pass


class StanArray(VariableDef):
    """
    Stan real array definition
    """

    def __init__(
            self,
            name: str,
            type_name: str,
            array_data: np.ndarray):
        VariableDef.__init__(self, name)
        self._array_data = array_data
        self._type = type_name

    @property
    def def_code(self) -> str:
        """
        See parent class
        """

        # Variable Definition
        stan_code = self._type + " " + self._name
        for shape_d in self._array_data.shape:
            stan_code += "[" + str(shape_d) + "]"
        #stan_code += "; \n"

        # Fill array
        arraystr = np.array2string(
            self._array_data,
            threshold=np.inf,
            separator=",")
        arraystr = arraystr.replace("[", "{")
        arraystr = arraystr.replace("]", "}")
        stan_code += " = " + arraystr + "; \n"
        return stan_code
