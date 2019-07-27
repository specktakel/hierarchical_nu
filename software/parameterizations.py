from abc import ABCMeta, abstractmethod
from typing import Union, List, Iterable
import numpy as np
from stan_generator import StanCodeBit, TListStrStanCodeBit


class StanExpression(metaclass=ABCMeta):
    """
    Generic Stan expression

    The expression can depend on inputs, such that it's possible to
    chain StanExpressions in a graph like manner
    """

    def __init__(self, inputs: List["TStanable"]):
        self.inputs = inputs

    @abstractmethod
    def to_stan(self) -> StanCodeBit:
        """
        Converts the expression into a StanCodeBit
        """
        pass


class PyMCExpression(metaclass=ABCMeta):
    def __init__(self, inputs):
        self.inputs = inputs

    @abstractmethod
    def to_pymc(self):
        pass


# Define type union for stanable types
TStanable = Union[StanExpression, str, float]


class Parameterization(StanExpression,
                       PyMCExpression,
                       metaclass=ABCMeta):
    """
    Base class for parameterizations.

    Parameterizations are functions of a given input variable.
    These can be splines, distributions, ...
    Comes with a converter to stan code / pymc3 variables.
    """

    def __init__(self, inputs: List[TStanable]):
        self._inputs = inputs

    @abstractmethod
    def to_stan(self) -> StanCodeBit:
        """
        Convert the parameterization to Stan
        """
        pass

    @abstractmethod
    def to_pymc(self):
        """
        Convert the parametrizaton to PyMC3
        """
        pass


def stanify(var: TStanable) -> StanCodeBit:
    """Return call to to_stan function if possible"""
    if isinstance(var, StanExpression):
        return var.to_stan()
    code_bit = StanCodeBit()
    code_bit.add_code([str(var)])
    return code_bit


class LogParameterization(Parameterization):
    """log with customizable base"""
    def __init__(self, inputs: TStanable, base: float = 10):
        Parameterization.__init__(self, [inputs])
        self._base = base

    def to_stan(self) -> StanCodeBit:
        """See base class"""

        x_eval_stan = stanify(self._inputs[0])
        stan_code: TListStrStanCodeBit = []
        if self._base != 10:
            base = str(self._base)
            stan_code += ["log10(", x_eval_stan, "}) / log10(", base, ")"]
        else:
            stan_code += ["log10(", x_eval_stan, "})"]

        stan_code_bit = StanCodeBit()
        stan_code_bit.add_code(stan_code)

        return stan_code_bit

    def to_pymc(self):
        pass


TArrayOrNumericIterable = Union[np.ndarray, Iterable[float]]


class PolynomialParameterization(Parameterization):
    """Polynomial parametrization"""

    def __init__(
            self,
            inputs: TStanable,
            coefficients: TArrayOrNumericIterable) -> None:

        Parameterization.__init__(self, [inputs])
        self._coeffs = coefficients

    def to_stan(self) -> StanCodeBit:
        """See base class"""

        x_eval_stan = stanify(self._inputs[0])

        # TODO: Make sure that eval_poly1d is part of some util lib.
        # Or maybe add a hook for loading ?

        coeffs_stan = "[" + ",".join([str(coeff)
                                      for coeff in self._coeffs]) + "]"

        stan_code: TListStrStanCodeBit = [
            "eval_poly1d(",
            x_eval_stan,
            "), ",
            coeffs_stan]

        stan_code_bit = StanCodeBit()
        stan_code_bit.add_code(stan_code)
        return stan_code_bit

    def to_pymc(self):
        pass


class LognormalParameterization(Parameterization):
    """Lognormal distribution"""

    def __init__(self, inputs: TStanable, mu: TStanable, sigma: TStanable):
        Parameterization.__init__(self, [inputs])
        self._mu = mu
        self._sigma = sigma

    def __call__(self, x):
        pass

    def to_stan(self) -> StanCodeBit:
        mu_stan = stanify(self._mu)
        sigma_stan = stanify(self._sigma)
        x_obs_stan = stanify(self._inputs[0])

        stan_code: TListStrStanCodeBit = []
        stan_code += ["lognormal_lpdf(", x_obs_stan, " | ", mu_stan, ", ",
                      sigma_stan, ")"]

        stan_code_bit = StanCodeBit()
        stan_code_bit.add_code(stan_code)

        return stan_code_bit

    def to_pymc(self):
        pass


if __name__ == "__main__":

    invar = "E_true"
    log_e_eval = LogParameterization(invar)
    test_poly_coeffs = [1, 1, 1, 1]
    param = PolynomialParameterization(log_e_eval, test_poly_coeffs)

    invar = "E_reco"
    lognorm = LognormalParameterization(invar, param, param)
    print(lognorm.to_stan())
