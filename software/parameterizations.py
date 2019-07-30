from abc import ABCMeta, abstractmethod
from typing import Union, List, Iterable, Collection
import numpy as np
from stan_generator import (
    StanCodeBit,
    TListStrStanCodeBit,
    StanGenerator,
    Expression,
    TExpression,
    stanify)


class Parameterization(Expression,
                       metaclass=ABCMeta):
    """
    Base class for parameterizations.

    Parameterizations are functions of input variables
    """

    def __init__(self, inputs: List[TExpression]):
        Expression.__init__(self, inputs)

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


def pymcify(var: TExpression):
    """Call to_pymc function if possible"""
    if isinstance(var, Expression):
        return var.to_pymc()

    # Not an Expression, just return
    return var


class LogParameterization(Parameterization):
    """log with customizable base"""
    def __init__(self, inputs: TExpression, base: float = 10):
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
        import theano.tensor as tt
        x_eval_pymc = pymcify(self._inputs[0])

        if self._base != 10:
            return tt.log10(x_eval_pymc)/tt.log10(self._base)
        else:
            return tt.log10(x_eval_pymc)


TArrayOrNumericIterable = Union[np.ndarray, Iterable[float]]


class PolynomialParameterization(Parameterization):
    """Polynomial parametrization"""

    def __init__(
            self,
            inputs: TExpression,
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

    def __init__(self, inputs: TExpression, mu: TExpression,
                 sigma: TExpression):
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


class VMFParameterization(Parameterization):
    """
    Von-Mises-Fisher Distribution
    """

    def __init__(self, inputs: List[TExpression], kappa: TExpression):
        Parameterization.__init__(self, inputs)
        self._kappa = kappa

    def __call__(self, x):
        pass

    def to_stan(self) -> StanCodeBit:
        kappa_stan = stanify(self._kappa)

        x_obs_stan = stanify(self._inputs[0])
        x_true_stan = stanify(self._inputs[1])

        stan_code: TListStrStanCodeBit = []

        stan_code += ["vMF_lpdf(", x_obs_stan, " | ", x_true_stan, ", ",
                      kappa_stan, ")"]

        stan_code_bit = StanCodeBit()
        stan_code_bit.add_code(stan_code)

        return stan_code_bit

    def to_pymc(self):
        pass


class TruncatedParameterization(Parameterization):
    """
    Truncate parameter to range
    """

    def __init__(self, inputs: TExpression, min_val: TExpression,
                 max_val: TExpression):
        """
        Args:
            inputs: TExpression
                Input parameter to truncate
            min_val: TExpression
                Lower bound
            max_val: TExpression
                Upper bound

        """
        Parameterization.__init__(self, [inputs])
        self._min_val = min_val
        self._max_val = max_val

    def __call__(self, x):
        pass

    def to_stan(self) -> StanCodeBit:
        """See base class"""
        min_val_stan = stanify(self._min_val)
        max_val_stan = stanify(self._max_val)

        x_obs_stan = stanify(self._inputs[0])

        stan_code: TListStrStanCodeBit = []

        stan_code += ["truncate_value(", x_obs_stan, ", ", min_val_stan, ", ",
                      max_val_stan, ")"]

        stan_code_bit = StanCodeBit()
        stan_code_bit.add_code(stan_code)

        return stan_code_bit

    def to_pymc(self):
        pass


class MixtureParameterization(Parameterization):
    """
    Mixture model parameterization
    """

    def __init__(
            self,
            inputs: TExpression,
            components: Collection[Parameterization],
            weighting: Union[None, Collection[TExpression]] = None):
        """
        Args:
            inputs: TExpression
                Input parameter to truncate
            components: Collection[TExpression]
                Mixture components. Input[0] of every component gets
                overwritten by inputs[0]
            weighting: Collection[TExpression]
                optional, weights for every component


        """
        Parameterization.__init__(self, [inputs])

        if weighting is not None and len(weighting) != len(components):
            raise ValueError("weights and components have different lengths")
        if weighting is None:
            weighting = ["1./{}".format(len(components))]*len(components)

        self._components = components
        self._weighting = weighting

    def __call__(self, x):
        pass

    def to_stan(self) -> StanCodeBit:
        """See base class"""

        stan_code: TListStrStanCodeBit = []
        for i, (comp, weight) in enumerate(
                zip(self._components, self._weighting)):
            comp._inputs[0] = self._inputs[0]
            comp_stan = stanify(comp)
            weight_stan = stanify(weight)

            stan_code += [weight_stan, " * ", comp_stan]
            if i < len(self._components)-1:
                stan_code += " + "

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

    sum_test = 1 + lognorm
    print(sum_test.to_stan())

    sum_test2 = sum_test + sum_test
    print(sum_test2.to_stan())

    # mixture test

    lognorm_mix = MixtureParameterization(invar, [lognorm, lognorm])
    print(lognorm_mix.to_stan())

    gen = StanGenerator()
    gen.add_code_bit(lognorm_mix.to_stan())
    print(gen.to_stan())
