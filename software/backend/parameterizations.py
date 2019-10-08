from abc import ABCMeta
from typing import Union, List, Sequence, Iterable
import numpy as np  # type: ignore
from .stan_generator import stanify
from .stan_code import TListStrStanCodeBit
from .expression import (Expression, TExpression,
                         StanFunction)
from .pymc_generator import pymcify
from .variable_definitions import StanArray, VariableDef
from .typedefs import TArrayOrNumericIterable

__all__ = ["Parameterization", "LogParameterization",
           "PolynomialParameterization", "LognormalParameterization",
           "VMFParameterization", "TruncatedParameterization",
           "MixtureParameterization", "SimpleHistogram",
           "FunctionCall", "StanExpressionFunction"]


class Parameterization(Expression,
                       metaclass=ABCMeta):
    """
    Base class for parameterizations.

    Parameterizations are functions of input variables
    """

    pass


class FunctionCall(Parameterization):
    """Simple stan function call"""

    def __init__(
            self,
            inputs: Sequence[TExpression],
            func_name: TExpression,
            nargs: int = 1):
        Parameterization.__init__(self, inputs)
        self._func_name = func_name
        self._nargs = nargs

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        """See base class"""

        stan_code: TListStrStanCodeBit = [self._func_name, "("]

        for i in range(self._nargs):
            stan_code.append(stanify(self._inputs[i]))
            if i != self._nargs-1:
                stan_code.append(", ")

        stan_code.append(");")
        return stan_code

    def to_pymc(self):
        pass


class StanExpressionFunction(StanFunction, VariableDef):
    """Encapsulates an expression as function"""
    def __init__(
            self,
            name: str,
            arg_names: Iterable[str],
            arg_types: Iterable[str],
            return_type: str,
            expression: "TExpression") -> None:
        header_code = return_type + " " + name + "("
        header_code += ",".join([arg_type+" "+arg_name for arg_type, arg_name
                                 in zip(arg_types, arg_names)])
        header_code += ")"
        StanFunction.__init__(self, name, header_code)
        VariableDef.__init__(self, name)

        self.add_func_code(["return ", stanify(expression), ";"])
        self.add_stan_hook(name, "function", self)

    @property
    def def_code(self):
        return ""


class LogParameterization(Parameterization):
    """log with customizable base"""
    def __init__(self, inputs: TExpression, base: float = 10):
        Parameterization.__init__(self, [inputs])
        self._base = base

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        """See base class"""
        x_eval_stan = stanify(self._inputs[0])
        stan_code: TListStrStanCodeBit = []
        if self._base != 10:
            base = str(self._base)
            stan_code += ["log10(", x_eval_stan, "}) / log10(", base, ")"]
        else:
            stan_code += ["log10(", x_eval_stan, "})"]
        return stan_code

    def to_pymc(self):
        import theano.tensor as tt
        x_eval_pymc = pymcify(self._inputs[0])

        if self._base != 10:
            return tt.log10(x_eval_pymc)/tt.log10(self._base)
        else:
            return tt.log10(x_eval_pymc)


class PolynomialParameterization(Parameterization):
    """Polynomial parametrization"""

    def __init__(
            self,
            inputs: TExpression,
            coefficients: TArrayOrNumericIterable,
            coeffs_var_name: str) -> None:

        Parameterization.__init__(self, [inputs])
        self._coeffs = StanArray(
            coeffs_var_name,
            "vector",
            coefficients)

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        """See base class"""

        x_eval_stan = stanify(self._inputs[0])

        # TODO: Make sure that eval_poly1d is part of some util lib.
        # Or maybe add a hook for loading ?

        """
        coeffs_stan = "[" + ",".join([str(coeff)
                                      for coeff in self._coeffs]) + "]"
        """
        coeffs_stan = stanify(self._coeffs)
        stan_code: TListStrStanCodeBit = [
            "eval_poly1d(",
            x_eval_stan,
            ",",
            coeffs_stan,
            ")",
           ]
        return stan_code

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

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        """See base class"""
        mu_stan = stanify(self._mu)
        sigma_stan = stanify(self._sigma)
        x_obs_stan = stanify(self._inputs[0])

        stan_code: TListStrStanCodeBit = []
        stan_code += ["lognormal_lpdf(", x_obs_stan, " | ", mu_stan, ", ",
                      sigma_stan, ")"]

        return stan_code

    def to_pymc(self):
        pass


class VMFParameterization(Parameterization):
    """
    Von-Mises-Fisher Distribution
    """

    def __init__(self, inputs: Sequence[TExpression], kappa: TExpression):
        Parameterization.__init__(self, inputs)
        self._kappa = kappa

    def __call__(self, x):
        pass

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        kappa_stan = stanify(self._kappa)

        x_obs_stan = stanify(self._inputs[0])
        x_true_stan = stanify(self._inputs[1])

        stan_code: TListStrStanCodeBit = []

        stan_code += ["vMF_lpdf(", x_obs_stan, " | ", x_true_stan, ", ",
                      kappa_stan, ")"]

        return stan_code

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

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        min_val_stan = stanify(self._min_val)
        max_val_stan = stanify(self._max_val)

        x_obs_stan = stanify(self._inputs[0])

        stan_code: TListStrStanCodeBit = []

        stan_code += ["truncate_value(", x_obs_stan, ", ", min_val_stan, ", ",
                      max_val_stan, ")"]

        return stan_code

    def to_pymc(self):
        pass


class MixtureParameterization(Parameterization):
    """
    Mixture model parameterization
    """

    def __init__(
            self,
            inputs: TExpression,
            components: Sequence[Parameterization],
            weighting: Union[None, Sequence[TExpression]] = None):
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

        self._components: List[Parameterization] = list(components)
        self._weighting: List[TExpression] = list(weighting)

    def __call__(self, x):
        pass

    @property
    def stan_code(self) -> TListStrStanCodeBit:

        expression: TExpression = 0
        for i, (comp, weight) in enumerate(
                zip(self._components, self._weighting)):
            comp._inputs[0] = self._inputs[0]
            # Here we implictely use the OperatorExpression, since
            # components are Parameterizations

            expression += comp*weight  # type: ignore

        return [expression.to_stan()]  # type: ignore

    def to_pymc(self):
        pass


class SimpleHistogram(Parameterization):
    """
    A step function implemented as lookup table
    """
    def __init__(
            self,
            inputs: Sequence[TExpression],
            histogram: np.ndarray,
            binedges: Sequence[np.ndarray],
            hist_var_name: str
            ):
        Parameterization.__init__(self, inputs)
        self._dim = len(binedges)
        self._histogram = StanArray(hist_var_name, "real", histogram)
        self._binedges = [
            StanArray(hist_var_name + "edge_{}".format(i), "real", be)
            for i, be in enumerate(binedges)]

    @property
    def stan_code(self) -> TListStrStanCodeBit:
        """See base class"""
        stan_code: TListStrStanCodeBit = [stanify(self._histogram)]
        for i in range(self._dim):
            value_stan = stanify(self._inputs[i])
            binedge_stan = stanify(self._binedges[i])
            stan_code += ["[binary_search(", value_stan, ", ",
                          binedge_stan, ")]"]

        return stan_code

    def to_pymc(self):
        """
        Convert the parametrizaton to PyMC3
        """
        pass


if __name__ == "__main__":

    from .stan_generator import StanGenerator
    with StanGenerator() as cg:
        invar = "E_true"
        log_e_eval = LogParameterization(invar)
        test_poly_coeffs = [1, 1, 1, 1]
        param = PolynomialParameterization(log_e_eval, test_poly_coeffs,
                                           "test_poly_coeffs")

        invar = "E_reco"
        lognorm = LognormalParameterization(invar, param, param)

        print(cg.generate())

    with StanGenerator() as cg:
        sum_test = 1 + lognorm
        sum_test2 = sum_test + sum_test

    print(cg.to_stan()[1])

    # mixture test

    lognorm_mix = MixtureParameterization(invar, [lognorm, lognorm])
    print(lognorm_mix.to_stan())

    gen = StanGenerator()
    gen.add_code_bit(lognorm_mix.to_stan())
    print(gen.to_stan())

    print("Hist test")
    # histogram test

    hist_var = np.zeros((4, 5), dtype=float)
    binedges_x = np.arange(5, dtype=float)
    binedges_y = np.arange(6, dtype=float)

    values = ["x", "y"]

    hist = SimpleHistogram(values, hist_var, [binedges_x, binedges_y],
                           "hist")
    print(hist.to_stan())

    hist_func = StanExpressionFunction(
        "test_hist", ["x", "y"], ["real", "real"], "real", hist)

    hist_call = FunctionCall(values, hist_func, 2)

    print(hist_call.to_stan())