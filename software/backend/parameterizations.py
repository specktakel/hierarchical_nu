from abc import ABCMeta, abstractmethod
from typing import Sequence
from enum import Enum
import numpy as np  # type: ignore
from .stan_generator import UserDefinedFunction, ForLoopContext
from .expression import (Expression, TExpression, TListTExpression,
                         ReturnStatement, StringExpression)
from .operations import FunctionCall
from .pymc_generator import pymcify
from .variable_definitions import StanArray, ForwardVariableDef
from .typedefs import TArrayOrNumericIterable


import logging
logger = logging.getLogger(__name__)

__all__ = ["Parameterization", "LogParameterization",
           "PolynomialParameterization", "LognormalParameterization",
           "VMFParameterization", "TruncatedParameterization",
           "MixtureParameterization", "SimpleHistogram",
           "DistributionMode", "LognormalMixture"
           ]


class Parameterization(Expression,
                       metaclass=ABCMeta):
    """
    Base class for parameterizations.

    Parameterizations are functions of input variables
    """

    pass


DistributionMode = Enum("DistributionMode", "PDF RNG")


class Distribution(Parameterization,
                   metaclass=ABCMeta):
    """
    Class for probility distributions.

    Derived classes should implement an option
    for switching between RNG and PDF mode.
    """

    def __init__(self,
                 inputs: Sequence["TExpression"],
                 mode: DistributionMode):
        Parameterization.__init__(self, inputs)
        self._mode = mode

    @property
    def stan_code(self):
        if self._mode == DistributionMode.PDF:
            return self.stan_code_pdf
        elif self._mode == DistributionMode.RNG:
            return self.stan_code_rng
        else:
            RuntimeError("This should not happen")

    @property
    @abstractmethod
    def stan_code_pdf(self):
        pass

    @property
    @abstractmethod
    def stan_code_rng(self):
        pass


class LogParameterization(Parameterization):
    """log with customizable base"""
    def __init__(self,
                 inputs: TExpression,
                 base: float = 10):
        Parameterization.__init__(self, [inputs])
        self._base = base

    @property
    def stan_code(self) -> TListTExpression:
        """See base class"""
        x_eval_stan = self._inputs[0]
        stan_code: TListTExpression = []
        if self._base != 10:
            base = str(self._base)
            stan_code += ["log10(", x_eval_stan, ") / log10(", base, ")"]
        else:
            stan_code += ["log10(", x_eval_stan, ")"]
        return stan_code

    def to_pymc(self):
        import theano.tensor as tt  # type: ignore
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
        self._coeffs.add_output(self)

    @property
    def stan_code(self) -> TListTExpression:
        """See base class"""

        x_eval_stan = self._inputs[0]

        # TODO: Make sure that eval_poly1d is part of some util lib.
        # Or maybe add a hook for loading ?

        """
        coeffs_stan = "[" + ",".join([str(coeff)
                                      for coeff in self._coeffs]) + "]"
        """
        coeffs_stan = self._coeffs
        stan_code: TListTExpression = [
            "eval_poly1d(",
            x_eval_stan,
            ",",
            coeffs_stan,
            ")",
           ]
        return stan_code

    def to_pymc(self):
        pass


class LognormalParameterization(Distribution):
    """Lognormal distribution"""

    def __init__(self,
                 inputs: TExpression,
                 mu: TExpression,
                 sigma: TExpression,
                 mode: DistributionMode = DistributionMode.PDF
                 ):
        Distribution.__init__(self, [inputs], mode)
        if isinstance(mu, Expression):
            mu.add_output(self)
        if isinstance(sigma, Expression):
            sigma.add_output(self)
        self._mu = mu
        self._sigma = sigma

    def __call__(self, x):
        pass

    @property
    def stan_code_rng(self):
        stan_code: TListTExpression = []
        stan_code += ["lognormal_rng(", self._mu, ", ",
                      self._sigma, ")"]
        return stan_code

    @property
    def stan_code_pdf(self):
        x_obs_stan = self._inputs[0]
        stan_code: TListTExpression = []
        stan_code += ["lognormal_lpdf(", x_obs_stan, " | ", self._mu, ", ",
                      self._sigma, ")"]
        return stan_code

    def to_pymc(self):
        pass


class VMFParameterization(Distribution):
    """
    Von-Mises-Fisher Distribution
    """

    def __init__(
            self,
            inputs: Sequence[TExpression],
            kappa: TExpression,
            mode: DistributionMode = DistributionMode.PDF):
        Distribution.__init__(self, inputs, mode)
        if isinstance(kappa, Expression):
            kappa.add_output(self)
        self._kappa = kappa

    def __call__(self, x):
        pass

    @property
    def stan_code_pdf(self) -> TListTExpression:

        x_obs_stan = self._inputs[0]
        x_true_stan = self._inputs[1]

        stan_code: TListTExpression = []

        stan_code += ["vMF_lpdf(", x_obs_stan, " | ", x_true_stan, ", ",
                      self._kappa, ")"]

        return stan_code

    @property
    def stan_code_rng(self) -> TListTExpression:

        x_true_stan = self._inputs[0]
        stan_code: TListTExpression = []

        stan_code += ["vMF_rng(", x_true_stan, ", ",
                      self._kappa, ")"]

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
        if isinstance(min_val, Expression):
            min_val.add_output(self)
        if isinstance(max_val, Expression):
            max_val.add_output(self)
        self._min_val = min_val
        self._max_val = max_val

    def __call__(self, x):
        pass

    @property
    def stan_code(self) -> TListTExpression:

        x_obs_stan = self._inputs[0]

        stan_code: TListTExpression = []

        stan_code += ["truncate_value(", x_obs_stan, ", ", self._min_val, ", ",
                      self._max_val, ")"]

        return stan_code

    def to_pymc(self):
        pass


class LognormalMixture(UserDefinedFunction):
    """LognormalMixture Mixture Model"""

    def __init__(
            self,
            name: str,
            n_components: int,
            mode: DistributionMode = DistributionMode.PDF):
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
        self.n_components = n_components

        self.means = StringExpression(["means"])
        self.sigmas = StringExpression(["sigmas"])
        self.weights = StringExpression(["weights"])

        val_names = ["x", "means", "sigmas", "weights"]

        val_types = ["real", "vector", "vector", "vector"]

        UserDefinedFunction.__init__(
            self, name, val_names, val_types, "real")

        if mode == DistributionMode.PDF:
            self._build_pdf()
        elif mode == DistributionMode.RNG:
            self._build_rng()
        else:
            raise RuntimeError("This should not happen")

    def _build_rng(self):
        with self:
            index = ForwardVariableDef("index", "int")

            index << ["categorical_rng(", self.weights, ")"]
            distribution = LognormalParameterization(
                    "x",
                    self.means[index],
                    self.sigmas[index],
                    mode=DistributionMode.RNG)
            ReturnStatement([distribution])

    def _build_pdf(self):

        with self:
            result = ForwardVariableDef(
                "result",
                "vector["+str(self.n_components)+"]")

            log_weights = FunctionCall([self.weights], "log")
            with ForLoopContext(1, self.n_components, "i") as i:
                distribution = LognormalParameterization(
                    "x",
                    self.means[i],
                    self.sigmas[i],
                    mode=DistributionMode.PDF)

                result[i] << [log_weights[i]+distribution]

            result_sum = ["log_sum_exp(", result, ")"]

            ReturnStatement(result_sum)

    @property
    def stan_code_pdf(self) -> TListTExpression:
        raise NotImplementedError("This should never be called")

    @property
    def stan_code_rng(self) -> TListTExpression:
        raise NotImplementedError("This should never be called")

    def to_pymc(self):
        pass


class MixtureParameterization(Distribution, UserDefinedFunction):
    """
    Mixture model parameterization
    """

    pass


class SimpleHistogram(UserDefinedFunction):
    """
    A step function implemented as lookup table
    """
    def __init__(
            self,
            histogram: np.ndarray,
            binedges: Sequence[np.ndarray],
            name: str
            ):

        self._dim = len(binedges)

        val_names = ["value_{}".format(i) for i in range(self._dim)]
        val_types = ["real"]*self._dim

        UserDefinedFunction.__init__(
            self, name, val_names, val_types, "real")

        with self:
            self._histogram = StanArray("hist_array", "real", histogram)
            self._binedges = [
                StanArray("hist_edge_{}".format(i), "real", be)
                for i, be in enumerate(binedges)]

            stan_code: TListTExpression = [self._histogram]
            for i in range(self._dim):
                stan_code += ["[binary_search(", val_names[i], ", ",
                              self._binedges[i], ")]"]
            _ = ReturnStatement(stan_code)

        """
        self._histogram.add_output(self)
        for be in self._binedges:
            be.add_output(self)
        """


if __name__ == "__main__":

    from .stan_generator import StanGenerator, GeneratedQuantitiesContext

    logging.basicConfig(level=logging.DEBUG)

    with StanGenerator() as cg:
        with GeneratedQuantitiesContext() as gq:
            """
            invar = "E_true"
            log_e_eval = LogParameterization(invar)
            test_poly_coeffs = [1, 1, 1, 1]
            param = PolynomialParameterization(log_e_eval, test_poly_coeffs,
                                               "test_poly_coeffs")

            invar = "E_reco"
            lognorm = ForwardVariableDef("lognorm", "real")
            lognorm_func = LognormalParameterization(invar, param, param)
            _ = AssignValue([lognorm_func], lognorm)

            lognorm_sum_def = ForwardVariableDef("lognorm_sum", "real")
            sum_test = 1 + lognorm_func
            lognorm_sum = AssignValue([sum_test], lognorm_sum_def)

            sum_test2 = sum_test + sum_test
            lognorm_sum = AssignValue([sum_test2], lognorm_sum_def)
            """
            lognorm_means = StanArray("means", "real", [1, 1])
            lognorm_sigmas = StanArray("sigmas", "real", [1, 1])
            lognorm_weights = StanArray("weights", "real", [0.5, 0.5])
            lognorm_mix = LognormalMixture(
                "lognorm",
                2,
                )
            lognorm_mix_rng = LognormalMixture(
                "lognorm",
                2,
                mode=DistributionMode.RNG)
            lognorm_mix_def = ForwardVariableDef("lognorm_mix", "real")
            

            lognorm_mix_def << lognorm_mix("x",
                                           lognorm_means,
                                           lognorm_sigmas,
                                           lognorm_weights
                                           )

            lognorm_mix_rng_def = ForwardVariableDef("lognorm_mix_rng", "real")
            
            lognorm_mix_rng_def << lognorm_mix_rng("x",
                                                   lognorm_means,
                                                   lognorm_sigmas,
                                                   lognorm_weights
                                                   )

            """
            # histogram test

            hist_var = np.zeros((4, 5), dtype=float)
            binedges_x = np.arange(5, dtype=float)
            binedges_y = np.arange(6, dtype=float)

            values = ["x", "y"]

            hist = SimpleHistogram(hist_var, [binedges_x, binedges_y],
                                   "hist")

            called_hist = ForwardVariableDef("called_hist", "real")
            hist_call = hist(*values)
            called_hist = AssignValue([hist_call], called_hist)
            """
        print(cg.generate())
