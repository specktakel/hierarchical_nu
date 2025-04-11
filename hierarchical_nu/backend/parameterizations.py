from abc import ABCMeta, abstractmethod
from typing import Sequence
from enum import Enum
from hierarchical_nu.backend.expression import TListTExpression
import numpy as np  # type: ignore
from .stan_generator import UserDefinedFunction, ForLoopContext
from .expression import (
    Expression,
    TExpression,
    TListTExpression,
    ReturnStatement,
    StringExpression,
)
from .operations import FunctionCall
from .pymc_generator import pymcify
from .variable_definitions import (
    StanArray,
    ForwardVariableDef,
    ForwardArrayDef,
    ForwardVectorDef,
)
from .typedefs import TArrayOrNumericIterable


import logging

logger = logging.getLogger(__name__)

__all__ = [
    "LogParameterization",
    "RayleighParameterization",
    "PolynomialParameterization",
    "LognormalParameterization",
    "VMFParameterization",
    "TruncatedParameterization",
    "SimpleHistogram",
    "SimpleHistogram_rng",
    "DistributionMode",
    "LognormalMixture",
    "TwoDimHistInterpolation",
]

DistributionMode = Enum("DistributionMode", "PDF RNG")


class Distribution(Expression, metaclass=ABCMeta):
    """
    Class for probility distributions.

    Derived classes should implement an option
    for switching between RNG and PDF mode.
    """

    def __init__(self, inputs: TListTExpression, mode: DistributionMode):
        self._mode = mode

        if self._mode == DistributionMode.PDF:
            stan_code = self.stan_code_pdf(inputs)
        elif self._mode == DistributionMode.RNG:
            stan_code = self.stan_code_rng(inputs)

        Expression.__init__(self, inputs, stan_code)

    @abstractmethod
    def stan_code_pdf(self, inputs: TListTExpression):
        pass

    @abstractmethod
    def stan_code_rng(self, inputs: TListTExpression):
        pass


class LogParameterization(Expression):
    """log with customizable base"""

    def __init__(self, inputs: TExpression, base: float = 10):
        self._base = base

        x_eval_stan = inputs
        stan_code: TListTExpression = []
        if self._base != 10:
            base_str = str(self._base)
            stan_code += ["log10(", x_eval_stan, ") / log10(", base_str, ")"]
        else:
            stan_code += ["log10(", x_eval_stan, ")"]
        Expression.__init__(self, [inputs], stan_code)

    def to_pymc(self):
        import theano.tensor as tt  # type: ignore

        x_eval_pymc = pymcify(self._inputs[0])

        if self._base != 10:
            return tt.log10(x_eval_pymc) / tt.log10(self._base)
        else:
            return tt.log10(x_eval_pymc)


class PolynomialParameterization(Expression):
    """Polynomial parametrization"""

    def __init__(
        self,
        inputs: TExpression,
        coefficients: TArrayOrNumericIterable,
        coeffs_var_name: str,
    ) -> None:

        coeffs = StanArray(coeffs_var_name, "vector", coefficients)
        coeffs.add_output(self)

        x_eval_stan = inputs

        stan_code: TListTExpression = [
            "eval_poly1d(",
            x_eval_stan,
            ",",
            coeffs,
            ")",
        ]

        Expression.__init__(self, [inputs], stan_code)


class LognormalParameterization(Distribution):
    """Lognormal distribution"""

    def __init__(
        self,
        inputs: TExpression,
        mu: TExpression,
        sigma: TExpression,
        mode: DistributionMode = DistributionMode.PDF,
    ):
        if isinstance(mu, Expression):
            mu.add_output(self)
        if isinstance(sigma, Expression):
            sigma.add_output(self)
        self._mu = mu
        self._sigma = sigma
        Distribution.__init__(self, [inputs], mode)

    def __call__(self, x):
        pass

    def stan_code_rng(self, inputs: TListTExpression):
        stan_code: TListTExpression = []
        stan_code += ["lognormal_rng(", self._mu, ", ", self._sigma, ")"]
        return stan_code

    def stan_code_pdf(self, inputs: TListTExpression):
        x_obs_stan = inputs[0]
        stan_code: TListTExpression = []
        stan_code += [
            "lognormal_lpdf(",
            x_obs_stan,
            " | ",
            self._mu,
            ", ",
            self._sigma,
            ")",
        ]
        return stan_code


class RayleighParameterization(Distribution):
    """
    Rayleigh Distribution.
    Additional 1 / (sin(x) * 2 * pi) is included to account for
    the spherical coordinate system in which the likelihood is defined.
    I am not amused to write this class with a 'z'.
    """

    def __init__(
        self,
        inputs: TListTExpression,
        sigma: TExpression,
        mode: DistributionMode = DistributionMode.PDF,
    ):
        self._sigma = sigma
        Distribution.__init__(self, inputs, mode)

    def __call__(self):
        pass

    def stan_code_rng(self, inputs: TListTExpression) -> TListTExpression:
        # Needs true direction and sigma
        x_true_stan = inputs[0]
        stan_code: TListTExpression = []
        stan_code += ["rayleigh_deflected_rng(", x_true_stan, ", ", self._sigma, ")"]
        return stan_code

    def stan_code_pdf(self, inputs: TListTExpression) -> TListTExpression:
        ang_sep = inputs[0]

        stan_code: TListTExpression = []

        stan_code += [
            "log(",
            ang_sep,
            " / sin(",
            ang_sep,
            ")) - log(2 * pi() * ",
            self._sigma,
            ") -  0.5 * (pow(",
            ang_sep,
            ", 2) / ",
            self._sigma,
            ")",
        ]

        return stan_code


class VMFParameterization(Distribution):
    """
    Von-Mises-Fisher Distribution
    """

    def __init__(
        self,
        inputs: TListTExpression,
        kappa: TExpression,
        mode: DistributionMode = DistributionMode.PDF,
    ):

        self._kappa = kappa
        if isinstance(kappa, Expression):
            kappa.add_output(self)
        Distribution.__init__(self, inputs, mode)

    def __call__(self, x):
        pass

    def stan_code_pdf(self, inputs) -> TListTExpression:

        x_obs_stan = inputs[0]
        x_true_stan = inputs[1]

        stan_code: TListTExpression = []

        stan_code += [
            "vMF_lpdf(",
            x_obs_stan,
            " | ",
            x_true_stan,
            ", ",
            self._kappa,
            ")",
        ]

        return stan_code

    def stan_code_rng(self, inputs) -> TListTExpression:

        x_true_stan = inputs[0]
        stan_code: TListTExpression = []

        stan_code += ["vMF_rng(", x_true_stan, ", ", self._kappa, ")"]

        return stan_code

    def to_pymc(self):
        pass


class TruncatedParameterization(Expression):
    """
    Truncate parameter to range
    """

    def __init__(self, inputs: TExpression, min_val: TExpression, max_val: TExpression):
        """
        Args:
            inputs: TExpression
                Input parameter to truncate
            min_val: TExpression
                Lower bound
            max_val: TExpression
                Upper bound

        """

        if isinstance(min_val, Expression):
            min_val.add_output(self)
        if isinstance(max_val, Expression):
            max_val.add_output(self)

        x_obs_stan = inputs

        stan_code: TListTExpression = []

        stan_code += ["truncate_value(", x_obs_stan, ", ", min_val, ", ", max_val, ")"]

        Expression.__init__(self, [inputs], stan_code)


class SimpleHistogram_rng(UserDefinedFunction):
    """
    Callable histogram rng
    Function signature is bin counts and bin edges.
    """

    def __init__(
        self,
        name: str,
    ):
        super().__init__(
            name, ["hist_array", "hist_edges"], ["array[] real", "array[] real"], "real"
        )

        with self:
            self._bin_width = ForwardArrayDef(
                "bin_width", "real", ["[size(hist_array)]"]
            )
            self._multiplied = ForwardArrayDef(
                "multiplied", "real", ["[size(hist_array)]"]
            )
            self._normalised = ForwardVectorDef("normalised", ["size(hist_array)"])
            with ForLoopContext(2, "size(hist_edges)", "i") as i:
                self._bin_width[i - 1] << StringExpression(
                    ["hist_edges[i] - hist_edges[i-1]"]
                )
            with ForLoopContext(1, "size(hist_array)", "i") as i:
                self._multiplied[i] << StringExpression(
                    ["hist_array[i] * bin_width[i]"]
                )
            with ForLoopContext(1, "size(hist_array)", "i") as i:
                self._normalised[i] << StringExpression(
                    ["hist_array[i] / sum(multiplied)"]
                )

            index = ForwardVariableDef("index", "int")
            index << StringExpression(["categorical_rng(", self._normalised, ")"])

            _ = ReturnStatement(["uniform_rng(hist_edges[index], hist_edges[index+1])"])


class LognormalMixture(UserDefinedFunction):
    """LognormalMixture Mixture Model"""

    def __init__(
        self,
        name: str,
        n_components: int,
        mode: DistributionMode = DistributionMode.PDF,
    ):
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

        if mode == DistributionMode.PDF:
            val_names = ["x", "means", "sigmas", "weights"]
            val_types = ["real", "vector", "vector", "vector"]
        elif mode == DistributionMode.RNG:
            val_names = ["means", "sigmas", "weights"]
            val_types = ["vector", "vector", "vector"]
        else:
            raise RuntimeError("This should not happen")

        UserDefinedFunction.__init__(self, name, val_names, val_types, "real")

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
                [], self.means[index], self.sigmas[index], mode=DistributionMode.RNG
            )
            ReturnStatement([distribution])

    def _build_pdf(self):

        with self:
            result = ForwardVariableDef(
                "result", "vector[" + str(self.n_components) + "]"
            )

            log_weights = FunctionCall([self.weights], "log")
            with ForLoopContext(1, self.n_components, "i") as i:
                distribution = LognormalParameterization(
                    "x", self.means[i], self.sigmas[i], mode=DistributionMode.PDF
                )

                result[i] << [log_weights[i] + distribution]

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


class SimpleHistogram(UserDefinedFunction):
    """
    A step function implemented as lookup table
    """

    def __init__(
        self, histogram: np.ndarray, binedges: Sequence[np.ndarray], name: str
    ):

        self._dim = len(binedges)

        val_names = ["value_{}".format(i) for i in range(self._dim)]
        val_types = ["real"] * self._dim

        UserDefinedFunction.__init__(self, name, val_names, val_types, "real")

        with self:
            self._histogram = StanArray("hist_array", "real", histogram)
            self._binedges = [
                StanArray("hist_edge_{}".format(i), "real", be)
                for i, be in enumerate(binedges)
            ]

            stan_code: TListTExpression = [self._histogram]
            for i in range(self._dim):
                stan_code += [
                    "[binary_search(",
                    val_names[i],
                    ", ",
                    self._binedges[i],
                    ")]",
                ]
            _ = ReturnStatement(stan_code)

        """
        self._histogram.add_output(self)
        for be in self._binedges:
            be.add_output(self)
        """


class TwoDimHistInterpolation(UserDefinedFunction):
    """
    2D Histogram that interpolates the returned value along the first axis
    Used for interpolating effective areas. Assumes linear energy as first axis
    and takes log of energy for interpolation.
    """

    def __init__(
        self, histogram: np.ndarray, binedges: Sequence[np.ndarray], name: str
    ):

        if not histogram.ndim == 2:
            raise ValueError("Only two dimensional histograms can be interpolated")
        self._dim = 2

        val_names = ["value_{}".format(i) for i in range(self._dim)]
        val_types = ["real"] * self._dim

        UserDefinedFunction.__init__(self, name, val_names, val_types, "real")

        with self:
            self._histogram = StanArray("hist_array", "real", histogram)
            self._binedges = [
                StanArray("hist_edge_{}".format(i), "real", be)
                for i, be in enumerate(binedges)
            ]
            loge_c = np.power(
                10, (np.log10(binedges[0][:-1]) + np.log10(binedges[0][1:])) / 2
            )
            self._loge = StanArray("energy", "real", loge_c)
            temp = ForwardArrayDef("temp", "real", ["[", binedges[0].size - 1, "]"])
            temp << StringExpression(
                [
                    "hist_array[:, binary_search(",
                    val_names[1],
                    ", ",
                    self._binedges[1],
                    ")]",
                ]
            )
            return_this: TListTExpression = [
                "interpolate(to_vector(energy), to_vector(temp), value_0)"
            ]
            _ = ReturnStatement(return_this)


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
            lognorm_mix_rng = LognormalMixture("lognorm", 2, mode=DistributionMode.RNG)
            lognorm_mix_def = ForwardVariableDef("lognorm_mix", "real")

            lognorm_mix_def << lognorm_mix(
                "x", lognorm_means, lognorm_sigmas, lognorm_weights
            )

            lognorm_mix_rng_def = ForwardVariableDef("lognorm_mix_rng", "real")

            lognorm_mix_rng_def << lognorm_mix_rng(
                "x", lognorm_means, lognorm_sigmas, lognorm_weights
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
