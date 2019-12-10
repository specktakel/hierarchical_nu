import unittest
import pystan
from ..parameterizations import PolynomialParameterization
from ..stan_generator import (
    StanGenerator, GeneratedQuantitiesContext, DataContext,
    FunctionsContext, Include)
from ..variable_definitions import ForwardVariableDef
import numpy as np


class TestPolynomialParameterization(unittest.TestCase):

    def test_polynomial_stan_code(self):
        with StanGenerator() as cg:
            with FunctionsContext():
                _ = Include("utils.stan")
            with DataContext():
                test_val = ForwardVariableDef("test_val", "real")
            with GeneratedQuantitiesContext():
                test_poly_coeffs = [1, 2, 3, 4]
                result = ForwardVariableDef("result", "real")
                poly = PolynomialParameterization(
                    test_val, test_poly_coeffs, "test_poly_coeffs")
                result << poly
            code = cg.generate()

        sm = pystan.StanModel(
            model_code=code,
            include_paths=["../dev/statistical_model/4_tracks_and_cascades/stan/"],
            verbose=False)
        data = {"test_val": 1}
        fit = sm.sampling(data=data, iter=1, chains=1, algorithm="Fixed_param")
        fit = fit.extract()
        self.assertEqual(fit["result"],
                         np.poly1d(test_poly_coeffs)(data["test_val"]))


if __name__ == '__main__':
    unittest.main()
