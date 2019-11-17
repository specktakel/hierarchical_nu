import unittest
from ..stan_generator import (
    StanGenerator, GeneratedQuantitiesContext, DataContext,
    FunctionsContext, Include, UserDefinedFunction)
from ..variable_definitions import ForwardVariableDef


class TestStanGenerator(unittest.TestCase):

    def test_context_order(self):
        with StanGenerator() as cg:
            gc = GeneratedQuantitiesContext()
            fc = FunctionsContext()
            dc = DataContext()

            with dc:
                test = ForwardVariableDef("var1", "real")

            with gc:
                func = UserDefinedFunction("func", ["var1"], ["real"], "real")
                test2 = ForwardVariableDef("var1", "real")
                test2["i"] << func(test)
                test2 << ["1+", test]

            with fc:
                Include("dummy")

        parsed = cg.parse_recursive(cg.objects)
        sort = sorted(cg.objects, reverse=True)
        self.assertEqual(sort, [fc, dc, gc])
        print(cg.generate())


if __name__ == '__main__':
    unittest.main()
