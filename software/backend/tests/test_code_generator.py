import unittest
from ..code_generator import (
    ContextSingleton, ToplevelContextSingleton, CodeGenerator)


class TestClass(ContextSingleton):
    ORDER = 0

    def __init__(self):
        ContextSingleton.__init__(self)
        self._name = "Test"

    @property
    def name(self):
        return self._name


class TestClass2(ContextSingleton):
    ORDER = 1

    def __init__(self):
        ContextSingleton.__init__(self)
        self._name = "Test2"

    @property
    def name(self):
        return self._name


class TestClass3(ToplevelContextSingleton):
    ORDER = 3

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "Test3"

    @property
    def name(self):
        return self._name


class MyGen(CodeGenerator):

    def __init__(self):
        CodeGenerator.__init__(self)
        self._name = "TOPLEVEL"

    @property
    def name(self):
        return self._name

    def generate(self):
        pass


class TestContextSingleton(unittest.TestCase):
    def test_context_singleton(self):
        with MyGen():
            with TestClass() as test:
                test.data = "foo"
            with TestClass() as test2:
                test2.data = "bar"
            with TestClass2() as test3:
                test3.data = "bar"
                with TestClass() as test4:
                    test4.data = "baz"

        self.assertEqual(test, test2)
        self.assertNotEqual(test, test3)
        self.assertNotEqual(test4, test)

    def test_toplevel_context_singleton(self):
        with MyGen():            
            with TestClass3() as test:
                test.data = "foo"

            with TestClass():
                with TestClass3() as test2:
                    test2.data = "bar"

        self.assertEqual(test, test2)

    def test_contextable_ordering(self):
        with MyGen():
            test = TestClass()
            test2 = TestClass2()

        self.assertGreater(test2, test)


if __name__ == '__main__':
    unittest.main()
