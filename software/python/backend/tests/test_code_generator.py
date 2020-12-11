import unittest
from ..code_generator import (
    ContextSingleton, ToplevelContextSingleton, CodeGenerator, 
    NamedContextSingleton)
from ..baseclasses import NamedObject
import logging


class TestClass(ContextSingleton, NamedObject):
    ORDER = 0

    def __init__(self):
        ContextSingleton.__init__(self)
        self._name = "Test"


class TestClass2(ContextSingleton, NamedObject):
    ORDER = 1

    def __init__(self):
        ContextSingleton.__init__(self)
        self._name = "Test2"


class TestClass3(ToplevelContextSingleton, NamedObject):
    ORDER = 3

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "Test3"


class TestClass4(NamedContextSingleton):
    ORDER = 1

    def __init__(self, name):
        NamedContextSingleton.__init__(self)
        self._name = name


class MyGen(CodeGenerator, NamedObject):

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
        self.assertEqual(test.__dict__, test2.__dict__)
        self.assertNotEqual(test, test3)
        self.assertNotEqual(test.__dict__, test3.__dict__)
        self.assertNotEqual(test4.__dict__, test.__dict__)

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

    def test_named_context_singleton(self):
        with MyGen():
            test = TestClass4("foo")
            test2 = TestClass4("bar")
            test3 = TestClass4("bar")

        self.assertNotEqual(test, test2)
        self.assertEqual(test2, test3)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    unittest.main()
