from typing import List, Any
from abc import ABCMeta, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ContextStack:
    """
    Similar to pymc3 models, this class implements a context manager
    which allows tracing expressions defined in a code generator context
    """
    STACK: List["CodeGenerator"] = []

    def __init__(self):
        self._objects: List[Any] = []  # TODO: Stricter type checking
        self._name: str = ""

    def __enter__(self):
        ContextStack.STACK.append(self)
        return self

    def __exit__(self, type, value, traceback):
        ContextStack.STACK.pop()

    @classmethod
    def get_context(cls):
        if cls.STACK:
            return cls.STACK[-1]
        else:
            raise RuntimeError("No code generator on stack")

    @classmethod
    def get_context_stack(cls):
        if cls.STACK:
            return cls.STACK
        else:
            raise RuntimeError("No code generator on stack")

    def add_object(self, obj, at_top=False):
        if at_top:
            self._objects.insert(0, obj)
        else:
            self._objects.append(obj)
        logger.debug("Objects in context {}: {}".format(self, self._objects))

    @property
    def objects(self):
        return self._objects

    @property
    def name(self) -> str:
        return self._name


class CodeGenerator(ContextStack, metaclass=ABCMeta):
    """
    Base class for code generators.
    """

    @abstractmethod
    def generate(self):
        pass


class Comparable:
    """Mixin class for making objects comparable"""

    ORDER = 0

    def __lt__(self, other):
        """Allow sorting of Contextables"""
        other_order = 0
        if hasattr(other, "ORDER"):
            other_order = other.ORDER
        logger.debug("Comparing {} < {}".format(self, other))
        logger.debug("My order: {}, other order: {}".format(self.ORDER, other_order))  # noqa: E501
        return self.ORDER < other_order

    def __gt__(self, other):
        """Allow sorting of Contextables"""
        other_order = 0
        if hasattr(other, "ORDER"):
            other_order = other.ORDER
        logger.debug("Comparing {} > {}".format(self, other))
        logger.debug("My order: {}, other order: {}".format(self.ORDER, other_order))  # noqa: E501
        return self.ORDER > other_order


class Contextable(Comparable):
    """
    Mixin class for everything that should be assigned to a certain context

    Parameters:
        at_top: bool
        Add instances at the top of the context objects list
    """

    def __init__(self, at_top=False):
        Comparable.__init__(self)
        ctx = ContextStack.get_context()
        logger.debug("Adding object of type {} to context: {}".format(type(self), ctx))  # noqa: E501
        ctx.add_object(self, at_top)


class ToplevelContextable(Comparable):
    """
    Mixin class for everything that should be assigned to the toplevel context
    """
    """
    def __new__(cls, *args, **kwargs):
        inst = object.__new__(cls)
        ContextStack.get_context_stack()[0].add_object(inst)
        return inst
    """

    def __init__(self):
        Comparable.__init__(self)
        ctx = ContextStack.get_context_stack()[0]
        logger.debug("Adding object of type {} to context: {}".format(type(self), ctx))  # noqa: E501
        ctx.add_object(self)


class ContextSingleton(Contextable, ContextStack):
    """
    Context singleton class.

    Only one instance of this object can exist
    within a given CodeGenerator context

    When using as mixin, make sure ContextSingleton is first
    """

    def __init__(self):
        ContextStack.__init__(self)
        context = ContextStack.get_context()
        for obj in context.objects:
            if isinstance(obj, type(self)):
                logger.info("Object of type {} already on stack".format(type(self)))  # noqa: E501
                self.__dict__ = obj.__dict__
                return
        Contextable.__init__(self)


class ToplevelContextSingleton(ToplevelContextable, ContextStack):
    """
    Toplevel context singleton class.

    Only one instance of this object can exist
    within the toplevel CodeGenerator context

    When using as mixin, make sure ContextSingleton is first
    """

    def __init__(self):
        ContextStack.__init__(self)
        context = ContextStack.get_context_stack()[0]
        for obj in context.objects:
            if isinstance(obj, type(self)):
                logger.info("Object of type {} already on stack".format(type(self)))  # noqa: E501
                self.__dict__ = obj.__dict__
                return
        ToplevelContextable.__init__(self)


if __name__ == "__main__":
    class TestClass(ContextSingleton):
        def __init__(self):
            ContextSingleton.__init__(self)
            self._name = "Test"

        @property
        def name(self):
            return self._name

    class TestClass2(ContextSingleton):
        def __init__(self):
            ContextSingleton.__init__(self)
            self._name = "Test2"

        @property
        def name(self):
            return self._name

    class TestClass3(ToplevelContextSingleton):
        def __init__(self):
            ContextSingleton.__init__(self)
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

        def add_object(self, object):
            self._objects.append(object)

        def generate(self):
            pass

    logging.basicConfig(level=logging.INFO)

    with MyGen() as cg:
        print("Entering TestClass context")
        with TestClass() as test:
            test.data = "test"
            with TestClass2() as test2:
                test2.data = "test2"
        print("Second context")
        with TestClass() as test:
            print(test.data)
            with TestClass2() as test2:
                print(test2.data)
        print("Third context")
        with TestClass() as test:
            print(test.data)
            with TestClass3() as test3:
                test3.data = "test"

        with TestClass2() as test:
            with TestClass3() as test2:
                print(hasattr(test, "data"))
                print(hasattr(test2, "data"))
