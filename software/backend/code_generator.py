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

    def add_object(self, obj):
        self._objects.append(obj)

    @property
    def objects(self):
        return self._objects

    @property
    @abstractmethod
    def name(self):
        return self._name


class CodeGenerator(ContextStack, metaclass=ABCMeta):
    """
    Base class for code generators.
    """

    @abstractmethod
    def generate(self):
        pass


class Contextable:
    """
    Mixin class for everything that should be assigned to a certain context
    """

    def __new__(cls, *args, **kwargs):
        inst = object.__new__(cls)
        ctx = ContextStack.get_context()
        ctx.add_object(inst)
        logger.debug("Added object of type {} to context: {}".format(cls, ctx))
        return inst


class ToplevelContextable:
    """
    Mixin class for everything that should be assigned to the toplevel context
    """

    def __new__(cls, *args, **kwargs):
        inst = object.__new__(cls)
        ContextStack.get_context_stack()[0].add_object(inst)
        return inst


class ContextSingleton(Contextable, ContextStack):
    """
    Context singleton class.

    Only one instance of this object can exist
    within a given CodeGenerator context

    When using as mixin, make sure ContextSingleton is first
    """
    __instance__ = None

    def __new__(cls, *args, **kwargs):

        context = ContextStack.get_context()
        for obj in context.objects:
            if isinstance(obj, cls):
                logger.info("Object of type {} already on stack".format(cls))

                return obj
        return super(ContextSingleton, cls).__new__(cls, *args, **kwargs)


class ToplevelContextSingleton(ToplevelContextable, ContextStack):
    """
    Toplevel context singleton class.

    Only one instance of this object can exist
    within the toplevel CodeGenerator context

    When using as mixin, make sure ContextSingleton is first
    """
    __instance__ = None

    def __new__(cls, *args, **kwargs):

        context = ContextStack.get_context_stack()[0]
        for obj in context.objects:
            if isinstance(obj, cls):
                logger.info("Object of type {} already on stack".format(cls))

                return obj
        return super(ToplevelContextSingleton, cls).__new__(
            cls, *args, **kwargs)


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
            ContextSingleton.__init__(self)
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
        with TestClass() as test:
            print("1", id(test))
            with TestClass2() as test2:
                print("2", id(test2))

        with TestClass() as test:
            print("1", id(test))
            with TestClass2() as test2:
                print("2", id(test2))

        with TestClass() as test:
            print("1", id(test))
            with TestClass3() as test3:
                print("3", id(test3))

        with TestClass2() as test:
            with TestClass3() as test2:
                print("3", id(test3))
