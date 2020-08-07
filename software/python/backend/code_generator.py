from typing import List, Any
from abc import ABCMeta, abstractmethod
import logging
import hashlib
from .baseclasses import NamedObject

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
        self._stack_id = len(ContextStack.STACK)

    def __enter__(self):
        ContextStack.STACK.append(self)
        return self

    def __exit__(self, type, value, traceback):
        ContextStack.STACK.pop()

    @property
    def stack_id(self):
        return self._stack_id

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
    Mixin class for everything that should be assigned to a certain context.

    On instantiation, this class fetches the context on top of the context
    stack and adds itself to that context.

    Parameters:
        at_top: bool
        Add instances at the top of the context objects list
    """

    def __init__(self, at_top=False):
        Comparable.__init__(self)
        self._ctx = ContextStack.get_context()
        logger.debug("Adding object of type {} to context: {}".format(type(self), self._ctx))  # noqa: E501
        self._ctx.add_object(self, at_top)

    @property
    def context_id(self):
        return self._ctx.stack_id


class ToplevelContextable(Contextable):
    """
    Mixin class for everything that should be assigned to the toplevel context

    On instantiation, this class fetches the context on top of the context
    stack and adds itself to the top of that context.
    """

    def __init__(self):
        Comparable.__init__(self)
        self._ctx = ContextStack.get_context_stack()[0]
        logger.debug("Adding object of type {} to context: {}".format(type(self), self._ctx))  # noqa: E501
        self._ctx.add_object(self)


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

        # If no object of the same type is in the context, initialize
        # Contextable
        if not self._check_context_and_set_dict(context):
            Contextable.__init__(self)

    def _check_context_and_set_dict(self, context) -> bool:
        """
        Check the given context for an object of type `cls' and copy its dict.

        If no object of type `cls` is on the context stack, return `None`
        """

        for obj in context.objects:
            if type(obj) == type(self):
                logger.info("Object of type {} already on stack".format(self))  # noqa: E501
                self.__dict__ = obj.__dict__
                return True
        return False

    def __eq__(self, other):
        if not isinstance(self, ContextSingleton):
            raise NotImplementedError()

        if type(self) != type(other):
            return False

        return self.context_id == other.context_id

    def __hash__(self):
        return self.context_id


class ToplevelContextSingleton(
        ToplevelContextable, ContextSingleton, ContextStack):
    """
    Toplevel context singleton class.

    Only one instance of this object can exist
    within the toplevel CodeGenerator context

    When using as mixin, make sure ContextSingleton is first
    """

    def __init__(self):
        ContextStack.__init__(self)
        context = ContextStack.get_context_stack()[0]

        if not self._check_context_and_set_dict(context):
            ToplevelContextable.__init__(self)


class NamedContextSingleton(ContextSingleton, NamedObject):
    """
    Named context singleton class

    Only one instance of this object with a given name can exist
    within a given context
    """

    def _check_context_and_set_dict(self, context) -> bool:
        """
        Check the given context for an object of type `cls' and copy its dict.

        If no object of type `cls` is on the context stack, return `None`
        """

        for obj in context.objects:
            if (type(obj) == type(self)) and (obj.name == self.name):
                logger.info("Object of type {} already on stack".format(self))  # noqa: E501
                self.__dict__ = obj.__dict__
                return True
        return False

    def __eq__(self, other):
        return super().__eq__(other) and (self.name == other.name)

    def __hash__(self):
        hash_gen = hashlib.sha256()
        hash_gen.update(str(self.context_id).encode())
        hash_gen.update(self.name.encode())
        return int.from_bytes(hash_gen.digest(), "big")
