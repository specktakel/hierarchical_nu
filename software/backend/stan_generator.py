"""Module for autogenerating Stan code"""
from typing import Dict, Iterable
from .code_generator import (
    CodeGenerator, ToplevelContextSingleton, ContextSingleton,
    ContextStack, Contextable)
from .stan_code import StanCodeBit
from .expression import TExpression, Expression
from .operations import FunctionCall
import logging
logger = logging.getLogger(__name__)

# if TYPE_CHECKING:


__all__ = ["StanGenerator", "UserDefinedFunction",
           "GeneratedQuantitiesContext", "Include",
           "FunctionsContext", "DataContext", "DefinitionContext"]


def stanify(var: TExpression) -> StanCodeBit:
    """Call to_stan function if possible"""
    if isinstance(var, Expression):
        return var.to_stan()

    # Not an Expression, so cast to string
    code_bit = StanCodeBit()
    code_bit.add_code([str(var)])
    return code_bit


class Include(Contextable):
    def __init__(
            self,
            file_name: str):
        Contextable.__init__(self)
        self._file_name = file_name

    @property
    def stan_code(self) -> str:
        return "#include "+ self._file_name


class FunctionsContext(ToplevelContextSingleton):
    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "functions"


class UserDefinedFunction(Contextable, ContextStack):
    def __init__(
            self,
            name: str,
            arg_names: Iterable[str],
            arg_types: Iterable[str],
            return_type: str,
            ) -> None:

        ContextStack.__init__(self)
        self._func_name = name
        self._fc = FunctionsContext()

        # Check if there's another UserDefinedFunction on the stack
        context = ContextStack.get_context()

        at_top = False
        if isinstance(context, UserDefinedFunction):
            logger.debug("Found a function definition inside function")
            # Move ourselves up in the  FunctionsContext object list
            at_top = True
        with self._fc:
            # Add ourselves to the Functions context
            Contextable.__init__(self, at_top=at_top)


        """
        if ContextStack.get_context().name != "functions":
            raise RuntimeError("Not in a functions context")
        """
        self._header_code = return_type + " " + name + "("
        self._header_code += ",".join([arg_type+" "+arg_name
                                       for arg_type, arg_name
                                       in zip(arg_types, arg_names)])
        self._header_code += ")"
        self._name = self._header_code

    @property
    def func_name(self):
        return self._func_name

    def __call__(self, *args) -> FunctionCall:
        call = FunctionCall(args, self.func_name, len(args))
        return call


class DataContext(ToplevelContextSingleton):
    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "data"


class GeneratedQuantitiesContext(ToplevelContextSingleton):
    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "generated quantities"


class DefinitionContext(ContextSingleton):
    def __init__(self):
        ContextSingleton.__init__(self)
        self._name = "__DEFS"


class StanGenerator(CodeGenerator):
    """
    Class for autogenerating Stan code
    """

    def __init__(self):
        CodeGenerator.__init__(self)
        self._name = "__TOPLEVEL"

    @property
    def name(self):
        return self._name

    @staticmethod
    def parse_recursive(objects):
        logger.debug("Entered recursive parser. Got {} objects".format(len(objects)))  # noqa: E501
        code_tree: Dict[str, str] = {}
        code_tree["main"] = ""
        for code_bit in objects:
            logger.debug("Currently parsing: {}".format(code_bit))
            if isinstance(code_bit, ContextStack):
                # Encountered a new context, parse before continueing
                objects = code_bit.objects
                code_tree[code_bit] = StanGenerator.parse_recursive(objects)
            else:
                if not isinstance(code_bit, Expression):
                    if hasattr(code_bit, "stan_code"):
                        code = code_bit.stan_code + "\n"
                    else:
                        logger.warn("Encountered a non-expression of type: {}".format(type(code_bit)))  # noqa: E501
                        continue
                else:
                    # Check whether this Expression is connected
                    logger.debug("This bit is connected to: {}".format(code_bit.output))  # noqa: E501

                    filtered_outs = [out for out in code_bit.output if
                                     isinstance(out, Expression)]

                    # If at least one output is an expression supress code gen
                    if filtered_outs:
                        continue

                    code_bit = code_bit.to_stan()
                    logger.debug("Adding: {}".format(type(code_bit)))
                    code = code_bit.code + ";\n"
                code_tree["main"] += code
        return code_tree

    @staticmethod
    def walk_code_tree(code_tree) -> str:
        code = ""
        defs = ""

        node_order = list(code_tree.keys())
        for node in list(node_order):
            if isinstance(node, FunctionsContext):
                node_order.remove(node)
                node_order.insert(0, node)

        for node in node_order:
            leaf = code_tree[node]
            if isinstance(leaf, dict):
                # encountered a sub-tree
                if isinstance(node, DefinitionContext):
                    if len(leaf) != 1:
                        raise RuntimeError("Malformed tree. Definition subtree should have exactly one node.")  # noqa: E501
                    defs += leaf["main"] + ""

                else:
                    code += node.name + "\n{\n"
                    code += StanGenerator.walk_code_tree(leaf)
                    code += "}\n"
            else:
                code += leaf  # + "\n"

        return defs + "\n" + code

    def generate(self) -> str:
        logger.debug("Start parsing")
        code_tree = self.parse_recursive(self.objects)
        return self.walk_code_tree(code_tree)
