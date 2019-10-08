"""Module for autogenerating Stan code"""
from typing import Dict
from .code_generator import (
    CodeGenerator, ToplevelContextSingleton, ContextSingleton,
    ContextStack)
from .stan_code import StanCodeBit
from .expression import TExpression, Expression
import logging
logger = logging.getLogger(__name__)

# if TYPE_CHECKING:


__all__ = ["StanGenerator"]


def stanify(var: TExpression) -> StanCodeBit:
    """Call to_stan function if possible"""
    if isinstance(var, Expression):
        return var.to_stan()

    # Not an Expression, so cast to string
    code_bit = StanCodeBit()
    code_bit.add_code([str(var)])
    return code_bit


class FunctionsContext(ToplevelContextSingleton):
    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "Functions"

    @property
    def name(self):
        return self._name


class GeneratedQuantitiesContext(ToplevelContextSingleton):
    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "generated quantities"

    @property
    def name(self):
        return self._name


class DefinitionContext(ContextSingleton):
    def __init__(self):
        ContextSingleton.__init__(self)
        self._name = "__DEFS"

    @property
    def name(self):
        return self._name


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
                    logger.warn("Encountered a non-expression of type: {}".format(type(code_bit)))  # noqa: E501
                    continue
                # Check whether this Expression is connected
                logger.debug("This bit is connected to: {}".format(code_bit.output))  # noqa: E501
                if (code_bit.output is not None
                        and isinstance(code_bit.output, Expression)):
                    continue

                code_bit = code_bit.to_stan()
                logger.debug("Adding: {}".format(code_bit.code))
                code_tree["main"] += code_bit.code +";\n"
        return code_tree

    @staticmethod
    def walk_code_tree(code_tree) -> str:
        code = ""
        defs = ""
        for node, leaf in code_tree.items():
            if isinstance(leaf, dict):
                # encountered a sub-tree
                if isinstance(node, DefinitionContext):
                    if len(leaf) != 1:
                        raise RuntimeError("Malformed tree. Definition subtree should have exactly one node.")
                    defs += leaf["main"] + "\n"

                else:
                    code += node.name + "\n{\n"
                    code += StanGenerator.walk_code_tree(leaf)
                    code += "}"
            else:
                code += leaf + "\n"

        return defs + code

    def generate(self) -> str:
        logger.debug("Start parsing")
        code_tree = self.parse_recursive(self.objects)
        print(code_tree)
        return self.walk_code_tree(code_tree)
