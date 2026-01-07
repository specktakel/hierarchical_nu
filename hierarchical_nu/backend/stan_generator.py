"""Module for autogenerating Stan code"""

from typing import Iterable, Union, Sequence
from .code_generator import (
    CodeGenerator,
    ToplevelContextSingleton,
    ContextSingleton,
    ContextStack,
    Contextable,
)
from .stan_code import StanCodeBit
from .expression import (
    TExpression,
    TListTExpression,
    TNamedExpression,
    Expression,
    NamedExpression,
    StringExpression,
    PlainStatement,
)
from .operations import FunctionCall
import logging
import os
import hashlib

logger = logging.getLogger(__name__)

# if TYPE_CHECKING:


__all__ = [
    "StanGenerator",
    "StanFileGenerator",
    "UserDefinedFunction",
    "GeneratedQuantitiesContext",
    "Include",
    "FunctionsContext",
    "DataContext",
    "TransformedDataContext",
    "FunctionCall",
    "ModelContext",
    "WhileLoopContext",
    "IfBlockContext",
    "ElseIfBlockContext",
    "ElseBlockContext",
    "DefinitionContext",
    "ForLoopContext",
    "DummyContext",
]


def stanify(var: TExpression) -> StanCodeBit:
    """Call to_stan function if possible"""
    if isinstance(var, Expression):
        return var.to_stan()

    # Not an Expression, so cast to string
    code_bit = StanCodeBit()
    code_bit.add_code([str(var)])
    return code_bit


class Include(Contextable):

    ORDER = 10

    def __init__(self, file_name: str):
        Contextable.__init__(self)
        self._file_name = file_name

    @property
    def stan_code(self) -> str:
        return "#include " + self._file_name


class FunctionsContext(ToplevelContextSingleton):

    ORDER = 9

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "functions"


class ForLoopContext(Contextable, ContextStack):
    @staticmethod
    def ensure_str(str: TNamedExpression) -> Union[str, int]:
        if isinstance(str, NamedExpression):
            str.add_output("FOR_LOOP_HEADER")
            return str.name
        elif isinstance(str, float):
            raise RuntimeError("For loop range cannot be float")
        else:
            return str

    def __init__(
        self, min_val: TNamedExpression, max_val: TNamedExpression, loop_var_name: str
    ) -> None:

        ContextStack.__init__(self)
        Contextable.__init__(self)

        header_code = "for ({} in {}:{})".format(
            loop_var_name, self.ensure_str(min_val), self.ensure_str(max_val)
        )
        self._loop_var_name = loop_var_name

        self._name = header_code

    def __enter__(self):
        ContextStack.__enter__(self)
        return StringExpression([self._loop_var_name])


class _WhileLoopHeaderContext(Contextable, ContextStack):
    def __init__(self) -> None:

        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = "while"
        self._delimiters = ("(", ")")

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


class WhileLoopContext(Contextable, ContextStack):
    def __init__(self, header_code: Sequence["TExpression"]) -> None:

        header_ctx = _WhileLoopHeaderContext()

        with header_ctx:
            _ = PlainStatement(header_code)

        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = ""

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


class DummyContext(Contextable, ContextStack):
    def __init__(self) -> None:
        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = ""

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


class _IndexingHeaderContext(Contextable, ContextStack):
    def __init__(self) -> None:
        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = ""
        self._delimiters = ("", "")

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


class IndexingContext(Contextable, ContextStack):

    def __init__(self, key) -> None:

        self._name = ""
        self._delimiters = ("", "")

        with _IndexingHeaderContext():
            if isinstance(key, tuple):
                stan_code: TListTExpression = ["["]
                for c, k in enumerate(key, start=-len(key) + 1):
                    if isinstance(k, slice):
                        stan_code += [k.start, ":", k.stop]
                    else:
                        stan_code += [k]
                    # If it's not the last key-entry, add a comma
                    if c != 0:
                        stan_code += [","]
                    # Last entry: close bracket
                    else:
                        stan_code += ["]"]
            elif isinstance(key, slice):
                start = key.start
                stop = key.stop
                stan_code: TListTExpression = ["[", start, ":", stop, "]"]
            else:
                stan_code: TListTExpression = ["[", key, "]"]

        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._idx = stan_code

    def __enter__(self):
        ContextStack.__enter__(self)
        return self._idx


class _IfHeaderContext(Contextable, ContextStack):
    def __init__(self) -> None:
        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = "if"
        self._delimiters = ("(", ")")

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


class _ElseIfHeaderContext(Contextable, ContextStack):
    def __init__(self) -> None:
        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = "else if"
        self._delimiters = ("(", ")")

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


class IfBlockContext(Contextable, ContextStack):
    def __init__(self, header_code: Sequence["TExpression"]) -> None:

        header_ctx = _IfHeaderContext()

        with header_ctx:
            _ = PlainStatement(header_code)

        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = ""

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


class ElseIfBlockContext(Contextable, ContextStack):
    def __init__(self, header_code: Sequence["TExpression"]) -> None:

        header_ctx = _ElseIfHeaderContext()

        with header_ctx:
            _ = PlainStatement(header_code)

        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = ""

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


class ElseBlockContext(Contextable, ContextStack):
    def __init__(self) -> None:

        ContextStack.__init__(self)
        Contextable.__init__(self)

        self._name = "else"

    def __enter__(self):
        ContextStack.__enter__(self)
        return None


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

        self._header_code = return_type + " " + name + "("
        self._header_code += ",".join(
            [
                arg_type + " " + arg_name
                for arg_type, arg_name in zip(arg_types, arg_names)
            ]
        )
        self._header_code += ")"
        self._name = self._header_code

        fc = FunctionsContext()
        context = ContextStack.get_context()

        at_top = False
        # Check if there's another UserDefinedFunction on the stack
        if isinstance(context, UserDefinedFunction):
            logger.debug("Found a function definition inside function")
            # Move ourselves up in the  FunctionsContext object list
            at_top = True

        with fc:
            Contextable.__init__(self, at_top=at_top)

    @property
    def func_name(self):
        return self._func_name

    def __call__(self, *args) -> FunctionCall:
        call = FunctionCall(args, self.func_name, len(args))
        return call

    def __eq__(self, other):
        """
        Two user-defined functions will compare equal, if they have the
        same function header.
        """
        return self.name == other.name

    def __hash__(self):
        """
        The hash is given by the function header
        """
        hash_gen = hashlib.sha256()
        hash_gen.update(self.name.encode())
        return int.from_bytes(hash_gen.digest(), "big")


class DefinitionContext(ContextSingleton):

    ORDER = 8

    def __init__(self):
        ContextSingleton.__init__(self)
        self._name = ""
        self._delimiters = ("", "")


class DataContext(ToplevelContextSingleton):

    ORDER = 7

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "data"


class TransformedDataContext(ToplevelContextSingleton):

    ORDER = 6

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "transformed data"


class ParametersContext(ToplevelContextSingleton):

    ORDER = 5

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "parameters"


class TransformedParametersContext(ToplevelContextSingleton):

    ORDER = 4

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "transformed parameters"


class ModelContext(ToplevelContextSingleton):

    ORDER = 3

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "model"


class GeneratedQuantitiesContext(ToplevelContextSingleton):

    ORDER = 2

    def __init__(self):
        ToplevelContextSingleton.__init__(self)
        self._name = "generated quantities"


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
        logger.debug(
            "Entered recursive parser. Got {} objects".format(len(objects))
        )  # noqa: E501

        code_list = []

        sorted_bits = sorted(objects, reverse=True)
        logger.debug("Objects on stack: {}".format(sorted_bits))
        for code_bit in sorted_bits:
            logger.debug("Currently parsing: {}".format(code_bit))
            if isinstance(code_bit, ContextStack):
                # Encountered a new context, parse before continuing
                new_objects = code_bit.objects
                code_list.append((code_bit, StanGenerator.parse_recursive(new_objects)))
            else:
                if not isinstance(code_bit, Expression):
                    if hasattr(code_bit, "stan_code"):
                        code = code_bit.stan_code + "\n"
                    else:
                        logger.warn(
                            "Encountered a non-expression of type: {}".format(
                                type(code_bit)
                            )
                        )  # noqa: E501
                        continue
                else:
                    # Check whether this Expression is connected
                    logger.debug(
                        "This bit is connected to: {}".format(code_bit.output)
                    )  # noqa: E501

                    # Filter expressions that are connected to an output node,
                    # since they will generate once the output node generates.
                    # Also filter out None outputs (used e.g. for VariableDefs)
                    filtered_outs = [
                        out
                        for out in code_bit.output
                        if (isinstance(out, Expression) or out is None)
                    ]

                    # If at least one output is an expression supress code gen
                    if filtered_outs:
                        continue

                    code_bit = code_bit.to_stan()
                    logger.debug("Adding: {}".format(type(code_bit)))
                    code = code_bit.code + code_bit.end_delim
                code_list.append(code)
        return code_list

    @staticmethod
    def walk_code_list(code_list) -> str:
        code = ""

        for node in code_list:
            if isinstance(node, tuple):
                # encountered a sub-tree

                node_obj, sub_code_list = node

                if hasattr(node_obj, "_delimiters"):
                    ldelim, rdelim = node_obj._delimiters
                else:
                    ldelim, rdelim = "\n{\n", "}\n"

                code += node_obj.name + ldelim
                code += StanGenerator.walk_code_list(sub_code_list)
                code += rdelim
            else:
                code += node  # + "\n"

        return code

    def generate(self) -> str:
        logger.debug("Start parsing")
        code_tree = self.parse_recursive(self.objects)
        # pprint.pprint(code_tree)
        return self.walk_code_list(code_tree)


class StanFileGenerator(StanGenerator):
    def __init__(self, base_filename: str):
        StanGenerator.__init__(self)
        dirname = os.path.dirname(base_filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._base_filename = base_filename

    @staticmethod
    def ensure_filename(obj):
        if hasattr(obj, "name"):
            name = obj.name
        else:
            name = str(obj)
        return name.replace(" ", "")

    def generate_files(self) -> None:
        code_list = self.parse_recursive(self.objects)

        for node in code_list:
            if isinstance(node, tuple):
                code = self.walk_code_list(node[1])
            else:
                code = node
            if code:
                name_ext = self.ensure_filename(node[0])
                with open(self._base_filename + "_" + name_ext + ".stan", "w") as f:
                    f.write(code)

    def generate_single_file(self) -> None:

        code_str = self.generate()

        self.filename = self._base_filename + ".stan"

        with open(self.filename, "w") as f:
            f.write(code_str)
