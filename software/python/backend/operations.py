from typing import Sequence, Optional
from .expression import Expression, TExpression, TListTExpression
import logging
logger = logging.getLogger(__name__)

__all__ = ["FunctionCall"]


class FunctionCall(Expression):
    """Simple stan function call"""

    def __init__(
            self,
            inputs: Sequence[TExpression],
            func_name: TExpression,
            nargs: Optional[int] = None):

        if isinstance(func_name, Expression):
            func_name.add_output(self)
        if nargs is None:
            nargs = len(inputs)
        stan_code: TListTExpression = [func_name, "("]
        for i in range(nargs):
            stan_code.append(inputs[i])
            if i != nargs-1:
                stan_code.append(", ")

        stan_code.append(")")

        Expression.__init__(self, inputs, stan_code)
