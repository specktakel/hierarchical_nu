from typing import Sequence
from .expression import Expression, TExpression, TListTExpression
import logging
logger = logging.getLogger(__name__)


class AssignValue(Expression):

    def __init__(self, inputs: Sequence[TExpression], output: TExpression):
        Expression.__init__(self, inputs)
        self._output = output

    @property
    def stan_code(self) -> TListTExpression:
        stan_code: TListTExpression = [self._output, " = ", self._inputs[0]]
        return stan_code

    def to_pymc(self):
        pass


if __name__ == "__main__":
    from .stan_generator import StanGenerator, GeneratedQuantitiesContext

    with StanGenerator() as cg:
        with GeneratedQuantitiesContext() as gq:
            val = AssignValue("a", "b")

        print(cg.generate())
