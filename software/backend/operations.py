from typing import Sequence
from .expression import Expression, TExpression, TListTExpression
import logging
logger = logging.getLogger(__name__)


class AssignValue(Expression):

    def __init__(self, inputs: Sequence[TExpression], output: TExpression):
        Expression.__init__(self, inputs)

        """
        This Expression is the root node of an expression graph.
        Here we will also generate the output for the LHS, so we need to
        supress code generation for the output node by setting its output
        to this instance.

        """
        if isinstance(output, Expression):
            if output.output:
                # Output node is already connected to something
                logger.debug("Output expression of AssignValue is connected to: {}".format(output.output))  # noqa: E501
            output.add_output(self)

        self._output_val = output
        self._output = []

    def add_output(self, output):
        """
        Ignore output setting.  This allows
        """
        pass
        # raise RuntimeError("Cannot set output for this Expression")

    @property
    def stan_code(self) -> TListTExpression:
        stan_code: TListTExpression = [
            self._output_val, " = ", self._inputs[0]]
        return stan_code

    def to_pymc(self):
        pass


if __name__ == "__main__":
    from .stan_generator import StanGenerator, GeneratedQuantitiesContext

    with StanGenerator() as cg:
        with GeneratedQuantitiesContext() as gq:
            val = AssignValue("a", "b")

        print(cg.generate())
