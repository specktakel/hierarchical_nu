from expression import TExpression, Expression


def pymcify(var: TExpression):
    """Call to_pymc function if possible"""
    if isinstance(var, Expression):
        return var.to_pymc()

    # Not an Expression, just return
    return var
