class Residuals:
    """
    Helper class for calculating residuals of model compared to data
    """
    def __init__(self, data, model):
        """

        Parameters:
        -----------
        data: tuple
            data x- and y coordinates

        model: callable
            A callable encoding the model. The callable should take
            `x` as first argument and a parameter list as second argument.
        """
        self.data_x, self.data_y = data
        self.model = model

    def __call__(self, params):
        """
        Return the residuals w.r.t to model(params)
        """
        expec = self.model(self.data_x, params)

        residuals = expec-self.data_y
        return residuals
