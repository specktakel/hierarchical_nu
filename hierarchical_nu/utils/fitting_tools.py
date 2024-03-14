from scipy.interpolate import PchipInterpolator
import numpy as np
from scipy.integrate import quad


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

        residuals = expec - self.data_y
        return residuals


class Spline1D:
    def __init__(self, f, x_edges, norm: bool = True):
        self.x_edges = x_edges

        try:
            self.x_c = x_edges[:-1] + np.diff(x_edges) / 2
            self.f = f
            self.x_min = self.x_c[0]
            self.x_max = self.x_c[-1]
            self.spline = PchipInterpolator(self.x_c, f, extrapolate=False)
            if norm:
                self.norm = quad(
                    self.spline, self.x_min, self.x_max, limit=200, full_output=1
                )[0]
            self._return_zeros = False
        except IndexError:
            # Histogram is degenerate
            if norm:
                self.norm = 1.0
            self._return_zeros = True

    def __call__(self, x):
        x = np.atleast_1d(x)
        if self._return_zeros:
            return np.zeros_like(x)

        output = self.spline(x)
        output = np.where(np.isnan(output), 0, output)
        return output
