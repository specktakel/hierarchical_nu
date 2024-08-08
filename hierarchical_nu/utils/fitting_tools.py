from scipy.interpolate import PchipInterpolator
import numpy as np
from scipy.integrate import quad

from abc import ABCMeta, abstractmethod


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


class PowerLawSegment:
    def __init__(self, xmin, xmax, slope, val, low=True):
        """
        :param xmin: lower bound of definition
        :param xmax: upper bound of definition
        :param slope: exponent of powerlaw, assuming x^slope
        :param val_low: power law evaluated at xmin
        """

        self._xmin = xmin
        self._xmax = xmax
        self._slope = slope
        self._low = low
        self._val = val
        if self._low:
            self._x0 = self._xmin
        else:
            self._x0 = self._xmax

        self._val_low = self.__call__(self._xmin)

    def __call__(self, x):
        return self._val * np.power(x / self._x0, self.slope)

    @property
    def integral(self):
        if self._slope == -1.0:
            return self.val_low * self.xmin * np.log(self.xmax / self.xmin)
        return (
            self.val_low
            * self.xmin
            / (1.0 + self.slope)
            * (np.power(self.xmax / self.xmin, self.slope + 1) - 1.0)
        )

    @property
    def slope(self):
        return self._slope

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def val_low(self):
        return self._val_low


class SegmentedApprox(metaclass=ABCMeta):
    def __init__(
        self,
        target,
        support,
        bins,
    ):
        self.target = target
        self.support = support

        self.target_max = np.max(self.target)
        self.support_max = support[np.argmax(self.target)]

        self.log_target = np.log10(self.target)
        self.log_support = np.log10(self.support)

        self.xmin = np.min(support)
        self.xmax = np.max(support)

        self.bins = bins

        self._segmented_functions = [lambda x: -1 for _ in range(len(bins) - 1)]

        self.diff = 0.02

    def target_log_approx(self, x):
        return np.power(10.0, np.interp(np.log10(x), self.log_support, self.log_target))

    def segment_factory(self, slope, logxmin, logxmax, val, low=True):
        xmin = np.power(10, logxmin)
        xmax = np.power(10, logxmax)

        return PowerLawSegment(xmin, xmax, slope, val, low=low)

    def init_slope(self, logxmin, logxmax):
        return np.log10(
            self.target_log_approx(np.power(10, logxmax))
            / self.target_log_approx(np.power(10, logxmin))
        ) / (logxmax - logxmin)

    def _fit_segment(self, xmin, xmax, low=True, val=None):
        self._trial_functions = []
        if low and val is None:
            val = self.__call__(xmin)
        elif not low and val is None:
            val = self.__call__(xmax)

        logxmin = np.log10(xmin)
        logxmax = np.log10(xmax)

        support = np.geomspace(xmin, xmax)

        # Propose first function
        slope = self.init_slope(logxmin, logxmax)
        function = self.segment_factory(slope, logxmin, logxmax, val, low=low)

        diff = function(support) - self.target_log_approx(support)

        if np.any(diff < 0.0) and low:
            step = self.diff
        elif low:
            step = -self.diff
        elif np.any(diff < 0.0) and not low:
            step = -self.diff
        elif not low:
            step = self.diff

        # Only allow 100 steps to guarantee an exit condition, although should raise an error if convergence is not reached
        for i in range(200):
            new_slope = slope + step
            new_function = self.segment_factory(
                new_slope, logxmin, logxmax, val, low=low
            )
            # self._trial_functions.append(new_function)
            negative = np.any(
                new_function(support) - self.target_log_approx(support) < 0.0
            )
            # Just spell all 8 cases out...
            if step > 0.0 and low and negative:
                slope = new_slope
                continue
            elif step > 0.0 and low:
                slope = new_slope
                break
            elif step < 0.0 and low and negative:
                break
            elif step < 0.0 and low:
                slope = new_slope
            elif step > 0.0 and negative and not low:
                break
            elif step > 0.0 and not low:
                slope = new_slope
            elif step < 0.0 and not low and negative:
                slope = new_slope
            elif step < 0.0 and not low:
                break
        else:
            print("did not converge")

        # Exit codition is met
        print(f"converged after {i} steps")
        function = self.segment_factory(slope, logxmin, logxmax, val, low=low)
        return function

    @property
    def slopes(self):
        return np.array([_.slope for _ in self._segmented_functions])

    @property
    def low_values(self):
        return np.array([_.val_low for _ in self._segmented_functions])

    @property
    def weights(self):
        return self.integrals / self.integrals.sum()

    @property
    def integrals(self):
        return np.array([_.integral for _ in self._segmented_functions])

    @property
    def N(self):
        return self.weights.size

    def __call__(self, x):
        # left or right boundary inclusive should depend on the creation scheme (going left or right),
        # should set private attribute accordingly that is used in digitize
        idx = np.digitize(x, self.bins) - 1
        func = self._segmented_functions[idx]
        val = func(x)
        if val == -1:
            func = self._segmented_functions[idx - 1]
            val = func(x)
        return val

    @abstractmethod
    def generate_segments(self):
        return


class TopDownSegmentation(SegmentedApprox):
    def __init__(self, target, support, dec_width):
        width = dec_width  # decadic width
        target_max_point = support[np.argmax(target).squeeze()]
        middle = np.log10(target_max_point)
        breaks = [middle - width / 2, middle + width / 2]
        logSuppMin = np.log10(np.min(support))
        logSuppMax = np.log10(np.max(support))
        if breaks[0] < logSuppMin:
            breaks[0] = logSuppMin
        if breaks[1] > logSuppMax:
            breaks[1] = logSuppMax

        if breaks[1] < logSuppMax:
            while True:
                proposal = breaks[-1] + width
                if proposal > logSuppMax:
                    proposal = logSuppMax
                breaks.append(proposal)

                if proposal == logSuppMax:
                    break

        if breaks[0] > logSuppMin:
            while True:
                proposal = breaks[0] - width
                if proposal < logSuppMin:
                    proposal = logSuppMin
                breaks.insert(0, proposal)
                if proposal == logSuppMin:
                    break

        breaks = np.power(10, breaks)

        super().__init__(target, support, breaks)

        self.bin_containing_peak = np.digitize(target_max_point, breaks) - 1

    def generate_segments(self):
        low_values = []
        for c, (l, h) in enumerate(
            zip(
                self.bins[self.bin_containing_peak : -1],
                self.bins[self.bin_containing_peak + 1 :],
            )
        ):
            idx = self.bin_containing_peak + c
            if c == 0:
                # Avoid accidental negative differences at the pivot if low==True
                # may happen if the max coincides with the pivot point
                val = self.target_max * 1.01
            func = self._fit_segment(l, h, low=True, val=val)

            val = func(h)
            low_values.append(func(l))

            self._segmented_functions[idx] = func

        if self.bin_containing_peak == 0:
            return
        for c, (l, h) in enumerate(
            zip(
                self.bins[0 : self.bin_containing_peak][::-1],
                self.bins[1 : self.bin_containing_peak + 1][::-1],
            ),
            1,
        ):
            idx = self.bin_containing_peak - c
            if c == 1:
                val = low_values[0]
            func = self._fit_segment(l, h, low=False, val=val)

            val = func(l)

            self._segmented_functions[idx] = func
