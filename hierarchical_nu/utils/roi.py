"""
Implements ROI with cuts on sky region for analysis
"""

# TODO: combine ROIList with ROI class

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np
from scipy.optimize import minimize_scalar
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)


class ROI(ABC):
    """
    Properties should be self explanatory
    """

    def __init__(self):
        ROIList.add(self)

    @property
    def MJD_min(self):
        return self._MJD_min

    @property
    def MJD_max(self):
        return self._MJD_max

    @property
    @abstractmethod
    def DEC_min(self):
        pass

    @property
    @abstractmethod
    def DEC_max(self):
        pass

    @property
    @abstractmethod
    def RA_min(self):
        pass

    @property
    @abstractmethod
    def RA_max(self):
        pass

    @property
    def apply_roi(self):
        return self._apply_roi


class ROIList:
    STACK = []

    @staticmethod
    def add(roi):
        """
        Add ROI to stack
        Currently only exclusively CircularROIs or non-CircularROIs can be stacked.
        :param roi: ROI instance to be added
        """

        if ROIList.STACK:
            if isinstance(roi, CircularROI) and not isinstance(
                ROIList.STACK[0], CircularROI
            ):
                raise ValueError("Circular and non-Circular ROIs must not be mixed.")
            if not isinstance(roi, CircularROI):
                raise ValueError("Non-CircularROIs cannot be stacked.")
        ROIList.STACK.append(roi)

    @staticmethod
    def pop(i):
        """
        Remove ROI from stack
        :param i: i-th ROI to be deleted
        """

        ROIList.STACK.pop(i)

    @staticmethod
    def clear_registry():
        """
        Cleares entire stack of ROIs
        """

        ROIList.STACK = []

    # The properties are defined such that the entire stack of ROIs is scanned over
    # and the most extreme values found will be returned

    # need to think about the wrapping at 2pi/0
    # TODO for future-Julian
    @staticmethod
    def RA_max():
        ra_max = 0.0 * u.rad
        for roi in ROIList.STACK:
            temp = roi.RA_max
            if temp > ra_max:
                ra_max = temp

        return ra_max

    @staticmethod
    def RA_min():
        ra_min = 2 * np.pi * u.rad
        for roi in ROIList.STACK:
            temp = roi.RA_min
            if temp < ra_min:
                ra_min = temp

        return ra_min

    @staticmethod
    def DEC_min():
        dec_min = np.pi / 2 * u.rad
        for roi in ROIList.STACK:
            temp = roi.DEC_min
            if temp < dec_min:
                dec_min = temp

        return dec_min

    @staticmethod
    def DEC_max():
        dec_max = -np.pi / 2 * u.rad
        for roi in ROIList.STACK:
            temp = roi.DEC_max
            if temp > dec_max:
                dec_max = temp

        return dec_max

    @staticmethod
    def MJD_min():
        return min([_.MJD_min for _ in ROIList.STACK])
    
    @staticmethod
    def MJD_max():
        return max([_.MJD_max for _ in ROIList.STACK])

    def __repr__(self):
        return "\n".join([roi.__repr__() for roi in ROIList.STACK])


class CircularROI(ROI):
    """
    Implements circular ROI
    """

    @u.quantity_input
    def __init__(
        self,
        center: SkyCoord,
        radius: u.deg = 5 * u.deg,
        MJD_min=0.0,
        MJD_max=np.inf,
        apply_roi: bool = False,
    ):
        """
        :param center: SkyCoord instance of ROI's center
        :param radius: Radius of ROI
        :param MJD_min: Minimum MJD, only used for data selection
        :param MJD_max: Maximum MJD, only used for data selection
        :param apply_roi: True if ROI should by applied at data selection
        """
        self._center = center
        self._radius = radius
        self._MJD_min = MJD_min
        self._MJD_max = MJD_max
        self._apply_roi = apply_roi

        # ROI.STACK.append(self)
        super().__init__()
        self._center.representation_type = "spherical"
        if self._center.dec.deg - self._radius.to_value(u.deg) < -10.0:
            logger.warning("ROI extends into Southern sky. Proceed with caution.")

        if self._radius.to(u.deg) > 180.0 * u.deg:
            raise ValueError("Radii larger than 180 degrees are not sensible.")

    def __repr__(self):
        self._center.representation_type = "spherical"
        return f"CircularROI, center RA={self._center.ra.deg:.1f}°, DEC={self._center.dec.deg:.1f}°, radius={self._radius.to_value(u.deg):.1f}°, apply={self.apply_roi}"

    def _get_roi_width(self):
        """
        The widest point in RA of a circle on a sphere
        does generally not coincide with the line of constant latitude/declination
        crossing the circle's center point.
        The angular separation between points on the ROI boundary (i.e. a small circle)
        and the central line of constant longitude/RA crossing the ROI's center point
        is maximised by varying the declination.
        Twice the angular separation at its maximum evaluates to the ROI width in RA.
        """
        self._center.representation_type = "spherical"
        epsilon = np.deg2rad(0.01)
        res = minimize_scalar(
            ROI_width,
            args=(self.radius.to_value(u.rad), self.center.dec.rad),
            bounds=(
                self.center.dec.rad - self.radius.to_value(u.rad) + epsilon,
                self.center.dec.rad + self.radius.to_value(u.rad) - epsilon,
            ),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": int(1e5)},
        )
        if not res.success:
            raise ValueError("Couldn't determine RA width of ROI")
        return np.abs(res.fun * u.rad)

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    @property
    def DEC_min(self):
        self._center.representation_type = "spherical"
        dec_min = self.center.dec.rad * u.rad - self.radius.to(u.rad)
        if dec_min < -np.pi / 2 * u.rad:
            return -np.pi / 2 * u.rad
        else:
            return dec_min

    @property
    def DEC_max(self):
        self._center.representation_type = "spherical"
        dec_max = self.center.dec.rad * u.rad + self.radius.to(u.rad)
        if dec_max > np.pi / 2 * u.rad:
            return np.pi / 2 * u.rad
        else:
            return dec_max

    @property
    def _RA_min(self):
        if (
            self.center.dec.deg * u.deg - self.radius.to(u.deg) < -90.0 * u.deg
            or self.center.dec.deg * u.deg + self.radius.to(u.deg) > 90.0 * u.deg
        ):
            return 0.0 * u.rad
        else:
            RA_width = self._get_roi_width()
            min = self.center.ra.rad * u.rad - RA_width.to(u.rad)
            return min

    @property
    def RA_min(self):
        min = self._RA_min
        # Check for wrapping at 2pi/0
        if min < 0 * u.rad:
            return min + 2.0 * np.pi * u.rad
        else:
            return min

    @property
    def _RA_max(self):
        self._center.representation_type = "spherical"
        if (
            self.center.dec.deg * u.deg - self.radius.to(u.deg) < -90.0 * u.deg
            or self.center.dec.deg * u.deg + self.radius.to(u.deg) > 90.0 * u.deg
        ):
            return 2.0 * np.pi * u.rad
        else:
            RA_width = self._get_roi_width()
            max = self.center.ra.rad * u.rad + RA_width.to(u.rad)
            return max

    @property
    def RA_max(self):
        # Check for wrapping at 2pi/0
        max = self._RA_max
        if max > 2.0 * np.pi * u.rad:
            return max - 2.0 * np.pi * u.rad
        else:
            return max


class RectangularROI(ROI):
    """
    Implements rectangular (in RA, DEC) ROI
    """

    @u.quantity_input
    def __init__(
        self,
        RA_min=0.0 * np.pi * u.rad,
        RA_max=2.0 * np.pi * u.rad,
        DEC_min=-np.pi / 2 * u.rad,
        DEC_max=np.pi / 2 * u.rad,
        MJD_min=0.0,
        MJD_max=np.inf,
        apply_roi: bool = False,
    ):
        """
        :param RA_min: Minimum RA
        :param RA_max: Maximum RA
        :param DEC_min: Minimum DEC
        :param DEC_max: Maximum DEC
        :param MJD_min: Minimum MJD, only used for data selection
        :param MJD_max: Maximum MJD, only used for data selection
        :param apply_roi: True if ROI should by applied at data selection
        """

        self._RA_min = RA_min
        self._RA_max = RA_max
        self._DEC_min = DEC_min
        self._DEC_max = DEC_max
        self._MJD_min = MJD_min
        self._MJD_max = MJD_max
        self._apply_roi = apply_roi

        self.check_boundaries()

        # ROI.STACK.append(self)
        super().__init__()

        if self._DEC_min.to(u.deg) < -10.0 * u.deg:
            logger.warning("ROI extends into Southern sky. Proceed with caution.")

    @property
    def RA_min(self):
        return self._RA_min

    @property
    def RA_max(self):
        return self._RA_max

    @property
    def DEC_min(self):
        return self._DEC_min

    @property
    def DEC_max(self):
        return self._DEC_max

    @property
    def MJD_min(self):
        return self._MJD_min

    @property
    def MJD_max(self):
        return self._MJD_max

    @RA_min.setter
    @u.quantity_input
    def RA_min(self, val: u.rad):
        if val < 0.0 * u.rad:
            raise ValueError("RA must be between 0 and 2pi.")
        if val > self._RA_max:
            logger.warning(
                f"RA_min is greater than RA_max={self._RA_max:.2f}. Event selection will wrap at 0/2pi."
            )
        self._RA_min = val

    @RA_max.setter
    @u.quantity_input
    def RA_max(self, val: u.rad):
        if val > 2.0 * np.pi * u.rad:
            raise ValueError("RA must be between 0 and 2 pi.")
        if val < self._RA_min:
            logger.warning(
                f"RA_max is smaller than RA_min={self._RA_min:.2f}. Event selection will wrap at 0/2pi."
            )
        self._RA_max = val

    @DEC_min.setter
    @u.quantity_input
    def DEC_min(self, val: u.rad):
        if val < -np.pi / 2 * u.rad or val > self._DEC_max:
            raise ValueError("DEC must be between -pi/2 and pi/2 and min < max.")
        self._DEC_min = val

    @DEC_max.setter
    @u.quantity_input
    def DEC_max(self, val: u.rad):
        if val > np.pi / 2 * u.rad or val < self._DEC_min:
            raise ValueError("DEC must be between -pi/2 and pi/2 and min < max.")
        self._DEC_max = val

    def check_boundaries(self):
        """
        Check if all values are allowed,
        actual checks done in setter methods
        """

        self.RA_min = self._RA_min
        self.RA_max = self._RA_max
        self.DEC_min = self._DEC_min
        self.DEC_max = self._DEC_max

    def __repr__(self):
        return f"RectangularROI, DEC=[{self.DEC_min.to_value(u.deg):.1f}°, {self.DEC_max.to_value(u.deg):.1f}°], RA=[{self.RA_min.to_value(u.deg):.1f}°, {self.RA_max.to_value(u.deg):.1f}°], apply={self.apply_roi}"


class FullSkyROI(RectangularROI):
    """
    Wrapper class for to easily define no ROI selection
    for extra user-friendliness.
    """

    def __init__(
        self,
        MJD_min=0.0,
        MJD_max=np.inf,
        apply_roi: bool = False,
    ):
        """
        :param MJD_min: Minimum MJD, only used for data selection
        :param MJD_max: Maximum MJD, only used for data selection
        :param apply_roi: True if ROI should by applied at data selection
        """

        super().__init__(MJD_min=MJD_min, MJD_max=MJD_max, apply_roi=apply_roi)

    def __repr__(self):
        return f"FullSkyROI, apply={self.apply_roi}"


class NorthernSkyROI(RectangularROI):
    """
    Wrapper class for to define an ROI in the Northern Sky,
    bounded at DEC_min=-5 deg
    """

    def __init__(
        self,
        MJD_min=0.0,
        MJD_max=np.inf,
        apply_roi: bool = False,
    ):
        """
        :param MJD_min: Minimum MJD, only used for data selection
        :param MJD_max: Maximum MJD, only used for data selection
        :param apply_roi: True if ROI should by applied at data selection
        """

        super().__init__(
            DEC_min=np.deg2rad(-5) * u.rad,
            MJD_min=MJD_min,
            MJD_max=MJD_max,
            apply_roi=apply_roi,
        )

    def __repr__(self):
        return f"NorthernSkyROI, apply={self.apply_roi}"


def ROI_width(d1, radius, d2):
    """
    Returns ROI width as function of declination.
    Only sensibly defined within the ROI of radius `radius`
    :param d1: declination at which width is to be determined
    :param radius: radius of ROI
    :param d2: declination of ROI center
    """

    return -np.arccos(
        (np.cos(radius) - np.sin(d1) * np.sin(d2)) / (np.cos(d1) * np.cos(d2))
    )
