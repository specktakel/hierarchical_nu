"""
Implements ROI with cuts on sky region for analysis
"""

import astropy.units as u
import numpy as np

import logging

logger = logging.getLogger(__file__)


class ROI:
    # TODO
    # Maybe have capability at later point to use multiple ROIs
    # s.t. multiple point sources can be considered simultaneously.
    # also, dataclass?!
    STACK = []

    @u.quantity_input
    def __init__(
        self,
        RA_min=0.0 * np.pi * u.rad,
        RA_max=2.0 * np.pi * u.rad,
        DEC_min=-np.pi / 2 * u.rad,
        DEC_max=np.pi / 2 * u.rad,
    ):
        self._RA_min = RA_min
        self._RA_max = RA_max
        self._DEC_min = DEC_min
        self._DEC_max = DEC_max

        self.check_boundaries()

        if ROI.STACK:
            logger.warning(
                "Only one ROI allowed at a time. Deleting previous instance.\n"
            )
            ROI.STACK = [self]
        else:
            ROI.STACK.append(self)

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

    @RA_min.setter
    @u.quantity_input
    def RA_min(self, val: u.rad):
        if val < 0.0 * u.rad or val > self._RA_max:
            raise ValueError("RA must be between 0 and 2pi and min < max.")
        self._RA_min = val

    @RA_max.setter
    @u.quantity_input
    def RA_max(self, val: u.rad):
        if val > 2.0 * np.pi * u.rad or val < self._RA_min:
            raise ValueError("RA must be between 0 and 2 pi and min < max.")
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
        self.RA_min = self._RA_min
        self.RA_max = self._RA_max
        self.DEC_min = self._DEC_min
        self.DEC_max = self._DEC_max
