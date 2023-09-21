"""
Implements ROI with cuts on sky region for analysis
"""

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np

import logging

logger = logging.getLogger(__name__)


class ROI:
    # TODO
    # Maybe have capability at later point to use multiple ROIs
    # s.t. multiple point sources can be considered simultaneously.
    # also, dataclass?!
    STACK = []

    @u.quantity_input
    def __init__(
        self,
        center: SkyCoord,
        radius: u.deg = 5 * u.deg,
        MJD_min=0.0,
        MJD_max=np.inf,
    ):
        self._center = center
        self._radius = radius
        self._MJD_min = MJD_min
        self._MJD_max = MJD_max

        if ROI.STACK:
            # logger.warning(
            #    "Only one ROI allowed at a time. Deleting previous instance.\n"
            # )
            ROI.STACK = [self]
        else:
            ROI.STACK = [self]

        if self._center.dec.deg - self._radius.to_value(u.deg) < -10.0:
            logger.warning("ROI extends into Southern sky. Proceed with chaution.")

        if self._radius.to(u.deg) > 180.0 * u.deg:
            raise ValueError("Radii larger than 180 degrees are not sensible.")

    @property
    def MJD_min(self):
        return self._MJD_min

    @property
    def MJD_max(self):
        return self._MJD_max

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius
