"""
Wrapper for `Uptime` class of `icecube_tools`
"""

from typing import Union
import logging

from hierarchical_nu.detector.icecube import (
    IC40,
    IC59,
    IC79,
    IC86_I,
    IC86_II,
    EventType,
)
from icecube_tools.utils.data import Uptime, available_data_periods

from astropy import units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class LifeTime:
    def __init__(self):
        self._uptime = Uptime(*available_data_periods)

    def lifetime_from_mjd(
        self, MJD_min: Union[float, int], MJD_max: Union[float, int]
    ) -> dict[EventType, u.quantity.Quantity[u.year]]:
        lifetime = self._uptime.find_obs_time(start=MJD_min, end=MJD_max)
        output = {}
        for s in [IC40, IC59, IC79, IC86_I, IC86_II]:
            try:
                lt = lifetime[s.P]
                if lt > 0.0:
                    output[s] = lt * u.year
            except KeyError:
                pass

        return output

    def lifetime_from_dm(
        self, *event_type: EventType
    ) -> dict[EventType, u.quantity.Quantity[u.year]]:
        lifetime = self._uptime.cumulative_time_obs()
        output = {}
        for et in event_type:
            output[et] = lifetime[et.P] * u.year
        return output

    def mjd_from_dm(self, event_type: EventType) -> tuple[float]:
        mjd_min = self._uptime._data[event_type.P].min()
        if event_type == IC86_II:
            mjd_max = self._uptime._data["IC86_VII"].max()
        else:
            mjd_max = self._uptime._data[event_type.P].max()
        return mjd_min, mjd_max
