from ..backend import DistributionMode
from .northern_tracks import (
    NorthernTracksDetectorModel,
)
from dataclasses import dataclass
from .cascades import (
    CascadesDetectorModel,
)
from .r2021 import (
    IC40DetectorModel,
    IC59DetectorModel,
    IC79DetectorModel,
    IC86_IDetectorModel,
    IC86_IIDetectorModel,
)


class EventType:
    """Event type base class"""

    # Only works with @dataclass(eq=False) decorator called in notebooks directly, but not in e.g. LifeTime.mjd_from_dm
    def __eq__(self, other):
        return self.S == other.S


@dataclass(eq=False)
class NT(EventType):
    P = "northern_tracks"
    F = "NorthernTracks"
    S = 0
    model = NorthernTracksDetectorModel


@dataclass(eq=False)
class CAS(EventType):
    P = "cascades"
    F = "Cascades"
    S = 1
    model = CascadesDetectorModel


@dataclass(eq=False)
class IC40(EventType):
    P = "IC40"
    F = P
    S = 2
    model = IC40DetectorModel


@dataclass(eq=False)
class IC59(EventType):
    P = "IC59"
    F = P
    S = 3
    model = IC59DetectorModel


@dataclass(eq=False)
class IC79(EventType):
    P = "IC79"
    F = P
    S = 4
    model = IC79DetectorModel


@dataclass(eq=False)
class IC86_I(EventType):
    P = "IC86_I"
    F = P
    S = 5
    model = IC86_IDetectorModel


@dataclass(eq=False)
class IC86_II(EventType):
    P = "IC86_II"
    F = P
    S = 6
    model = IC86_IIDetectorModel


class Refrigerator:
    """Collect all event types"""
    
    detectors = [NT, CAS, IC40, IC59, IC79, IC86_I, IC86_II]

    @classmethod
    def python2dm(cls, python):
        """Returns EventType corresponding to python event-type string"""

        for dm in cls.detectors:
            if dm.P == python:
                return dm
        else:
            raise ValueError(f"No detector {python} available.")

    @classmethod
    def stan2dm(cls, stan):
        """Returns EventType corresponding to stan event-type"""

        for dm in cls.detectors:
            if dm.S == stan:
                return dm
        else:
            raise ValueError(f"No detector {stan} available.")

    @classmethod
    def stan2python(cls, stan):
        """Returns python event-type string corresponding to  stan event-type"""

        for dm in cls.detectors:
            if stan == dm.S:
                return dm.P
        else:
            raise ValueError(f"No detector {stan} available.")

    @classmethod
    def python2stan(cls, python):
        """Returns stan event-type corresponding to python event-type string"""

        for dm in cls.detectors:
            if python == dm.P:
                return dm.S
        else:
            raise ValueError(f"No detector {python} available.")
