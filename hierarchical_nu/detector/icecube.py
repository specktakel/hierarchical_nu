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
    IC86DetectorModel,
    #IC86_IDetectorModel,
    #IC86_IIDetectorModel,
)

from icecube_data_reader.event_types import IC40, IC59, IC79, IC86
from icecube_data_reader import event_types
EventType = event_types.EventType
Refrigerator = event_types.Refrigerator


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


IC40.model = IC40DetectorModel
IC59.model = IC59DetectorModel
IC79.model = IC79DetectorModel
IC86.model = IC86DetectorModel

"""
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
"""

"""
class Refrigerator:
    detectors = [NT, CAS, IC40, IC59, IC79, IC86_I, IC86_II]

    @classmethod
    def python2dm(cls, python):
        for dm in cls.detectors:
            if dm.P == python:
                return dm
        else:
            raise ValueError(f"No detector {python} available.")

    @classmethod
    def stan2dm(cls, stan):
        for dm in cls.detectors:
            if dm.S == stan:
                return dm
        else:
            raise ValueError(f"No detector {stan} available.")

    @classmethod
    def stan2python(cls, stan):
        for dm in cls.detectors:
            if stan == dm.S:
                return dm.P
        else:
            raise ValueError(f"No detector {stan} available.")

    @classmethod
    def python2stan(cls, python):
        for dm in cls.detectors:
            if python == dm.P:
                return dm.S
        else:
            raise ValueError(f"No detector {python} available.")
"""
