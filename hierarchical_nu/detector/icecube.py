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
    IC40GridDetectorModel,
    IC59GridDetectorModel,
    IC79GridDetectorModel,
    IC86_IGridDetectorModel,
    IC86_IIGridDetectorModel,
)


class EventType:
    pass


@dataclass
class NT(EventType):
    P = "northern_tracks"
    F = "NorthernTracks"
    S = 0
    model = NorthernTracksDetectorModel


@dataclass
class CAS(EventType):
    P = "cascades"
    F = "Cascades"
    S = 1
    model = CascadesDetectorModel


@dataclass
class IC40(EventType):
    P = "IC40"
    F = P
    S = 2
    model = IC40DetectorModel


@dataclass
class IC59(EventType):
    P = "IC59"
    F = P
    S = 3
    model = IC59DetectorModel


@dataclass
class IC79(EventType):
    P = "IC79"
    F = P
    S = 4
    model = IC79DetectorModel


@dataclass
class IC86_I(EventType):
    P = "IC86_I"
    F = P
    S = 5
    model = IC86_IDetectorModel


@dataclass
class IC86_II(EventType):
    P = "IC86_II"
    F = P
    S = 6
    model = IC86_IIDetectorModel


@dataclass
class IC40Grid(EventType):
    P = "IC40"
    F = P
    S = 2
    model = IC40GridDetectorModel


@dataclass
class IC59Grid(EventType):
    P = "IC59"
    F = P
    S = 3
    model = IC59GridDetectorModel


@dataclass
class IC79Grid(EventType):
    P = "IC79"
    F = P
    S = 4
    model = IC79GridDetectorModel


@dataclass
class IC86_IGrid(EventType):
    P = "IC86_I"
    F = P
    S = 5
    model = IC86_IGridDetectorModel


@dataclass
class IC86_IIGrid(EventType):
    P = "IC86_II"
    F = P
    S = 6
    model = IC86_IIGridDetectorModel


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
