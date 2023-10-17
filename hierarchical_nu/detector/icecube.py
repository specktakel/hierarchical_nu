from ..backend import DistributionMode
from .northern_tracks import (
    NorthernTracksDetectorModel,
)
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


class Refrigerator:
    STAN_NT = 0
    STAN_CAS = 1
    STAN_IC40 = 2
    STAN_IC59 = 3
    STAN_IC79 = 4
    STAN_IC86_I = 5
    STAN_IC86_II = 6

    PYTHON_NT = "northern_tracks"
    PYTHON_CAS = "cascades"
    PYTHON_IC40 = "IC40"
    PYTHON_IC59 = "IC59"
    PYTHON_IC79 = "IC79"
    PYTHON_IC86_I = "IC86_I"
    PYTHON_IC86_II = "IC86_II"

    stan_detectors = [
        STAN_NT,
        STAN_CAS,
        STAN_IC40,
        STAN_IC59,
        STAN_IC79,
        STAN_IC86_I,
        STAN_IC86_II,
    ]

    python_detectors = [
        PYTHON_NT,
        PYTHON_CAS,
        PYTHON_IC40,
        PYTHON_IC59,
        PYTHON_IC79,
        PYTHON_IC86_I,
        PYTHON_IC86_II,
    ]

    @classmethod
    def stan2python(cls, stan):
        for c, s_dm in enumerate(cls.stan_detectors):
            if stan == s_dm:
                break
        else:
            raise ValueError("No such detector available.")

        return cls.python_detectors[c]

    @classmethod
    def python2stan(cls, python):
        for c, p_dm in enumerate(cls.python_detectors):
            if python == p_dm:
                break
        else:
            raise ValueError("No such detector available.")

        return cls.stan_detectors[c]


# Dictionary of currently supported detector configs
DETECTOR_DICT = {
    Refrigerator.PYTHON_NT: NorthernTracksDetectorModel,
    Refrigerator.PYTHON_CAS: CascadesDetectorModel,
    Refrigerator.PYTHON_IC40: IC40DetectorModel,
    Refrigerator.PYTHON_IC59: IC59DetectorModel,
    Refrigerator.PYTHON_IC79: IC79DetectorModel,
    Refrigerator.PYTHON_IC86_I: IC86_IDetectorModel,
    Refrigerator.PYTHON_IC86_II: IC86_IIDetectorModel,
}


def IceCube(detector, mode: DistributionMode = DistributionMode.PDF):
    return DETECTOR_DICT[detector](mode)
