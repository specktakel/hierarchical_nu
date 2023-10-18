from ..backend import DistributionMode
from .northern_tracks import (
    NorthernTracksDetectorModel,
)
from .cascades import (
    CascadesDetectorModel,
)
from .r2021 import R2021DetectorModel


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


"""
class ChilledGoods:
    def __init__(self, *p_dm):
        self._p_dm = []
        self._s_dm = []
        p_dm = list(p_dm)
        for dm in p_dm:
"""


# Dictionary of currently supported detector configs
DETECTOR_DICT = {
    Refrigerator.PYTHON_NT: NorthernTracksDetectorModel,
    Refrigerator.PYTHON_CAS: CascadesDetectorModel,
    Refrigerator.PYTHON_IC86_II: R2021DetectorModel,
}


def IceCube(detector, mode: DistributionMode = DistributionMode.PDF):
    return DETECTOR_DICT[detector](mode)


'''
class IceCubeDetectorModel(DetectorModel):
    """
    Unified interface to detector models for both track and
    cascade events for joint analyses.
    """

    event_types = ["tracks", "cascades"]

    PDF_FILENAME = "icecube_pdf.stan"
    RNG_FILENAME = "icecube_rng.stan"

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        event_type="tracks",
    ):
        super().__init__(mode=mode, event_type=event_type)

        if self._event_type == "cascades":
            ang_res = CascadesAngularResolution(mode)
            self._angular_resolution = ang_res

            energy_res = CascadesEnergyResolution(mode)
            self._energy_resolution = energy_res

            # if mode == DistributionMode.PDF:
            self._eff_area = CascadesEffectiveArea()

        elif self._event_type == "tracks":
            ang_res = NorthernTracksAngularResolution(mode)
            self._angular_resolution = ang_res

            energy_res = NorthernTracksEnergyResolution(mode)
            self._energy_resolution = energy_res

            # if mode == DistributionMode.PDF:
            self._eff_area = NorthernTracksEffectiveArea()

        else:
            raise ValueError("event_type %s not recognised" % event_type)

    def _get_effective_area(self):
        return self._eff_area

    def _get_energy_resolution(self):
        return self._energy_resolution

    def _get_angular_resolution(self):
        return self._angular_resolution
'''
