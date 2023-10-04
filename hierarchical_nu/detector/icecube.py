from ..backend import DistributionMode
from .northern_tracks import (
    NorthernTracksDetectorModel,
)
from .cascades import (
    CascadesDetectorModel,
)
from .r2021 import R2021DetectorModel

NT = "northern_tracks"
CAS = "cascades"
IC40 = "IC40"
IC59 = "IC59"
IC79 = "IC79"
IC86_I = "IC86_I"
IC86_II = "IC86_II"

# DETECTORS = [NT, CAS, IC40, IC59, IC79, IC86_I, IC86_II]
# Dictionary of currently supported detector configs
DETECTOR_DICT = {NT: NorthernTracksDetectorModel, CAS: CascadesDetectorModel}


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
