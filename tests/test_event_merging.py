import astropy.units as u
from hierarchical_nu.events import Events
from hierarchical_nu.utils.roi import NorthernSkyROI, ROIList
import numpy as np


def test_merging():
    MJD = np.array([[56940, 56980], [56990, 57100]])

    NorthernSkyROI(MJD_min=MJD[0, 0], MJD_max=MJD[0, 1])

    ev1 = Events.from_event_files()
    ev1.apply_ROIS()

    ROIList.clear_registry()

    NorthernSkyROI(MJD_min=MJD[1, 0], MJD_max=MJD[1, 1])
    ev2 = Events.from_event_files()
    ev2.apply_ROIS()

    merged = ev1.merge(ev2)

    assert merged.N == ev1.N + ev2.N

    assert np.all(np.isclose(merged.energies.to_value(u.GeV), np.array([
        ev1.energies.to_value(u.GeV).tolist() + ev2.energies.to_value(u.GeV).tolist()
    ])))

    assert np.all(
        np.isclose(np.vstack((ev1.unit_vectors, ev2.unit_vectors)), merged.unit_vectors)
    )
