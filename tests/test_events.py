from icecube_tools.utils.data import RealEvents
from hierarchical_nu.events import Events

from astropy import units as u
import numpy as np


def test_read():
    periods = ["IC86_II"]

    for p in periods:
        it_ev = RealEvents.from_event_files(p)
        it_ev.restrict(ereco_low=5e4)
        hnu_ev = Events.from_ev_file(p, ereco_low=5e4)

        # assert energy and some angles to check if rad/deg is correct
        assert np.isclose(it_ev.reco_energy[p][0], hnu_ev.energies[0].to(u.GeV).value)

        assert np.isclose(it_ev.ang_err[p][0], hnu_ev.ang_errs[0].to(u.deg).value)

        assert hnu_ev.energies.to(u.GeV).min() > 5e4 * u.GeV