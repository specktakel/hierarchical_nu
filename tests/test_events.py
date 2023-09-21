from icecube_tools.utils.data import RealEvents
from hierarchical_nu.events import Events
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.utils.roi import ROI

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np


def test_read():
    periods = ["IC86_II"]
    Parameter.clear_registry()

    roi = ROI(SkyCoord(ra=0 * u.deg, dec=0 * u.deg, frame="icrs"), radius=180 * u.deg)
    for p in periods:
        it_ev = RealEvents.from_event_files(p, use_all=True)
        it_ev.restrict(ereco_low=5e4)
        hnu_ev = Events.from_ev_file(p, Emin_det=5e4 * u.GeV, use_all=True)

        assert hnu_ev.N == it_ev.N_restricted[p]

        # assert energy and some angles to check if rad/deg is correct
        assert np.isclose(it_ev.reco_energy[p][0], hnu_ev.energies[0].to(u.GeV).value)

        assert np.isclose(it_ev.ang_err[p][0], hnu_ev.ang_errs[0].to(u.deg).value)

        assert hnu_ev.energies.to(u.GeV).min() >= 5e4 * u.GeV
