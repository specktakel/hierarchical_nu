
from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.source.source import Sources, PointSource, DetectorFrame
from hierarchical_nu.utils.lifetime import LifeTime
import matplotlib.pyplot as plt

from hierarchical_nu.utils.roi import FullSkyROI
from hierarchical_nu.detector.icecube import IC86
import numpy as np
import matplotlib.pyplot as plt
import ligo.skymap.plot

import pytest


def test_energy_resolution():
    src_index = Parameter(2.2, "src_index", fixed=False, par_range=(1, 4))
    L = Parameter(
        1e47 * (u.erg / u.s), "luminosity", fixed=True, par_range=(0, 1e60) * (u.erg / u.s)
    )
    Nex_src = Parameter(0.0, "Nex_src", fixed=True, par_range=(0, 100))
    z = 0.3365
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e4 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(2e4 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(Emin.value, "Emin_src", fixed=True)
    Emax_src = Parameter(Emax.value, "Emax_src", fixed=True)

    Emin_det = Parameter(1e1 * u.GeV, "Emin_det", fixed=True)

    # Single PS for testing and usual components
    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    width = np.deg2rad(6) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_powerlaw_source(
        "test",
        dec,
        ra,
        L,
        src_index,
        z,
        Emin_src,
        Emax_src,
        DetectorFrame,
    )

    my_sources = Sources()
    my_sources.add(point_source)

    FullSkyROI()

    lt = LifeTime()
    lifetime = lt.lifetime_from_season(IC86)


    event_types = list(lifetime.keys())

    sim = Simulation(my_sources, event_types, lifetime, N={IC86: [10000]})

    sim.precomputation()

    sim.generate_stan_code()
    sim.compile_stan_code()

    sim.run()

    irf = sim._exposure_integral[IC86].energy_resolution.irf

    dec_idx = np.digitize(dec.to_value(u.deg), irf.dec_bin_edges) - 1

    recoE = np.log10(sim._sim_output.stan_variable("Edet"))

    tE_idx = np.digitize(np.log10(Emax.value.to_value(u.GeV)), irf.log_tE_bin_edges) - 1

    hist = irf.recoE_hists[tE_idx, dec_idx]
    bins = irf.recoE_bin_edges[tE_idx, dec_idx]

    n, _ = np.histogram(recoE, bins, density=True)
    cdf_per_bin = hist * np.sum(np.diff(bins))
    mask = cdf_per_bin >= 0.1

    assert pytest.approx(hist[mask], abs=0.1) == n[mask]

