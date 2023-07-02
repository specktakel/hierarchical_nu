import numpy as np
from astropy import units as u
from hierarchical_nu.utils.roi import ROI
from hierarchical_nu.events import Events
from hierarchical_nu.detector.r2021 import R2021DetectorModel
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import PointSource, Sources
from hierarchical_nu.simulation import Simulation
import pytest


def test_event_selection():
    roi = ROI(DEC_min=0.0 * u.rad)
    events = Events.from_ev_file("IC86_II")
    assert events.coords.z.min() >= 0.0


def test_range():
    with pytest.raises(ValueError):
        roi = ROI(DEC_max=3 * u.rad)


def test_precomputation():
    roi = ROI()
    src_index = Parameter(2.3, "src_index", par_range=(1.5, 3.6))
    L = Parameter(
        1e47 * u.erg / u.s,
        "luminosity",
        fixed=True,
        par_range=(0, 1e53) * (u.erg / u.s),
    )
    diff_index = Parameter(2.3, "diff_index", par_range=(1.5, 3.6))
    diffuse_norm = Parameter(
        1e-13 / (u.GeV * u.m**2 * u.s),
        "diffuse_norm",
        fixed=True,
        par_range=(0.0, 1e-8),
    )
    z = 0.4
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e4 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(Emin.value * (1 + z), "Emin_src", fixed=True)
    Emax_src = Parameter(Emax.value * (1 + z), "Emax_src", fixed=True)
    Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
    Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

    Emin_det = Parameter(4e4 * u.GeV, "Emin_det", fixed=True)

    # Single PS for testing and usual components
    point_source = PointSource.make_powerlaw_source(
        "test",
        np.deg2rad(5) * u.rad,
        np.pi * u.rad,
        L,
        src_index,
        z,
        Emin_src,
        Emax_src,
    )

    my_sources = Sources()

    my_sources.add(point_source)

    # auto diffuse component
    my_sources.add_diffuse_component(
        diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff
    )

    sim = Simulation(my_sources, NorthernTracksDetectorModel, 5 * u.year)

    sim.precomputation()

    default = sim._get_sim_inputs()

    roi = ROI(RA_min=1.0 * np.pi * u.rad)

    sim.precomputation()
    cut = sim._get_sim_inputs()

    assert pytest.approx(np.array(cut["integral_grid_t"][1]) * 2.0) == np.array(
        default["integral_grid_t"][1]
    )
    assert pytest.approx(np.array(cut["integral_grid_t"][0])) == np.array(
        default["integral_grid_t"][0]
    )
