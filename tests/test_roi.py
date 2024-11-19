import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from hierarchical_nu.utils.roi import CircularROI, RectangularROI, FullSkyROI, ROIList
from hierarchical_nu.events import Events
from hierarchical_nu.detector.icecube import IC86_II
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import PointSource, Sources
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.detector.input import mceq
import pytest

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def test_circular_event_selection():
    Parameter.clear_registry()
    ROIList.clear_registry()
    Emin_det = Parameter(1e1 * u.GeV, "Emin_det", fixed=True)
    roi = CircularROI(
        center=SkyCoord(ra=90 * u.deg, dec=10 * u.deg, frame="icrs"),
        radius=10.0 * u.deg,
    )
    logger.warning(roi)
    events = Events.from_ev_file(IC86_II)
    events.coords.representation_type = "cartesian"
    assert events.coords.z.min() >= 0.0


def test_rectangular_event_selection():
    ROIList.clear_registry()
    Parameter.clear_registry()
    Emin_det = Parameter(1e1 * u.GeV, "Emin_det", fixed=True)
    roi = RectangularROI(DEC_min=0.0 * u.rad)
    logger.warning(roi)
    events = Events.from_ev_file(IC86_II)
    events.coords.representation_type = "cartesian"
    assert events.coords.z.min() >= 0.0


def test_roi_south(caplog):
    ROIList.clear_registry()
    caplog.set_level(logging.WARNING)
    roi = CircularROI(
        center=SkyCoord(ra=90 * u.deg, dec=0 * u.deg, frame="icrs"),
        radius=12.0 * u.deg,
    )

    assert "ROI extends into Southern sky. Proceed with caution." in caplog.text


def test_humongous_roi():
    ROIList.clear_registry()
    with pytest.raises(ValueError):
        roi = CircularROI(
            center=SkyCoord(ra=90 * u.deg, dec=10 * u.deg, frame="icrs"),
            radius=181.0 * u.deg,
        )


def test_event_selection_wrap(caplog):
    ROIList.clear_registry()
    Parameter.clear_registry()
    Emin_det = Parameter(1e1 * u.GeV, "Emin_det", fixed=True)
    roi = RectangularROI(RA_min=np.deg2rad(350) * u.rad, RA_max=np.deg2rad(10) * u.rad)
    events = Events.from_ev_file(IC86_II)
    events.coords.representation_type = "spherical"
    ra = events.coords.ra.rad
    mask = np.nonzero((ra >= np.pi))
    ra[mask] -= 2 * np.pi

    assert pytest.approx(np.average(ra), abs=1e-3) == 0.0

    assert "RA_min is greater than RA_max" in caplog.text

    assert "RA_max is smaller than RA_min" in caplog.text


def test_rectangular_precomputation():
    ROIList.clear_registry()
    Parameter.clear_registry()
    roi = FullSkyROI()
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

    sim = Simulation(my_sources, IC86_II, 5 * u.year)

    sim.precomputation()

    default = sim._get_sim_inputs()
    ROIList.clear_registry()
    # test RA wrapping from 270 degrees to 90 degrees
    roi = RectangularROI(RA_max=np.deg2rad(90) * u.rad, RA_min=np.deg2rad(270) * u.rad)
    sim.precomputation()
    cut = sim._get_sim_inputs()

    assert pytest.approx(np.exp(cut["integral_grid"][0][1]) * 2.0, rel=5e-2) == np.exp(
        default["integral_grid"][0][1]
    )
    assert pytest.approx(np.exp(cut["integral_grid"][0][0])) == np.exp(
        default["integral_grid"][0][0]
    )


def test_multiple_rois():
    ROIList.clear_registry()
    Parameter.clear_registry()
    z = 0.3
    diff_index = Parameter(2.3, "diff_index", par_range=(1.5, 3.6))
    diffuse_norm = Parameter(
        1e-13 / (u.GeV * u.m**2 * u.s),
        "diffuse_norm",
        fixed=True,
        par_range=(0.0, 1e-8),
    )
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e4 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(Emin.value * (1 + z), "Emin_src", fixed=True)
    Emax_src = Parameter(Emax.value * (1 + z), "Emax_src", fixed=True)
    Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
    Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

    Emin_det = Parameter(2e2 * u.GeV, "Emin_det", fixed=True)

    my_sources = Sources()

    # auto diffuse component
    my_sources.add_diffuse_component(
        diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff
    )
    my_sources.add_atmospheric_component(cache_dir=mceq)
    sim = Simulation(my_sources, IC86_II, 5 * u.year)

    roi = CircularROI(
        SkyCoord(ra=np.pi / 2 * u.rad, dec=0 * u.rad, frame="icrs"), radius=5 * u.deg
    )
    sim.precomputation()
    _ = sim._get_expected_Nnu(sim._get_sim_inputs())
    Nex_1 = sim._expected_Nnu_per_comp

    ROIList.clear_registry()
    roi = CircularROI(SkyCoord(ra=np.pi * u.rad, dec=5 * u.deg), radius=5 * u.deg)
    sim.precomputation()
    _ = sim._get_expected_Nnu(sim._get_sim_inputs())
    Nex_2 = sim._expected_Nnu_per_comp

    ROIList.clear_registry()
    roi = CircularROI(
        SkyCoord(ra=np.pi / 2 * u.rad, dec=0 * u.rad, frame="icrs"), radius=5 * u.deg
    )
    roi = CircularROI(SkyCoord(ra=np.pi * u.rad, dec=5 * u.deg), radius=5 * u.deg)
    sim = Simulation(my_sources, IC86_II, 5 * u.year)
    sim.precomputation()
    _ = sim._get_expected_Nnu(sim._get_sim_inputs())
    Nex_12 = sim._expected_Nnu_per_comp

    assert np.sum(Nex_1) + np.sum(Nex_2) == pytest.approx(np.sum(Nex_12), abs=0.02)


def test_compare_precomputation():
    ROIList.clear_registry()
    Parameter.clear_registry()
    z = 0.3
    diff_index = Parameter(2.3, "diff_index", par_range=(1.5, 3.6))
    diffuse_norm = Parameter(
        1e-13 / (u.GeV * u.m**2 * u.s),
        "diffuse_norm",
        fixed=True,
        par_range=(0.0, 1e-8),
    )
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e4 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(Emin.value * (1 + z), "Emin_src", fixed=True)
    Emax_src = Parameter(Emax.value * (1 + z), "Emax_src", fixed=True)
    Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
    Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

    Emin_det = Parameter(2e2 * u.GeV, "Emin_det", fixed=True)

    my_sources = Sources()

    # auto diffuse component
    my_sources.add_diffuse_component(
        diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff
    )
    my_sources.add_atmospheric_component(cache_dir=mceq)
    sim = Simulation(my_sources, IC86_II, 5 * u.year)

    roi = RectangularROI(DEC_min=np.deg2rad(-5) * u.rad)
    sim.precomputation()
    _ = sim._get_expected_Nnu(sim._get_sim_inputs())
    Nex_rectangle = sim._expected_Nnu_per_comp

    ROIList.clear_registry()
    roi = CircularROI(
        SkyCoord(ra=90 * u.deg, dec=90 * u.deg), radius=np.deg2rad(95) * u.rad
    )
    sim.precomputation()
    _ = sim._get_expected_Nnu(sim._get_sim_inputs())
    Nex_circular = sim._expected_Nnu_per_comp
    assert Nex_rectangle[0] == pytest.approx(Nex_circular[0], rel=0.02)
    assert Nex_rectangle[1] == pytest.approx(Nex_circular[1], rel=0.0005)
