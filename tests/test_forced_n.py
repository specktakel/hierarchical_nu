import numpy as np
from astropy import units as u
import pytest

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource

from hierarchical_nu.detector.icecube import IC86_I, IC86_II
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.utils.roi import RectangularROI, ROIList

from hierarchical_nu.detector.input import mceq


def test_N():
    Parameter.clear_registry()
    ROIList.clear_registry()

    roi = RectangularROI(DEC_min=-5 * u.deg)

    src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))

    diff_index = Parameter(2.0, "diff_index", fixed=False, par_range=(1, 4))

    L = Parameter(
        2e47 * (u.erg / u.s),
        "luminosity",
        fixed=True,
        par_range=(0, 1e60) * (u.erg / u.s),
    )

    diffuse_norm = Parameter(
        1e-13 / u.GeV / u.m**2 / u.s,
        "diffuse_norm",
        fixed=True,
        par_range=(0, np.inf),
    )
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(5e4 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_det = Parameter(1e5 * u.GeV, "Emin_det", fixed=True)

    z = 0.4
    Emin_src = Parameter(Emin.value * (z + 1.0), "Emin_src", fixed=True)
    Emax_src = Parameter(Emax.value * (z + 1.0), "Emax_src", fixed=True)

    Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
    Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

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

    my_sources.add_diffuse_component(
        diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff
    )
    my_sources.add_atmospheric_component(cache_dir=mceq)

    sim = Simulation(
        my_sources,
        [IC86_I, IC86_II],
        {IC86_I: 5 * u.year, IC86_II: 5 * u.year},
        N={IC86_II: [2, 3, 2], IC86_I: [1, 2, 3]},
    )

    sim.precomputation()

    sim.generate_stan_code()

    sim.compile_stan_code()

    sim.run()

    assert np.all(
        np.isclose(
            sim._sim_output.stan_variable("Lambda"),
            np.array(
                [[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]]
            ),
        )
    )

    assert np.all(
        np.isclose(
            sim._sim_output.stan_variable("event_type"),
            np.array([[IC86_I.S] * 6 + [IC86_II.S] * 7]),
        )
    )


def test_multi_ps_n():
    Parameter.clear_registry()
    ROIList.clear_registry()
    roi = RectangularROI(DEC_min=-5 * u.deg)

    src_names = ["test_%i" % i for i in range(3)]
    src_index_params = []
    src_lumi_params = []
    for sn in src_names:
        src_index_params.append(
            Parameter(2.0, sn + "_src_index", fixed=False, par_range=(1, 4))
        )
        src_lumi_params.append(
            Parameter(
                2e47 * (u.erg / u.s),
                sn + "_luminosity",
                fixed=True,
                par_range=(0, 1e60) * (u.erg / u.s),
            )
        )

    diff_index = Parameter(2.0, "diff_index", fixed=False, par_range=(1, 4))

    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(5e4 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_det = Parameter(1e5 * u.GeV, "Emin_det", fixed=True)

    z = 0.4
    Emin_src = Parameter(Emin.value * (z + 1.0), "Emin_src", fixed=True)
    Emax_src = Parameter(Emax.value * (z + 1.0), "Emax_src", fixed=True)

    Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
    Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

    point_source_0 = PointSource.make_powerlaw_source(
        src_names[0],
        np.deg2rad(5) * u.rad,
        np.pi * u.rad,
        src_lumi_params[0],
        src_index_params[0],
        z,
        Emin_src,
        Emax_src,
    )

    point_source_1 = PointSource.make_powerlaw_source(
        src_names[1],
        np.deg2rad(25) * u.rad,
        np.pi * u.rad,
        src_lumi_params[1],
        src_index_params[1],
        z,
        Emin_src,
        Emax_src,
    )

    point_source_2 = PointSource.make_powerlaw_source(
        src_names[2],
        np.deg2rad(45) * u.rad,
        np.pi * u.rad,
        src_lumi_params[2],
        src_index_params[2],
        z,
        Emin_src,
        Emax_src,
    )

    my_sources = Sources()
    my_sources.add(point_source_0)
    my_sources.add(point_source_1)

    sim = Simulation(
        my_sources,
        [IC86_I, IC86_II],
        {IC86_I: 5 * u.year, IC86_II: 5 * u.year},
        N={IC86_I: [1, 2], IC86_II: [2, 1]},
    )
    sim.precomputation()
    sim.generate_stan_code()
    sim.compile_stan_code()

    sim.run()

    assert np.all(
        np.isclose(
            sim._sim_output.stan_variable("Lambda"),
            np.array([[1.0, 2.0, 2.0, 1.0, 1.0, 2.0]]),
        )
    )

    assert np.all(
        np.isclose(
            sim._sim_output.stan_variable("event_type"),
            np.array([[IC86_I.S] * 3 + [IC86_II.S] * 3]),
        )
    )

    my_sources.add(point_source_2)

    sim = Simulation(
        my_sources,
        [IC86_I, IC86_II],
        {IC86_I: 5 * u.year, IC86_II: 5 * u.year},
        N={IC86_I: [1, 2, 3], IC86_II: [4, 5, 6]},
    )
    sim.precomputation()
    sim.setup_stan_sim()
    sim.run()

    assert np.all(
        np.isclose(
            sim._sim_output.stan_variable("Lambda"),
            np.array(
                [[1.0, 2.0, 2.0, 3.0, 3.0, 3.0] + [1.0] * 4 + [2.0] * 5 + [3.0] * 6]
            ),
        )
    )

    assert np.all(
        np.isclose(
            sim._sim_output.stan_variable("event_type"),
            np.array([[IC86_I.S] * 6 + [IC86_II.S] * 15]),
        )
    )


def test_asimov():
    Parameter.clear_registry()
    ROIList.clear_registry()
    roi = RectangularROI(DEC_min=-5 * u.deg)

    src_names = ["test_%i" % i for i in range(3)]
    src_index_params = []
    src_lumi_params = []
    for sn in src_names:
        src_index_params.append(
            Parameter(2.0, sn + "_src_index", fixed=False, par_range=(1, 4))
        )
        src_lumi_params.append(
            Parameter(
                2e47 * (u.erg / u.s),
                sn + "_luminosity",
                fixed=True,
                par_range=(0, 1e60) * (u.erg / u.s),
            )
        )

    diff_index = Parameter(2.0, "diff_index", fixed=False, par_range=(1, 4))

    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(5e4 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_det = Parameter(1e5 * u.GeV, "Emin_det", fixed=True)

    z = 0.4
    Emin_src = Parameter(Emin.value * (z + 1.0), "Emin_src", fixed=True)
    Emax_src = Parameter(Emax.value * (z + 1.0), "Emax_src", fixed=True)

    Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
    Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

    point_source_0 = PointSource.make_powerlaw_source(
        src_names[0],
        np.deg2rad(5) * u.rad,
        np.pi * u.rad,
        src_lumi_params[0],
        src_index_params[0],
        z,
        Emin_src,
        Emax_src,
    )

    point_source_1 = PointSource.make_powerlaw_source(
        src_names[1],
        np.deg2rad(25) * u.rad,
        np.pi * u.rad,
        src_lumi_params[1],
        src_index_params[1],
        z,
        Emin_src,
        Emax_src,
    )

    point_source_2 = PointSource.make_powerlaw_source(
        src_names[2],
        np.deg2rad(45) * u.rad,
        np.pi * u.rad,
        src_lumi_params[2],
        src_index_params[2],
        z,
        Emin_src,
        Emax_src,
    )

    my_sources = Sources()
    my_sources.add(point_source_0)
    my_sources.add(point_source_1)

    sim = Simulation(
        my_sources,
        [IC86_I, IC86_II],
        {IC86_I: 5 * u.year, IC86_II: 5 * u.year},
        asimov=True,
    )
    sim.precomputation()
    sim.generate_stan_code()
    sim.compile_stan_code()

    sim.run()

    assert np.sum(np.rint(sim._Nex_et.sum(axis=0))) == np.sum(list(sim._N.values()))

    my_sources.add(point_source_2)

    sim = Simulation(
        my_sources,
        [IC86_I, IC86_II],
        {IC86_I: 5 * u.year, IC86_II: 5 * u.year},
        asimov=True,
    )

    sim.precomputation()
    sim.setup_stan_sim()
    sim.run()

    assert np.sum(np.rint(sim._Nex_et.sum(axis=0))) == np.sum(list(sim._N.values()))
