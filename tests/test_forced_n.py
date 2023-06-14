import numpy as np
from astropy import units as u

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.simulation import Simulation

def test_N():
    Parameter.clear_registry()

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
    Emin_src = Parameter(Emin.value * (z + 1.), "Emin_src", fixed=True)
    Emax_src = Parameter(Emax.value * (z + 1.), "Emax_src", fixed=True)

    Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
    Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

    point_source = PointSource.make_powerlaw_source(
        "test", np.deg2rad(5) * u.rad, np.pi * u.rad, L, src_index, z, Emin_src, Emax_src
    )

    my_sources = Sources()
    my_sources.add(point_source)

    my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff)
    my_sources.add_atmospheric_component()

    sim = Simulation(my_sources, IceCubeDetectorModel, 5*u.year, N={"cascades": [2, 3, 2], "tracks":[1, 2, 3]})

    sim.precomputation()

    sim.generate_stan_code()

    sim.compile_stan_code()

    sim.run()

    assert np.all(
        np.isclose(
            sim._sim_output.stan_variable("Lambda"),
            np.array([[1., 2., 2., 3., 3., 3., 1., 1., 2., 2., 2.]])
        )
    )