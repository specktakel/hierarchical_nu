import pytest
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.source.source import Sources, PointSource
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.utils.roi import RectangularROI, ROIList
from hierarchical_nu.detector.icecube import IC86_II
import numpy as np
import os


@pytest.fixture
def sources():
    def make_sources(ps, diff, atmo):

        Parameter.clear_registry()
        ROIList.clear_registry()
        ra = np.deg2rad(77.35)
        dec = np.deg2rad(5.7)
        bandwidth = np.deg2rad(5)
        DEC_min = (dec - bandwidth) * u.rad
        DEC_max = (dec + bandwidth) * u.rad
        RA_min = (ra - bandwidth) * u.rad
        RA_max = (ra + bandwidth) * u.rad
        roi = RectangularROI(
            DEC_min=DEC_min, DEC_max=DEC_max, RA_min=RA_min, RA_max=RA_max
        )

        Emin_det = Parameter(1e3 * u.GeV, "Emin_det", fixed=True)
        source_list = Sources()
        z = 0.3365
        src_index = Parameter(2.2, "src_index", fixed=False, par_range=(1.0, 4.0))
        L = Parameter(
            1e47 * (u.erg / u.s),
            "luminosity",
            fixed=True,
            par_range=(0, 1e60) * (u.erg / u.s),
        )
        diffuse_norm = Parameter(
            (2.26e-13 / u.GeV / u.m**2 / u.s),
            "diffuse_norm",
            fixed=True,
            par_range=(1e-15, 1e-10) * (1 / u.GeV / u.m**2 / u.s),
        )
        diff_index = Parameter(2.13, "diff_index", fixed=False, par_range=(1.5, 3.5))
        Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
        Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
        Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
        Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
        Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)
        Emin_src = Parameter(Emin.value * (1 + z), "Emin_src", fixed=True)
        Emax_src = Parameter(Emax.value * (1 + z), "Emax_src", fixed=True)
        if ps:
            ps = PointSource.make_powerlaw_source(
                "txs", dec * u.rad, ra * u.rad, L, src_index, z, Emin_src, Emax_src
            )
            source_list.add(ps)
        if diff:
            source_list.add_diffuse_component(
                diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff
            )
        if atmo:
            source_list.add_atmospheric_component()

        return source_list

    return make_sources


def test_ps(output_directory, sources):
    my_sources = sources(True, False, False)
    sim = Simulation(my_sources, IC86_II, {IC86_II: 180 * u.d})

    sim.precomputation()

    sim._get_expected_Nnu(sim._get_sim_inputs())

    sim.generate_stan_code()
    sim.compile_stan_code()

    for i in range(5):
        sim.run(verbose=True, seed=i)
        sim.save(os.path.join(output_directory, f"ps_only_{i}.h5"), overwrite=True)

    events = Events.from_file(os.path.join(output_directory, f"ps_only_0.h5"))

    events.N

    fit = StanFit(my_sources, IC86_II, events, {IC86_II: 180 * u.d})

    fit.precomputation()

    fit.generate_stan_code()
    fit.compile_stan_code()

    for i in range(5):
        events = Events.from_file(os.path.join(output_directory, f"ps_only_{i}.h5"))
        fit.events = events
        fit.run(seed=42, show_progress=True, inits={"L": 1e50, "src_index": 2.2})
        fit.save(os.path.join(output_directory, f"ps_only_fit_{i}.h5"))

    mt_fit = StanFit(my_sources, IC86_II, events, {IC86_II: 180 * u.d}, nshards=2)

    mt_fit.precomputation()

    mt_fit.generate_stan_code()
    mt_fit.compile_stan_code()

    for i in range(5):
        events = Events.from_file(os.path.join(output_directory, f"ps_only_{i}.h5"))
        mt_fit.events = events
        mt_fit.run(seed=42, show_progress=True, inits={"L": 1e50, "src_index": 2.2})
        mt_fit.save(os.path.join(output_directory, f"ps_only_mt_fit_{i}.h5"))

    for i in range(5):
        mt = StanFit.from_file(os.path.join(output_directory, f"ps_only_mt_fit_{i}.h5"))
        fit = StanFit.from_file(os.path.join(output_directory, f"ps_only_fit_{i}.h5"))
        bins = np.linspace(1.0, 4.0, 20)
        mt_n, _ = np.histogram(mt._fit_output["src_index"][0], bins=bins, density=True)
        n, _ = np.histogram(fit._fit_output["src_index"][0], bins=bins, density=True)
        assert np.sum(~np.isclose(mt_n, n, atol=0.3)) < 3


def test_lp(output_directory, sources):
    my_sources = sources(True, False, False)
    events = Events.from_file(os.path.join(output_directory, f"ps_only_0.h5"))
    fit = StanFit(my_sources, IC86_II, events, 180 * u.d, nshards=2, debug=True)
    fit.precomputation()
    fit.generate_stan_code()
    fit.compile_stan_code()

    fit.run(iterations=1, iter_warmup=1)

    assert np.all(
        np.isclose(
            fit._fit_output.stan_variable("lp").squeeze(),
            fit._fit_output.stan_variable("lp_gen_q").squeeze(),
        )
    )


def test_ps_diff(output_directory, sources):

    my_sources = sources(True, True, False)
    sim = Simulation(my_sources, IC86_II, 180 * u.d)

    sim.precomputation()

    sim._get_expected_Nnu(sim._get_sim_inputs())

    sim.generate_stan_code()
    sim.compile_stan_code()

    for i in range(3):
        sim.run(verbose=True, seed=i)
        sim.save(os.path.join(output_directory, f"ps_only_{i}.h5"), overwrite=True)

    events = Events.from_file(os.path.join(output_directory, f"ps_only_0.h5"))

    events.N

    fit = StanFit(my_sources, IC86_II, events, 180 * u.d)

    fit.precomputation()

    fit.generate_stan_code()
    fit.compile_stan_code()

    for i in range(3):
        events = Events.from_file(os.path.join(output_directory, f"ps_only_{i}.h5"))
        fit.events = events
        fit.run(seed=42, show_progress=True, inits={"L": 1e50, "src_index": 2.2})
        fit.save(os.path.join(output_directory, f"ps_only_fit_{i}.h5"))

    mt_fit = StanFit(my_sources, IC86_II, events, 180 * u.d, nshards=2)

    mt_fit.precomputation()

    mt_fit.generate_stan_code()
    mt_fit.compile_stan_code()

    for i in range(3):
        events = Events.from_file(os.path.join(output_directory, f"ps_only_{i}.h5"))
        mt_fit.events = events
        mt_fit.run(seed=42, show_progress=True, inits={"L": 1e50, "src_index": 2.2})
        mt_fit.save(os.path.join(output_directory, f"ps_only_mt_fit_{i}.h5"))

    for i in range(3):
        mt = StanFit.from_file(os.path.join(output_directory, f"ps_only_mt_fit_{i}.h5"))
        fit = StanFit.from_file(os.path.join(output_directory, f"ps_only_fit_{i}.h5"))
        bins = np.linspace(1.0, 4.0, 20)
        mt_n, _ = np.histogram(mt._fit_output["src_index"][0], bins=bins, density=True)
        n, _ = np.histogram(fit._fit_output["src_index"][0], bins=bins, density=True)
        assert np.sum(~np.isclose(mt_n, n, atol=0.3)) < 3
