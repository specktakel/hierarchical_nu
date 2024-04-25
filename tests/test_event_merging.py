from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.detector.input import mceq
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.simulation import Simulation, SimInfo
from hierarchical_nu.fit import StanFit
from hierarchical_nu.source.source import Sources, PointSource
from hierarchical_nu.events import Events
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.utils.roi import CircularROI, ROIList
from hierarchical_nu.detector.icecube import IC86_II
from hierarchical_nu.detector.input import mceq
import numpy as np


def test_merging(output_directory):
    Parameter.clear_registry()
    src_index = Parameter(
        value=2.2,
        name="src_index",
        fixed=False,
        par_range=(1.0, 4.0),
    )
    diff_index = Parameter(
        value=2.0,
        name="diff_index",
        fixed=False,
        par_range=(1.0, 4.0),
    )
    L = Parameter(
        value=1e47 * (u.erg / u.s),
        name="luminosity",
        fixed=True,
        par_range=(0, 1e55) * (u.erg / u.s),
    )
    diffuse_norm = Parameter(
        value=2.26e-13 / u.GeV / u.s / u.m**2,
        name="diffuse_norm",
        fixed=True,
        par_range=(0, np.inf),
    )

    z = 0.3365  # Redshift
    # True energy range and normalisation
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_diff = Parameter(1e2 * u.GeV, "Emin_diff", fixed=True)
    Emax_diff = Parameter(1e8 * u.GeV, "Emax_diff", fixed=True)
    Emin_src = Parameter(1e2 * (1 + z) * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8 * (1 + z) * u.GeV, "Emax_src", fixed=True)

    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    sources = Sources()
    ps = PointSource.make_powerlaw_source(
        "txs",
        5.7 * u.deg,
        77.3 * u.deg,
        L,
        src_index,
        0.3365,
        Emin_src,
        Emax_src,
    )
    sources.add(ps)

    ra, dec = sources[0].ra, sources[0].dec
    source_coords = SkyCoord(ra=ra, dec=dec, frame="icrs")

    roi = CircularROI(source_coords, 5 * u.deg)

    sim = Simulation(sources, IC86_II, 0.5 * u.year)

    sim.precomputation()
    sim.generate_stan_code()
    sim.compile_stan_code()
    sim._get_expected_Nnu(sim._get_sim_inputs())

    sim.run()
    sim.save(output_directory / Path("ps.h5"))

    sources = Sources()
    sources.add_diffuse_component(
        diffuse_norm,
        1e5 * u.GeV,
        diff_index,
        Emin_diff,
        Emax_diff,
    )
    sources.add_atmospheric_component(cache_dir=mceq)

    sim = Simulation(sources, IC86_II, 0.5 * u.year)

    sim.precomputation()
    sim.generate_stan_code()
    sim.compile_stan_code()
    sim.run()
    sim.save(output_directory / Path("bg.h5"))

    SimInfo.merge(
        output_directory / Path("bg.h5"),
        output_directory / Path("ps.h5"),
        output_directory / Path("combined.h5"),
    )

    bg = Events.from_file(output_directory / Path("bg.h5"))
    ps = Events.from_file(output_directory / Path("ps.h5"))
    comb = Events.from_file(output_directory / Path("combined.h5"))

    assert np.all(
        np.isclose(
            np.array(
                [
                    ps.energies.to_value(u.GeV).tolist()
                    + bg.energies.to_value(u.GeV).tolist()
                ]
            )
            * u.GeV,
            comb.energies,
        )
    )

    assert np.all(
        np.isclose(np.vstack((ps.unit_vectors, bg.unit_vectors)), comb.unit_vectors)
    )

    assert np.all(
        np.isclose(
            np.array(
                ps.ang_errs.to_value(u.deg).tolist()
                + bg.ang_errs.to_value(u.deg).tolist()
            ),
            comb.ang_errs.to_value(u.deg),
        )
    )

    assert np.all(
        np.isclose(
            np.array(ps.mjd.to_value("mjd").tolist() + bg.mjd.to_value("mjd").tolist()),
            comb.mjd.to_value("mjd").tolist(),
        )
    )

    assert np.all(np.isclose(ps.types.tolist() + bg.types.tolist(), comb.types))
