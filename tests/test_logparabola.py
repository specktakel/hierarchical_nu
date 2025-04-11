from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter, ParScale
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.source.source import (
    Sources,
    PointSource,
    DetectorFrame,
)
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.detector.input import mceq
from hierarchical_nu.utils.roi import CircularROI, ROIList
from hierarchical_nu.detector.icecube import IC86_II
import numpy as np
from pathlib import Path

import pytest


@pytest.fixture
def simulation_E0(output_directory):
    Parameter.clear_registry()
    src_index = Parameter(2.2, "src_index", fixed=False, par_range=(1.0, 4.0))
    beta_index = Parameter(0.5, "beta_index", fixed=False, par_range=(-0.5, 1.0))
    E0 = Parameter(
        1e5 * u.GeV,
        "E0_src",
        fixed=True,
        par_range=(1e3, 1e9) * u.GeV,
        scale=ParScale.log,
    )
    L = Parameter(
        1e47 * (u.erg / u.s),
        "luminosity",
        fixed=True,
        par_range=(0, 1e60) * (u.erg / u.s),
    )
    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    z = 0.3365
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2 * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)
    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test",
        dec,
        ra,
        L,
        src_index,
        beta_index,
        z,
        Emin_src,
        Emax_src,
        E0,
        frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5 * u.yr}

    sim = Simulation(sources, IC86_II, lifetime, n_grid_points=20)
    sim.precomputation()
    sim.generate_stan_code()
    sim.compile_stan_code()
    sim.run()

    sim.save(output_directory / Path("events.h5"), overwrite=True)

    events = Events.from_file(output_directory / Path("events.h5"))

    return events


@pytest.fixture
def simulation_beta_E0(output_directory):
    Parameter.clear_registry()
    src_index = Parameter(2.2, "src_index", fixed=False, par_range=(1.0, 4.0))
    beta_index = Parameter(0.5, "beta_index", fixed=True, par_range=(-0.5, 1.0))
    E0 = Parameter(
        1e5 * u.GeV,
        "E0_src",
        fixed=True,
        par_range=(1e3, 1e9) * u.GeV,
        scale=ParScale.log,
    )
    L = Parameter(
        1e47 * (u.erg / u.s),
        "luminosity",
        fixed=True,
        par_range=(0, 1e60) * (u.erg / u.s),
    )
    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    z = 0.3365
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2 * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)
    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test",
        dec,
        ra,
        L,
        src_index,
        beta_index,
        z,
        Emin_src,
        Emax_src,
        E0,
        frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5 * u.yr}

    sim = Simulation(sources, IC86_II, lifetime, n_grid_points=20)
    sim.precomputation()
    sim.generate_stan_code()
    sim.compile_stan_code()
    sim.run()

    sim.save(output_directory / Path("events.h5"), overwrite=True)

    events = Events.from_file(output_directory / Path("events.h5"))

    return events


@pytest.fixture
def simulation_index(output_directory):
    Parameter.clear_registry()
    src_index = Parameter(2.2, "src_index", fixed=True, par_range=(1.0, 4.0))
    beta_index = Parameter(0.5, "beta_index", fixed=False, par_range=(-0.5, 1.0))
    E0 = Parameter(
        1e5 * u.GeV,
        "E0_src",
        fixed=False,
        par_range=(1e3, 1e9) * u.GeV,
        scale=ParScale.log,
    )
    L = Parameter(
        1e47 * (u.erg / u.s),
        "luminosity",
        fixed=True,
        par_range=(0, 1e60) * (u.erg / u.s),
    )
    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    z = 0.3365
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2 * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)
    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test",
        dec,
        ra,
        L,
        src_index,
        beta_index,
        z,
        Emin_src,
        Emax_src,
        E0,
        frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5 * u.yr}

    sim = Simulation(sources, IC86_II, lifetime, n_grid_points=20)
    sim.precomputation()
    sim.generate_stan_code()
    sim.compile_stan_code()
    sim.run()

    sim.save(output_directory / Path("events.h5"), overwrite=True)

    events = Events.from_file(output_directory / Path("events.h5"))

    return events


# Run through all combinations of one of alpha, beta, E0 being fixed, named in function
def test_logparabola_E0(simulation_E0):
    Parameter.clear_registry()
    src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1.0, 4.0))
    beta_index = Parameter(0.5, "beta_index", fixed=False, par_range=(-0.5, 1.0))
    E0 = Parameter(
        1e7 * u.GeV,
        "E0_src",
        fixed=True,
        par_range=(1e3, 1e9) * u.GeV,
        scale=ParScale.log,
    )
    L = Parameter(
        1e47 * (u.erg / u.s),
        "luminosity",
        fixed=True,
        par_range=(0, 1e60) * (u.erg / u.s),
    )
    z = 0.3365
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2 * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)

    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test",
        dec,
        ra,
        L,
        src_index,
        beta_index,
        z,
        Emin_src,
        Emax_src,
        E0,
        frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5 * u.yr}

    # Less grid points to speed up testing
    fit = StanFit(sources, IC86_II, simulation_E0, lifetime, n_grid_points=20)
    fit.precomputation()
    fit.generate_stan_code()
    fit.compile_stan_code()
    fit.run(
        inits={
            "E": fit.events.N * [1e5],
            "L": 1e48,
            "src_index": 2.0,
            "beta_index": 0.0,
        },
        show_console=True,
    )


def test_logparabola_beta_E0(simulation_beta_E0):
    Parameter.clear_registry()
    src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1.0, 4.0))
    beta_index = Parameter(0.0, "beta_index", fixed=True, par_range=(-0.5, 1.0))
    E0 = Parameter(
        1e5 * u.GeV,
        "E0_src",
        fixed=True,
        par_range=(1e3, 1e9) * u.GeV,
        scale=ParScale.log,
    )
    L = Parameter(
        1e47 * (u.erg / u.s),
        "luminosity",
        fixed=True,
        par_range=(0, 1e60) * (u.erg / u.s),
    )
    z = 0.3365
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2 * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)

    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test",
        dec,
        ra,
        L,
        src_index,
        beta_index,
        z,
        Emin_src,
        Emax_src,
        E0,
        frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5 * u.yr}

    # Less grid points to speed up testing
    # use mulithreading for one of the tests
    fit = StanFit(
        sources, IC86_II, simulation_beta_E0, lifetime, n_grid_points=20, nshards=2
    )
    print(fit._logparabola)
    print(fit._fit_index)
    print(fit._fit_beta)
    print(fit._fit_Enorm)
    fit.precomputation()
    fit.generate_stan_code()
    fit.compile_stan_code()
    fit.run(
        inits={"E": fit.events.N * [1e5], "L": 1e48, "src_index": 2.0, "E0_src": 1e5}
    )


def test_logparabola_index(simulation_index):
    Parameter.clear_registry()
    src_index = Parameter(2.0, "src_index", fixed=True, par_range=(1.0, 4.0))
    beta_index = Parameter(0.0, "beta_index", fixed=False, par_range=(-0.5, 1.0))
    E0 = Parameter(
        1e7 * u.GeV,
        "E0_src",
        fixed=False,
        par_range=(1e3, 1e9) * u.GeV,
        scale=ParScale.log,
    )
    L = Parameter(
        1e47 * (u.erg / u.s),
        "luminosity",
        fixed=True,
        par_range=(0, 1e60) * (u.erg / u.s),
    )
    z = 0.3365
    Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2 * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)

    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test",
        dec,
        ra,
        L,
        src_index,
        beta_index,
        z,
        Emin_src,
        Emax_src,
        E0,
        frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5 * u.yr}

    # Less grid points to speed up testing
    fit = StanFit(sources, IC86_II, simulation_index, lifetime, n_grid_points=20)
    fit.precomputation()
    fit.generate_stan_code()
    fit.compile_stan_code()
    print(fit._logparabola)
    print(fit._fit_index)
    print(fit._fit_beta)
    print(fit._fit_Enorm)
    fit.run(
        inits={"E": fit.events.N * [1e5], "L": 1e48, "beta_index": 0.05, "E0_src": 1e7}
    )
