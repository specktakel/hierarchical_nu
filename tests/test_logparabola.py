from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter, ParScale
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.priors import Priors, EnergyPrior, LogNormalPrior
from hierarchical_nu.source.source import Sources, PointSource, SourceFrame, DetectorFrame
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior, LuminosityPrior, IndexPrior, FluxPrior
from hierarchical_nu.detector.input import mceq
from hierarchical_nu.utils.roi import CircularROI, ROIList
from hierarchical_nu.detector.icecube import IC86_II, IC86_I, IC40, IC59, IC79
from icecube_tools.utils.data import Uptime
import numpy as np
import matplotlib.pyplot as plt
import ligo.skymap.plot
import sys
from pathlib import Path

import pytest


# Re-use the same simulated data set
@pytest.fixture(scope='session')
def simulation(output_directory):
    Parameter.clear_registry()
    src_index = Parameter(2.2, "src_index", fixed=False, par_range=(1., 4.))
    beta_index = Parameter(0.5, "beta_index", fixed=False, par_range=(-.5, 1.))
    E0 = Parameter(1e7*u.GeV, "E0_src", fixed=True, par_range=(1e3, 1e9)*u.GeV, scale=ParScale.log)
    L = Parameter(1E47 * (u.erg / u.s), "luminosity", fixed=True, par_range=(0, 1E60) * (u.erg/u.s))
    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    z = 0.3365
    Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1E2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2*u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8*u.GeV, "Emax_src", fixed=True)
    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_powerlaw_source(
        "test", dec, ra, L, src_index, z, Emin_src, Emax_src, frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5*u.yr}

    sim = Simulation(sources, IC86_II, lifetime)
    sim.precomputation()
    sim.generate_stan_code()
    sim.compile_stan_code()
    sim.run()

    sim.save(output_directory / Path("events.h5"))

    events = Events.from_file(output_directory / Path("events.h5"))

    return events


# Run through all combinations of one of alpha, beta, E0 being fixed, named in function
def test_logparabola_E0(simulation):
    Parameter.clear_registry()
    src_index = Parameter(2., "src_index", fixed=False, par_range=(1., 4.))
    beta_index = Parameter(0.5, "beta_index", fixed=False, par_range=(-.5, 1.))
    E0 = Parameter(1e7*u.GeV, "E0_src", fixed=True, par_range=(1e3, 1e9)*u.GeV, scale=ParScale.log)
    L = Parameter(1E47 * (u.erg / u.s), "luminosity", fixed=True, 
                par_range=(0, 1E60) * (u.erg/u.s))
    z = 0.3365
    Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1E2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2*u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8*u.GeV, "Emax_src", fixed=True)

    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test", dec, ra, L, src_index, beta_index, z, Emin_src, Emax_src, E0, frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5*u.yr}


    # Less grid points to speed up testing
    fit = StanFit(sources, IC86_II, simulation, lifetime, n_grid_points=20)
    fit.precomputation()
    fit.generate_stan_code()
    fit.compile_stan_code()
    fit.run(
        inits={"E": fit.events.N * [1e5], "L": 1e48, "src_index": 2.0, "beta_index": 0.0},
        show_console=True,
    )

def test_logparabola_beta(simulation):
    Parameter.clear_registry()
    src_index = Parameter(2., "src_index", fixed=False, par_range=(1., 4.))
    beta_index = Parameter(0., "beta_index", fixed=True, par_range=(-.5, 1.))
    E0 = Parameter(1e7*u.GeV, "E0_src", fixed=False, par_range=(1e3, 1e9)*u.GeV, scale=ParScale.log)
    L = Parameter(1E47 * (u.erg / u.s), "luminosity", fixed=True, 
                par_range=(0, 1E60) * (u.erg/u.s))
    z = 0.3365
    Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1E2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2*u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8*u.GeV, "Emax_src", fixed=True)

    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test", dec, ra, L, src_index, beta_index, z, Emin_src, Emax_src, E0, frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5*u.yr}


    # Less grid points to speed up testing
    # use mulithreading for one of the tests
    fit = StanFit(sources, IC86_II, simulation, lifetime, n_grid_points=20, nshards=2)
    fit.precomputation()
    fit.generate_stan_code()
    fit.compile_stan_code()
    fit.run(inits={"E": fit.events.N * [1e5], "L": 1e48, "src_index": 2.0, "E0": 1e7})

def test_logparabola_index(simulation):
    Parameter.clear_registry()
    src_index = Parameter(2., "src_index", fixed=True, par_range=(1., 4.))
    beta_index = Parameter(0., "beta_index", fixed=False, par_range=(-.5, 1.))
    E0 = Parameter(1e7*u.GeV, "E0_src", fixed=False, par_range=(1e3, 1e9)*u.GeV, scale=ParScale.log)
    L = Parameter(1E47 * (u.erg / u.s), "luminosity", fixed=True, 
                par_range=(0, 1E60) * (u.erg/u.s))
    z = 0.3365
    Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(1E2 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(1e2*u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8*u.GeV, "Emax_src", fixed=True)

    Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)

    ra = np.deg2rad(77.35) * u.rad
    dec = np.deg2rad(5.7) * u.rad
    txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
    point_source = PointSource.make_logparabola_source(
        "test", dec, ra, L, src_index, beta_index, z, Emin_src, Emax_src, E0, frame=DetectorFrame,
    )

    sources = Sources()
    sources.add(point_source)

    ROIList.clear_registry()
    roi = CircularROI(txs, 5 * u.deg, apply_roi=True)

    lifetime = {IC86_II: 0.5*u.yr}


    # Less grid points to speed up testing
    fit = StanFit(sources, IC86_II, simulation, lifetime, n_grid_points=20)
    fit.precomputation()
    fit.generate_stan_code()
    fit.compile_stan_code()
    fit.run(inits={"E": fit.events.N * [1e5], "L": 1e48, "beta_index": 0.05, "E0": 1e7})