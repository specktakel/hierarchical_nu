import os
import numpy as np
from astropy import units as u
from cmdstanpy import CmdStanModel

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource

from hierarchical_nu.stan.interface import STAN_PATH
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.detector.r2021 import R2021DetectorModel

# Set up sources
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

point_source = PointSource.make_powerlaw_source(
    "test", np.deg2rad(5) * u.rad, np.pi * u.rad, L, src_index, 0.4, Emin, Emax
)

my_sources = Sources()
my_sources.add(point_source)

my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index)
# my_sources.add_atmospheric_component()

detector_models = [
    NorthernTracksDetectorModel,
    CascadesDetectorModel,
    IceCubeDetectorModel,
    R2021DetectorModel,
]

stanc_options = {"include-paths": [STAN_PATH]}


def test_stan_sim_interface(output_directory):

    file_name = os.path.join(output_directory, "test_sim_interface")

    for dm in detector_models:

        interface = StanSimInterface(file_name, my_sources, dm)

        # Generate Stan code
        stan_file = interface.generate()

        # Compile Stan code
        stan_model = CmdStanModel(stan_file=stan_file, stanc_options=stanc_options)


def test_stan_fit_interface(output_directory):

    file_name = os.path.join(output_directory, "test_fit_interface")

    for dm in detector_models:

        for nshards in [1, 2]:

            interface = StanFitInterface(file_name, my_sources, dm, nshards=nshards)

            # Generate Stan code
            stan_file = interface.generate()

            # Compile Stan code
            stan_model = CmdStanModel(stan_file=stan_file, stanc_options=stanc_options)
