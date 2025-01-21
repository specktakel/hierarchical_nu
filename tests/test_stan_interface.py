import os
import numpy as np
from astropy import units as u
from cmdstanpy import CmdStanModel

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.utils.roi import RectangularROI, ROIList
from hierarchical_nu.detector.icecube import IC86_I, IC86_II
from hierarchical_nu.detector.input import mceq
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

detector_list = [
    IC86_II,
    [IC86_I, IC86_II],
]


stanc_options = {"include-paths": [STAN_PATH, STAN_GEN_PATH]}


def test_stan_sim_interface(output_directory):
    ROIList.clear_registry()
    roi = RectangularROI(DEC_min=-5 * u.deg)
    logger.warning(roi)
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
    file_name = os.path.join(output_directory, "test_sim_interface")

    for dm in detector_list:
        if not isinstance(dm, list):
            dm = [dm]
        interface = StanSimInterface(file_name, my_sources, dm)

        # Generate Stan code
        stan_file = interface.generate()

        # Compile Stan code
        stan_model = CmdStanModel(stan_file=stan_file, stanc_options=stanc_options)


def test_stan_fit_interface(output_directory):
    # Set up sources
    Parameter.clear_registry()
    ROIList.clear_registry()
    roi = RectangularROI(DEC_min=5 * u.deg)
    logger.warning(roi)

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
    file_name = os.path.join(output_directory, "test_fit_interface")

    for dm in detector_list:
        if not isinstance(dm, list):
            dm = [dm]
        for nshards in [1, 2]:
            interface = StanFitInterface(file_name, my_sources, dm, nshards=nshards)

            # Generate Stan code
            stan_file = interface.generate()

            # Compile Stan code
            stan_model = CmdStanModel(stan_file=stan_file, stanc_options=stanc_options)
