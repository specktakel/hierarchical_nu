import numpy as np
import h5py
import time
from joblib import Parallel, delayed
from astropy import units as u

from .source.atmospheric_flux import AtmosphericNuMuFlux
from .source.flux_model import PowerLawSpectrum
from .detector_model import NorthernTracksDetectorModel
from .simulation import __generate_atmospheric_sim_code, __generate_main_sim_code
from .fit import __generate_stan_fit_code


class ModelCheck:
    """
    Check statistical model by repeatedly
    fitting simulated data using different random seeds.
    """

    def __init__(self):

        pass

    @classmethod
    @u.quantity_input
    def initialise_env(
        cls,
        Emin: u.GeV,
        Emax: u.GeV,
        atmo_sim_name,
        main_sim_name,
        fit_name,
    ):
        """
        Script to setup enviroment for parallel
        model checking runs.

        * Runs MCEq for atmo flux if needed
        * Generates and compiles necessary Stan files
        Only need to run once before calling ModelCheck(...).run()
        """

        # Run MCEq computation
        print("Setting up MCEq run for AtmopshericNumuFlux")
        atmo_flux_model = AtmosphericNuMuFlux(Emin.value, Emax.value)

        atmo_sim_filename = __generate_atmospheric_sim_code(
            atmo_sim_name, atmo_flux_model, theta_points=30
        )
        print("Generated atmo_sim Stan file at:", atmo_sim_filename)

        ps_spec_shape = PowerLawSpectrum
        detector_model_type = NorthernTracksDetectorModel
        main_sim_filename = __generate_main_sim_code(
            main_sim_name, ps_spec_shape, detector_model_type
        )
        print("Generated main_sim Stan file at:", main_sim_filename)

        fit_filename = __generate_stan_fit_code(
            fit_name,
            ps_spec_shape,
            atmo_flux_model,
            detector_model_type,
            diffuse_bg_comp=True,
            atmospheric_comp=True,
            theta_points=30,
        )
        print("Generated fit Stan file at:", fit_filename)

    def run(self):

        pass
