import os
import sys
import numpy as np
import h5py
import time
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from astropy import units as u
from cmdstanpy import CmdStanModel

from .source.atmospheric_flux import AtmosphericNuMuFlux
from .source.flux_model import PowerLawSpectrum
from .source.parameter import Parameter
from .source.source import PointSource, Sources
from .detector_model import NorthernTracksDetectorModel
from .simulation import generate_atmospheric_sim_code_, generate_main_sim_code_
from .fit import generate_stan_fit_code_
from .config import FileConfig, ParameterConfig

from python.simulation import Simulation
from python.fit import StanFit


class ModelCheck:
    """
    Check statistical model by repeatedly
    fitting simulated data using different random seeds.
    """

    def __init__(self):

        parameter_config = ParameterConfig()

        Parameter.clear_registry()
        index = Parameter(
            parameter_config["alpha"],
            "index",
            fixed=False,
            par_range=parameter_config["alpha_range"],
        )
        L = Parameter(
            parameter_config["L"] * u.erg / u.s,
            "luminosity",
            fixed=True,
            par_range=parameter_config["L_range"],
        )
        diffuse_norm = Parameter(
            parameter_config["diff_norm"] * 1 / (u.GeV * u.m ** 2 * u.s),
            "diffuse_norm",
            fixed=True,
            par_range=(0, np.inf),
        )
        Enorm = Parameter(parameter_config["Enorm"] * u.GeV, "Enorm", fixed=True)
        Emin = Parameter(parameter_config["Emin"] * u.GeV, "Emin", fixed=True)
        Emax = Parameter(parameter_config["Emax"] * u.GeV, "Emax", fixed=True)

        point_source = PointSource.make_powerlaw_source(
            "test", np.deg2rad(5) * u.rad, np.pi * u.rad, L, index, 0.43, Emin, Emax
        )

        self._sources = Sources()
        self._sources.add(point_source)
        self._sources.add_diffuse_component(diffuse_norm, Enorm.value)
        self._sources.add_atmospheric_component()

        f = self._sources.associated_fraction().value

        self.truths = {}
        diffuse_bg = self._sources.diffuse_component()
        self.truths["F_diff"] = diffuse_bg.flux_model.total_flux_int.value
        atmo_bg = self._sources.atmo_component()
        self.truths["F_atmo"] = atmo_bg.flux_model.total_flux_int.value
        self.truths["L"] = L.value.to(u.GeV / u.s).value
        self.truths["f"] = f
        self.truths["alpha"] = parameter_config["alpha"]

        self._default_var_names = [key for key in self.truths]

    @classmethod
    @u.quantity_input
    def initialise_env(
        cls,
        output_dir,
    ):
        """
        Script to setup enviroment for parallel
        model checking runs.

        * Runs MCEq for atmo flux if needed
        * Generates and compiles necessary Stan files
        Only need to run once before calling ModelCheck(...).run()
        """

        parameter_config = ParameterConfig()

        # Run MCEq computation
        print("Setting up MCEq run for AtmopshericNumuFlux")
        Emin = parameter_config["Emin"] * u.GeV
        Emax = parameter_config["Emax"] * u.GeV
        atmo_flux_model = AtmosphericNuMuFlux(Emin, Emax)

        file_config = FileConfig()

        atmo_sim_name = file_config["atmo_sim_filename"][:-5]
        _ = generate_atmospheric_sim_code_(
            atmo_sim_name, atmo_flux_model, theta_points=30
        )
        print("Generated atmo_sim Stan file at:", file_config["atmo_sim_filename"])

        ps_spec_shape = PowerLawSpectrum
        detector_model_type = NorthernTracksDetectorModel
        main_sim_name = file_config["main_sim_filename"][:-5]
        _ = generate_main_sim_code_(main_sim_name, ps_spec_shape, detector_model_type)
        print("Generated main_sim Stan file at:", file_config["main_sim_filename"])

        fit_name = file_config["fit_filename"][:-5]
        _ = generate_stan_fit_code_(
            fit_name,
            ps_spec_shape,
            atmo_flux_model,
            detector_model_type,
            diffuse_bg_comp=True,
            atmospheric_comp=True,
            theta_points=30,
        )
        print("Generated fit Stan file at:", file_config["fit_filename"])

        print("Compile Stan models")
        stanc_options = {"include_paths": file_config["include_paths"]}

        _ = CmdStanModel(
            stan_file=file_config["atmo_sim_filename"],
            stanc_options=stanc_options,
        )
        _ = CmdStanModel(
            stan_file=file_config["main_sim_filename"],
            stanc_options=stanc_options,
        )
        _ = CmdStanModel(
            stan_file=file_config["fit_filename"], stanc_options=stanc_options
        )

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def parallel_run(self, n_jobs=1, n_subjobs=1, seed=None):

        job_seeds = [(seed + job) * 10 for job in range(n_jobs)]

        self._results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._single_run)(n_subjobs, seed=s) for s in job_seeds
        )

    def save(self, filename):

        with h5py.File(filename, "w") as f:

            truths_folder = f.create_group("truths")
            for key, value in self.truths.items():
                truths_folder.create_dataset(key, data=value)

            for i, res in enumerate(self._results):
                folder = f.create_group("results_%i" % i)

                for key, value in res.items():
                    folder.create_dataset(key, data=value)

    def load(self, filename_list):

        self.truths = {}

        self.results = {}
        self.results["F_atmo"] = []
        self.results["F_diff"] = []
        self.results["L"] = []
        self.results["alpha"] = []
        self.results["f"] = []

        file_truths = {}
        for filename in filename_list:
            with h5py.File(filename, "r") as f:

                truths_folder = f["truths"]
                for key, value in truths_folder.items():
                    file_truths[key] = value[()]

                if not self.truths:
                    self.truths = file_truths

                if self.truths != file_truths:
                    raise ValueError(
                        "Files in list have different truth settings and should not be combined"
                    )

                n_jobs = len([key for key in f if key != "truths"])
                for i in range(n_jobs):
                    job_folder = f["results_%i" % i]
                    self.results["F_atmo"].extend(job_folder["F_atmo"][()])
                    self.results["F_diff"].extend(job_folder["F_diff"][()])
                    self.results["L"].extend(job_folder["L"][()])
                    self.results["alpha"].extend(job_folder["alpha"][()])
                    self.results["f"].extend(job_folder["f"][()])

    def compare(self, var_names=None):

        if not var_names:
            var_names = self._default_var_names

        N = len(var_names)
        fig, ax = plt.subplots(N, figsize=(5, 15))

        for v, var_name in enumerate(var_names):
            for i in range(len(self.results[var_name])):
                ax[v].hist(
                    self.results[var_name][i],
                    color="g",
                    alpha=0.1,
                    histtype="step",
                )

            ax[v].axvline(self.truths[var_name], color="k", linestyle="--")
            ax[v].set_xlabel(var_name)

        fig.tight_layout()
        return fig, ax

    def _single_run(self, n_subjobs, seed):
        """
        Single run to be called using Parallel.
        """

        sys.stderr.write("Random seed: %i\n" % seed)

        parameter_config = ParameterConfig()
        file_config = FileConfig()
        Parameter.clear_registry()
        index = Parameter(
            parameter_config["alpha"],
            "index",
            fixed=False,
            par_range=parameter_config["alpha_range"],
        )
        L = Parameter(
            parameter_config["L"] * u.erg / u.s,
            "luminosity",
            fixed=True,
            par_range=parameter_config["L_range"],
        )
        diffuse_norm = Parameter(
            parameter_config["diff_norm"] * 1 / (u.GeV * u.m ** 2 * u.s),
            "diffuse_norm",
            fixed=True,
            par_range=(0, np.inf),
        )
        Enorm = Parameter(parameter_config["Enorm"] * u.GeV, "Enorm", fixed=True)
        Emin = Parameter(parameter_config["Emin"] * u.GeV, "Emin", fixed=True)
        Emax = Parameter(parameter_config["Emax"] * u.GeV, "Emax", fixed=True)
        Emin_det = Parameter(
            parameter_config["Emin_det"] * u.GeV, "Emin_det", fixed=True
        )

        point_source = PointSource.make_powerlaw_source(
            "test", np.deg2rad(5) * u.rad, np.pi * u.rad, L, index, 0.43, Emin, Emax
        )

        self._sources = Sources()
        self._sources.add(point_source)
        self._sources.add_diffuse_component(diffuse_norm, Enorm.value)
        self._sources.add_atmospheric_component()

        subjob_seeds = [(seed + subjob) * 10 for subjob in range(n_subjobs)]

        start_time = time.time()

        outputs = {}
        outputs["F_diff"] = []
        outputs["F_atmo"] = []
        outputs["L"] = []
        outputs["f"] = []
        outputs["alpha"] = []

        for i, s in enumerate(subjob_seeds):

            sys.stderr.write("Run %i\n" % i)

            # Simulation
            obs_time = parameter_config["obs_time"] * u.year
            sim = Simulation(self._sources, NorthernTracksDetectorModel, obs_time)
            sim.precomputation()
            sim.set_stan_filenames(
                file_config["atmo_sim_filename"], file_config["main_sim_filename"]
            )
            sim.compile_stan_code(include_paths=file_config["include_paths"])
            sim.run(seed=s)
            self.sim = sim

            lam = sim._sim_output.stan_variable("Lambda").values[0]
            sim_output = {}
            sim_output["Lambda"] = lam

            self.events = sim.events

            # Fit
            fit = StanFit(
                self._sources, NorthernTracksDetectorModel, sim.events, obs_time
            )
            fit.precomputation(exposure_integral=sim._exposure_integral)
            fit.set_stan_filename(file_config["fit_filename"])
            fit.compile_stan_code(include_paths=file_config["include_paths"])
            fit.run(seed=s)

            self.fit = fit

            # Store output
            outputs["F_diff"].append(
                fit._fit_output.stan_variable("F_diff").values.T[0]
            )
            outputs["F_atmo"].append(
                fit._fit_output.stan_variable("F_atmo").values.T[0]
            )
            outputs["L"].append(fit._fit_output.stan_variable("L").values.T[0])
            outputs["f"].append(fit._fit_output.stan_variable("f").values.T[0])
            outputs["alpha"].append(fit._fit_output.stan_variable("alpha").values.T[0])

            fit.check_classification(sim_output)

            sys.stderr.write("time: %.5f\n" % (time.time() - start_time))

        return outputs
