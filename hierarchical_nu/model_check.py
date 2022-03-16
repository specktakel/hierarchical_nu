import os
import sys
import numpy as np
import h5py
import time
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from astropy import units as u
from cmdstanpy import CmdStanModel
from scipy.stats import lognorm, norm, uniform

from hierarchical_nu.source.atmospheric_flux import AtmosphericNuMuFlux
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import PointSource, Sources

from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel

from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.utils.config import hnu_config


class ModelCheck:
    """
    Check statistical model by repeatedly
    fitting simulated data using different random seeds.
    """

    def __init__(self):

        self._sources = _initialise_sources()

        f_arr = self._sources.f_arr().value

        f_arr_astro = self._sources.f_arr_astro().value

        self.truths = {}

        flux_unit = 1 / (u.m ** 2 * u.s)

        diffuse_bg = self._sources.diffuse
        self.truths["F_diff"] = diffuse_bg.flux_model.total_flux_int.to(flux_unit).value
        atmo_bg = self._sources.atmospheric
        self.truths["F_atmo"] = atmo_bg.flux_model.total_flux_int.to(flux_unit).value
        self.truths["L"] = (
            Parameter.get_parameter("luminosity").value.to(u.GeV / u.s).value
        )
        self.truths["f_arr"] = f_arr
        self.truths["f_arr_astro"] = f_arr_astro
        self.truths["src_index"] = Parameter.get_parameter("src_index").value
        self.truths["diff_index"] = Parameter.get_parameter("diff_index").value

        self._default_var_names = [key for key in self.truths]

        self._default_var_labels = [
            r"$F_\mathrm{diff}$ / $\mathrm{m}^{-2}~\mathrm{s}^{-1}$",
            r"$F_\mathrm{atmo}$ / $\mathrm{m}^{-2}~\mathrm{s}^{-1}$",
            r"$L$ / $\mathrm{GeV}~\mathrm{s}^{-1}$",
            r"$f_\mathrm{arr}$",
            r"$f_\mathrm{arr}(\mathrm{astro})$",
            "src_index",
            "diff_index",
        ]

    @classmethod
    def initialise_env(
        cls,
        output_dir,
    ):
        """
        Script to set up enviroment for parallel
        model checking runs.

        * Runs MCEq for atmo flux if needed
        * Generates and compiles necessary Stan files
        Only need to run once before calling ModelCheck(...).run()
        """

        # Config
        parameter_config = hnu_config["parameter_config"]
        file_config = hnu_config["file_config"]

        # Run MCEq computation
        print("Setting up MCEq run for AtmopshericNumuFlux")
        Emin = parameter_config["Emin"] * u.GeV
        Emax = parameter_config["Emax"] * u.GeV
        atmo_flux_model = AtmosphericNuMuFlux(Emin, Emax)

        # Build necessary details to define simulation and fit code
        sources = _initialise_sources()
        detector_model_type = ModelCheck._get_dm_from_config(
            parameter_config["detector_model_type"]
        )

        # Generate atmo and main sim Stan files
        main_sim_name = file_config["main_sim_filename"][:-5]
        atmo_sim_name = file_config["atmo_sim_filename"][:-5]
        stan_sim_interface = StanSimInterface(
            main_sim_name,
            sources,
            detector_model_type,
            atmo_output_file=atmo_sim_name,
        )

        stan_sim_interface.generate_atmo()
        print("Generated atmo_sim Stan file at:", atmo_sim_name)

        stan_sim_interface.generate()
        print("Generated sim_code Stan file at:", main_sim_name)

        # Generate fit Stan file
        fit_name = file_config["fit_filename"][:-5]
        stan_fit_interface = StanFitInterface(
            fit_name,
            sources,
            detector_model_type,
        )

        stan_fit_interface.generate()
        print("Generated fit Stan file at:", fit_name)

        # Comilation of Stan models
        print("Compile Stan models")
        stanc_options = {"include-paths": list(file_config["include_paths"])}

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
        self.results["src_index"] = []
        self.results["diff_index"] = []
        self.results["f_arr"] = []
        self.results["f_arr_astro"] = []

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
                    self.results["src_index"].extend(job_folder["src_index"][()])
                    self.results["diff_index"].extend(job_folder["diff_index"][()])
                    self.results["f_arr"].extend(job_folder["f_arr"][()])
                    self.results["f_arr_astro"].extend(job_folder["f_arr_astro"][()])

    def compare(self, var_names=None, var_labels=None, show_prior=False):

        if not var_names:
            var_names = self._default_var_names

        if not var_labels:
            var_labels = self._default_var_labels

        N = len(var_names)
        fig, ax = plt.subplots(N, figsize=(5, 15))

        for v, var_name in enumerate(var_names):
            bins = np.linspace(
                np.min(self.results[var_name]), np.max(self.results[var_name]), 35
            )

            for i in range(len(self.results[var_name])):
                ax[v].hist(
                    self.results[var_name][i],
                    color="#017B76",
                    alpha=0.05,
                    histtype="step",
                    bins=bins,
                    lw=1.0,
                    density=True,
                )

            if show_prior:

                prior_func = self._get_prior_func(var_name)
                xmin, xmax = ax[v].get_xlim()
                x = np.linspace(xmin, xmax)
                ax[v].plot(x, prior_func(x), lw=1, color="k", alpha=0.5)

            ax[v].axvline(self.truths[var_name], color="k", linestyle="-")
            ax[v].set_xlabel(var_labels[v], labelpad=10)

        fig.tight_layout()
        return fig, ax

    def _single_run(self, n_subjobs, seed):
        """
        Single run to be called using Parallel.
        """

        sys.stderr.write("Random seed: %i\n" % seed)

        self._sources = _initialise_sources()

        file_config = hnu_config["file_config"]
        parameter_config = hnu_config["parameter_config"]

        detector_model_type = ModelCheck._get_dm_from_config(
            parameter_config["detector_model_type"]
        )

        subjob_seeds = [(seed + subjob) * 10 for subjob in range(n_subjobs)]

        start_time = time.time()

        outputs = {}
        outputs["F_diff"] = []
        outputs["F_atmo"] = []
        outputs["L"] = []
        outputs["f_arr"] = []
        outputs["f_arr_astro"] = []
        outputs["src_index"] = []
        outputs["diff_index"] = []

        for i, s in enumerate(subjob_seeds):

            sys.stderr.write("Run %i\n" % i)

            # Simulation
            obs_time = parameter_config["obs_time"] * u.year
            sim = Simulation(self._sources, detector_model_type, obs_time)
            sim.precomputation()
            sim.set_stan_filenames(
                file_config["atmo_sim_filename"], file_config["main_sim_filename"]
            )
            sim.compile_stan_code(include_paths=list(file_config["include_paths"]))
            sim.run(seed=s)
            self.sim = sim

            lam = sim._sim_output.stan_variable("Lambda")[0]
            sim_output = {}
            sim_output["Lambda"] = lam

            self.events = sim.events

            # Fit
            fit = StanFit(self._sources, detector_model_type, sim.events, obs_time)
            fit.precomputation(exposure_integral=sim._exposure_integral)
            fit.set_stan_filename(file_config["fit_filename"])
            fit.compile_stan_code(include_paths=list(file_config["include_paths"]))
            fit.run(seed=s)

            self.fit = fit

            # Store output
            outputs["F_diff"].append(fit._fit_output.stan_variable("F_diff"))
            outputs["F_atmo"].append(fit._fit_output.stan_variable("F_atmo"))
            outputs["L"].append(fit._fit_output.stan_variable("L"))
            outputs["f_arr"].append(fit._fit_output.stan_variable("f_arr"))
            outputs["f_arr_astro"].append(fit._fit_output.stan_variable("f_arr_astro"))
            outputs["src_index"].append(fit._fit_output.stan_variable("src_index"))
            outputs["diff_index"].append(fit._fit_output.stan_variable("diff_index"))

            fit.check_classification(sim_output)

            sys.stderr.write("time: %.5f\n" % (time.time() - start_time))

        return outputs

    @staticmethod
    def _get_dm_from_config(dm_key):

        if dm_key == "northern_tracks":

            dm = NorthernTracksDetectorModel

        elif dm_key == "cascades":

            dm = CascadesDetectorModel

        elif dm_key == "icecube":

            dm = IceCubeDetectorModel

        else:

            raise ValueError("Detector model key in config not recognised")

        return dm

    def _get_prior_samples(self, var_name):
        """
        Return prior samples for the parameter "var_name".
        """

        N = len(self.results[var_name][0])

        if var_name == "F_diff":

            F_diff_scale = self.truths["F_diff"]

            return lognorm(5, 0, F_diff_scale).rvs(N)

        elif var_name == "L":

            L_scale = self.truths["L"]

            return lognorm(5, 0, L_scale).rvs(N)

        elif var_name == "F_atmo":

            F_atmo_scale = self.truths["F_atmo"]

            return norm(F_atmo_scale, 0.1 * F_atmo_scale).rvs(N)

        elif var_name == "f_arr" or var_name == "f_arr_astro":

            return uniform(0, 1).rvs(N)

        elif var_name == "src_index":

            return norm(2, 2).rvs(N)

        elif var_name == "diff_index":

            return norm(2, 2).rvs(N)

        elif var_name == "F_tot":

            F_tot_scale = self.truths["F_tot"]

            return norm(F_tot_scale, 0.5 * F_tot_scale).rvs(N)

        else:

            raise ValueError("var_name not recognised")

    def _get_prior_func(self, var_name):
        """
        Return function of param "var_name" that
        describes its prior.
        """

        if var_name == "F_diff":

            def prior_func(F_diff):
                F_diff_scale = self.truths["F_diff"]
                return norm(F_diff_scale, 2 * F_diff_scale).pdf(F_diff)

        elif var_name == "L":

            def prior_func(L):
                L_scale = self.truths["L"]
                return norm(L_scale, 2 * L_scale).pdf(L)

        elif var_name == "F_atmo":

            def prior_func(F_atmo):
                F_atmo_scale = self.truths["F_atmo"]
                return norm(F_atmo_scale, 0.1 * F_atmo_scale).pdf(F_atmo)

        elif var_name == "f_arr" or var_name == "f_arr_astro":

            def prior_func(f):
                return uniform(0, 1).pdf(f)

        elif var_name == "F_tot":

            def prior_func(F_tot):
                F_tot_scale = self.truths["F_tot"]
                return norm(F_tot_scale, 0.5 * F_tot_scale)

        elif var_name == "src_index":

            def prior_func(src_index):
                return norm(2, 2).pdf(src_index)

        elif var_name == "diff_index":

            def prior_func(diff_index):
                return norm(2, 2).pdf(diff_index)

        else:

            raise ValueError("var_name not recognised")

        return prior_func


def _initialise_sources():

    parameter_config = hnu_config["parameter_config"]

    Parameter.clear_registry()
    src_index = Parameter(
        parameter_config["src_index"],
        "src_index",
        fixed=False,
        par_range=parameter_config["src_index_range"],
    )
    diff_index = Parameter(
        parameter_config["diff_index"],
        "diff_index",
        fixed=False,
        par_range=parameter_config["diff_index_range"],
    )
    L = Parameter(
        parameter_config["L"] * u.erg / u.s,
        "luminosity",
        fixed=True,
        par_range=parameter_config["L_range"] * u.erg / u.s,
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

    if parameter_config["Emin_det_eq"]:

        Emin_det = Parameter(
            parameter_config["Emin_det"] * u.GeV, "Emin_det", fixed=True
        )

    else:

        Emin_det_tracks = Parameter(
            parameter_config["Emin_det_tracks"] * u.GeV,
            "Emin_det_tracks",
            fixed=True,
        )
        Emin_det_cascades = Parameter(
            parameter_config["Emin_det_cascades"] * u.GeV,
            "Emin_det_cascades",
            fixed=True,
        )

    # Simple point source for testing
    point_source = PointSource.make_powerlaw_source(
        "test", np.deg2rad(5) * u.rad, np.pi * u.rad, L, src_index, 0.43, Emin, Emax
    )

    sources = Sources()
    sources.add(point_source)
    sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index)
    sources.add_atmospheric_component()

    return sources
