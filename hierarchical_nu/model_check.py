import os
import sys
import numpy as np
import h5py
import time
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from astropy import units as u
from astropy.coordinates import SkyCoord
from cmdstanpy import CmdStanModel
from scipy.stats import uniform
from typing import List, Union

from hierarchical_nu.source.atmospheric_flux import AtmosphericNuMuFlux
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import PointSource, Sources

from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.detector.r2021 import R2021DetectorModel

from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.utils.config import hnu_config
from hierarchical_nu.utils.roi import ROI, CircularROI, RectangularROI
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior


class ModelCheck:
    """
    Check statistical model by repeatedly
    fitting simulated data using different random seeds.
    """

    def __init__(self, truths=None, priors=None):
        if priors:
            self.priors = priors
            print("Found priors")

        else:
            print("Loading priors from config")
            prior_config = hnu_config["prior_config"]
            priors = Priors()
            priors.luminosity = self.make_prior(prior_config["L"])
            priors.src_index = self.make_prior(prior_config["src_index"])
            priors.atmospheric_flux = self.make_prior(prior_config["atmo_flux"])
            priors.diffuse_flux = self.make_prior(prior_config["diff_flux"])
            priors.diff_index = self.make_prior(prior_config["diff_index"])
            self.priors = priors

        if truths:
            print("Found true values")
            self.truths = truths

        else:
            print("Loading true values from config")
            # Config
            file_config = hnu_config["file_config"]
            parameter_config = hnu_config["parameter_config"]

            # Sources
            self._sources = _initialise_sources()
            f_arr = self._sources.f_arr().value
            f_arr_astro = self._sources.f_arr_astro().value

            # Detector
            self._detector_model_type = ModelCheck._get_dm_from_config(
                parameter_config["detector_model_type"]
            )

            self._obs_time = parameter_config["obs_time"] * u.year
            # self._nshards = parameter_config["nshards"]
            self._threads_per_chain = parameter_config["threads_per_chain"]

            sim = Simulation(self._sources, self._detector_model_type, self._obs_time)
            sim.precomputation()
            self._exposure_integral = sim._exposure_integral
            # sim.set_stan_filename(file_config["sim_filename"])
            # sim.compile_stan_code(include_paths=list(file_config["include_paths"]))
            sim_inputs = sim._get_sim_inputs()
            Nex = sim._get_expected_Nnu(sim_inputs)
            Nex_per_comp = sim._expected_Nnu_per_comp

            # Truths
            self.truths = {}

            flux_unit = 1 / (u.m**2 * u.s)

            diffuse_bg = self._sources.diffuse
            self.truths["F_diff"] = diffuse_bg.flux_model.total_flux_int.to(
                flux_unit
            ).value
            atmo_bg = self._sources.atmospheric
            self.truths["F_atmo"] = atmo_bg.flux_model.total_flux_int.to(
                flux_unit
            ).value
            self.truths["L"] = (
                Parameter.get_parameter("luminosity").value.to(u.GeV / u.s).value
            )
            self.truths["f_arr"] = f_arr
            self.truths["f_arr_astro"] = f_arr_astro
            self.truths["src_index"] = Parameter.get_parameter("src_index").value
            self.truths["diff_index"] = Parameter.get_parameter("diff_index").value

            self.truths["Nex"] = Nex
            self.truths["Nex_src"] = Nex_per_comp[0]
            self.truths["Nex_diff"] = Nex_per_comp[1]
            self.truths["Nex_atmo"] = Nex_per_comp[2]
            self.truths["f_det"] = Nex_per_comp[0] / Nex
            self.truths["f_det_astro"] = Nex_per_comp[0] / sum(Nex_per_comp[0:2])

        self._default_var_names = [key for key in self.truths]
        self._default_var_names.append("Fs")

        self._diagnostic_names = ["lp__", "divergent__", "treedepth__", "energy__"]

    @staticmethod
    def make_prior(p):
        if p["name"] == "LogNormalPrior":
            prior = LogNormalPrior(mu=np.log(p["mu"]), sigma=p["sigma"])
        elif p["name"] == "NormalPrior":
            prior = NormalPrior(mu=p["mu"], sigma=p["sigma"])
        else:
            raise ValueError("Currently no other prior implemented")
        return prior

    @staticmethod
    def initialise_env(output_dir, priors: Priors = Priors()):
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
        prior_config = hnu_config["prior_config"]

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

        # Generate sim Stan file
        sim_name = file_config["sim_filename"][:-5]
        stan_sim_interface = StanSimInterface(
            sim_name,
            sources,
            detector_model_type,
        )

        stan_sim_interface.generate()
        print("Generated sim_code Stan file at:", sim_name)

        # Generate fit Stan file
        nshards = parameter_config["nshards"]
        fit_name = file_config["fit_filename"][:-5]

        priors = Priors()
        priors.luminosity = ModelCheck.make_prior(prior_config["L"])
        priors.src_index = ModelCheck.make_prior(prior_config["src_index"])
        priors.atmospheric_flux = ModelCheck.make_prior(prior_config["atmo_flux"])
        priors.diffuse_flux = ModelCheck.make_prior(prior_config["diff_flux"])
        priors.diff_index = ModelCheck.make_prior(prior_config["diff_index"])

        stan_fit_interface = StanFitInterface(
            fit_name,
            sources,
            detector_model_type,
            nshards=nshards,
            priors=priors,
        )

        stan_fit_interface.generate()
        print("Generated fit Stan file at:", fit_name)

        # Comilation of Stan models
        print("Compile Stan models")
        stanc_options = {"include-paths": list(file_config["include_paths"])}
        cpp_options = None

        if nshards not in [0, 1]:
            cpp_options = {"STAN_THREADS": True}

        _ = CmdStanModel(
            stan_file=file_config["sim_filename"],
            stanc_options=stanc_options,
        )
        _ = CmdStanModel(
            stan_file=file_config["fit_filename"],
            stanc_options=stanc_options,
            cpp_options=cpp_options,
        )

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def parallel_run(self, n_jobs=1, n_subjobs=1, seed=None, **kwargs):
        job_seeds = [(seed + job) * 10 for job in range(n_jobs)]

        self._results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._single_run)(n_subjobs, seed=s, **kwargs) for s in job_seeds
        )

    def save(self, filename):
        with h5py.File(filename, "w") as f:
            truths_folder = f.create_group("truths")
            for key, value in self.truths.items():
                truths_folder.create_dataset(key, data=value)

            sim_folder = f.create_group("sim")

            for i, res in enumerate(self._results):
                folder = f.create_group("results_%i" % i)

                for key, value in res.items():
                    if key != "Lambda":
                        folder.create_dataset(key, data=value)
                    else:
                        sim_folder.create_dataset("sim_%i" % i, data=np.vstack(value))

        self.priors.addto(filename, "priors")

    @classmethod
    def load(cls, filename_list):
        with h5py.File(filename_list[0], "r") as f:
            job_folder = f["results_0"]
            _result_names = [key for key in job_folder]

        truths = {}
        priors = None

        results = {}
        for key in _result_names:
            results[key] = []

        file_truths = {}
        for filename in filename_list:
            file_priors = Priors.from_group(filename, "priors")

            if not priors:
                priors = file_priors

            with h5py.File(filename, "r") as f:
                truths_folder = f["truths"]
                sim_folder = f["sim"]
                for key, value in truths_folder.items():
                    file_truths[key] = value[()]

                if not truths:
                    truths = file_truths

                if truths != file_truths:
                    raise ValueError(
                        "Files in list have different truth settings and should not be combined"
                    )

                n_jobs = len(
                    [
                        key
                        for key in f
                        if (key != "truths" and key != "priors" and key != "sim")
                    ]
                )
                sim = {}
                for i in range(n_jobs):
                    job_folder = f["results_%i" % i]
                    for res_key in job_folder:
                        results[res_key].extend(job_folder[res_key][()])
                    sim["sim_%i_Lambda" % i] = sim_folder["sim_%i" % i][()]

        output = cls(truths, priors)
        output.results = results
        output.sim_Lambdas = sim

        return output

    def compare(
        self,
        var_names: Union[None, List[str]] = None,
        var_labels: Union[None, List[str]] = None,
        show_prior: bool = False,
        nbins: int = 50,
        mask_results: Union[None, np.ndarray] = None,
        alpha=0.1,
        show_N: bool = False,
    ):
        if not var_names:
            var_names = self._default_var_names

        if not var_labels:
            var_labels = self._default_var_names

        mask = np.tile(False, len(self.results[var_names[0]]))
        if mask_results is not None:
            mask[mask_results] = True
        else:
            mask_results = np.array([])

        N = len(var_names)
        fig, ax = plt.subplots(N, figsize=(5, N * 3))

        for v, var_name in enumerate(var_names):
            if (
                var_name == "L"
                or var_name == "F_diff"
                or var_name == "F_atmo"
                or var_name == "Fs"
            ):
                bins = np.geomspace(
                    np.min(np.array(self.results[var_name])[~mask]),
                    np.max(np.array(self.results[var_name])[~mask]),
                    nbins,
                )
                prior_supp = np.geomspace(
                    np.min(np.array(self.results[var_name])[~mask]),
                    np.max(np.array(self.results[var_name])[~mask]),
                    1000,
                )
                ax[v].set_xscale("log")

            else:
                prior_supp = np.linspace(
                    np.min(self.results[var_name]),
                    np.max(self.results[var_name]),
                    1000,
                )
                bins = np.linspace(
                    np.min(np.array(self.results[var_name])[~mask]),
                    np.max(np.array(self.results[var_name])[~mask]),
                    nbins,
                )
            max_value = 0

            for i in range(len(self.results[var_name])):
                if i not in mask_results and len(self.results[var_name]) != 0:
                    n, bins, _ = ax[v].hist(
                        self.results[var_name][i],
                        color="#017B76",
                        alpha=alpha,
                        histtype="step",
                        bins=bins,
                        lw=1.0,
                    )

                    if show_N and var_name == "Nex_src":
                        for val in self.sim_Lambdas.values():
                            # Overplot the actual number of events for each component
                            for line in val:
                                ax[v].axvline(line[0], ls="-", c="red", lw=0.3)

                    elif show_N and var_name == "Nex_diff":
                        for val in self.sim_Lambdas.values():
                            # Overplot the actual number of events for each component
                            for line in val:
                                ax[v].axvline(line[1], ls="-", c="red", lw=0.3)

                    elif show_N and var_name == "Nex_atmo":
                        for val in self.sim_Lambdas.values():
                            # Overplot the actual number of events for each component
                            for line in val:
                                ax[v].axvline(line[2], ls="-", c="red", lw=0.3)

                    max_value = n.max() if n.max() > max_value else max_value

            if show_prior:
                N = len(self.results[var_name][0]) * 100
                xmin, xmax = ax[v].get_xlim()

                if "f_" in var_name and not "diff" in var_name:  # yikes
                    prior_samples = uniform(0, 1).rvs(N)
                    prior_density = uniform(0, 1).pdf(prior_supp)
                    plot = True

                elif "Nex" in var_name:
                    plot = False

                elif "index" in var_name:
                    prior_density = self.priors.to_dict()[var_name].pdf(prior_supp)
                    plot = True

                elif not "Fs" in var_name:
                    prior_samples = self.priors.to_dict()[var_name].sample(N)
                    prior_density = self.priors.to_dict()[var_name].pdf_logspace(
                        prior_supp
                    )
                    plot = True
                else:
                    plot = False

                if plot:
                    ax[v].plot(
                        prior_supp,
                        prior_density / prior_density.max() * max_value / 2.0,
                        color="k",
                        alpha=0.5,
                        lw=2,
                        label="Prior",
                    )
                    ax[v].hist(
                        prior_samples,
                        color="k",
                        alpha=0.5,
                        histtype="step",
                        bins=bins,
                        lw=2,
                        weights=np.tile(0.01, len(prior_samples)),
                        label="Prior samples",
                    )
            try:
                ax[v].axvline(
                    self.truths[var_name], color="k", linestyle="-", label="Truth"
                )
            except KeyError:
                pass
            ax[v].set_xlabel(var_labels[v], labelpad=10)
            ax[v].legend(loc="best")

        fig.tight_layout()
        return fig, ax

    def diagnose(self):
        """
        Quickly check output of CmdStanMCMC.diagnose().
        Return index of fits with issues.
        """

        diagnostics_array = np.array(self.results["diagnostics_ok"])

        ind_not_ok = np.where(diagnostics_array == 0)[0]

        if len(ind_not_ok) > 0:
            print(
                "%.2f percent of fits have issues!"
                % (len(ind_not_ok) / len(diagnostics_array) * 100)
            )

        else:
            print("No issues found")

        return ind_not_ok

    def _single_run(self, n_subjobs, seed, **kwargs):
        """
        Single run to be called using Parallel.
        """

        sys.stderr.write("Random seed: %i\n" % seed)

        self._sources = _initialise_sources()

        file_config = hnu_config["file_config"]

        subjob_seeds = [(seed + subjob) * 10 for subjob in range(n_subjobs)]

        outputs = {}
        for key in self._default_var_names:
            outputs[key] = []
        for key in self._diagnostic_names:
            outputs[key] = []

        outputs["diagnostics_ok"] = []
        outputs["run_time"] = []
        outputs["Lambda"] = []

        for i, s in enumerate(subjob_seeds):
            sys.stderr.write("Run %i\n" % i)

            # Simulation
            # Should reduce time consumption if only on first iteration model is compiled
            if i == 0:
                sim = Simulation(
                    self._sources, self._detector_model_type, self._obs_time
                )
                sim.precomputation(self._exposure_integral)
                # sim.set_stan_filename(file_config["sim_filename"])
                # sim.compile_stan_code(include_paths=list(file_config["include_paths"]))
                sim.setup_stan_sim(os.path.splitext(file_config["sim_filename"])[0])
            sim.run(seed=s, verbose=True)
            self.sim = sim

            lambd = sim._sim_output.stan_variable("Lambda").squeeze()
            ps = np.sum(lambd == 1.0)
            diff = np.sum(lambd == 2.0)
            atmo = np.sum(lambd == 3.0)
            lam = np.array([ps, diff, atmo])
            # sim_output = {}
            # sim_output["Lambda"] = lam

            # Skip if no detected events
            if not sim.events:
                continue

            self.events = sim.events

            # Fit
            # Same as above, save time
            if i == 0:
                fit = StanFit(
                    self._sources,
                    self._detector_model_type,
                    sim.events,
                    self._obs_time,
                    priors=self.priors,
                    nshards=self._threads_per_chain,
                )
                fit.precomputation()
                # fit.set_stan_filename(file_config["fit_filename"])
                # fit.compile_stan_code(include_paths=list(file_config["include_paths"]))
                fit.setup_stan_fit(os.path.splitext(file_config["fit_filename"])[0])

            else:
                fit.events = sim.events

            start_time = time.time()
            fit.run(
                seed=s,
                show_progress=True,
                threads_per_chain=self._threads_per_chain,
                inits={"src_index": 2.2, "L": 1e50, "diff_index": 2.2},
                **kwargs
            )

            self.fit = fit

            # Store output
            run_time = time.time() - start_time
            sys.stderr.write("time: %.5f\n" % run_time)
            outputs["run_time"].append(run_time)
            outputs["Lambda"].append(lam)

            for key in self._default_var_names:
                outputs[key].append(fit._fit_output.stan_variable(key))

            for key in self._diagnostic_names:
                outputs[key].append(fit._fit_output.method_variables()[key])

            diagnostics_output_str = fit._fit_output.diagnose()

            if "no problems detected" in diagnostics_output_str:
                outputs["diagnostics_ok"].append(1)
            else:
                outputs["diagnostics_ok"].append(0)

        return outputs

    @staticmethod
    def _get_dm_from_config(dm_key):
        if dm_key == "northern_tracks":
            dm = NorthernTracksDetectorModel

        elif dm_key == "cascades":
            dm = CascadesDetectorModel

        elif dm_key == "icecube":
            dm = IceCubeDetectorModel

        elif dm_key == "r2021":
            dm = R2021DetectorModel

        else:
            raise ValueError("Detector model key in config not recognised")

        return dm

    def _get_prior_func(self, var_name):
        """
        Return function of param "var_name" that
        describes its prior.
        """

        try:
            prior = self.priors.to_dict()[var_name]

            prior_func = prior.pdf

        except:
            if var_name == "f_arr" or var_name == "f_arr_astro":

                def prior_func(f):
                    return uniform(0, 1).pdf(f)

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
        parameter_config["diff_norm"] * 1 / (u.GeV * u.m**2 * u.s),
        "diffuse_norm",
        fixed=True,
        par_range=(0, np.inf),
    )
    Enorm = Parameter(parameter_config["Enorm"] * u.GeV, "Enorm", fixed=True)
    Emin = Parameter(parameter_config["Emin"] * u.GeV, "Emin", fixed=True)
    Emax = Parameter(parameter_config["Emax"] * u.GeV, "Emax", fixed=True)

    Emin_src = Parameter(parameter_config["Emin_src"] * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(parameter_config["Emax_src"] * u.GeV, "Emax_src", fixed=True)

    Emin_diff = Parameter(
        parameter_config["Emin_diff"] * u.GeV, "Emin_diff", fixed=True
    )
    Emax_diff = Parameter(
        parameter_config["Emax_diff"] * u.GeV, "Emax_diff", fixed=True
    )

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
    dec = np.deg2rad(parameter_config["src_dec"]) * u.rad
    ra = np.deg2rad(parameter_config["src_ra"]) * u.rad
    center = SkyCoord(ra=ra, dec=dec, frame="icrs")
    radius = 10 * u.deg
    roi = CircularROI(center, radius)
    point_source = PointSource.make_powerlaw_source(
        "test",
        dec,
        ra,
        L,
        src_index,
        parameter_config["z"],
        Emin_src,
        Emax_src,
    )

    sources = Sources()
    sources.add(point_source)
    sources.add_diffuse_component(
        diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff, 0.0
    )
    sources.add_atmospheric_component()

    return sources
