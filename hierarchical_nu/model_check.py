import os
import sys
import numpy as np
import h5py
import arviz as av
import time
from matplotlib import pyplot as plt
import matplotlib.patches as mpl_patches
from joblib import Parallel, delayed
from astropy import units as u
from astropy.coordinates import SkyCoord
from cmdstanpy import CmdStanModel
from scipy.stats import uniform
from typing import List, Union

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import PointSource, Sources

from hierarchical_nu.detector.icecube import Refrigerator
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.stan.interface import STAN_GEN_PATH
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.utils.config import hnu_config
from hierarchical_nu.utils.roi import (
    CircularROI,
    RectangularROI,
    FullSkyROI,
    NorthernSkyROI,
    ROIList,
)
from hierarchical_nu.priors import (
    Priors,
    LogNormalPrior,
    NormalPrior,
    ParetoPrior,
    LuminosityPrior,
    IndexPrior,
    FluxPrior,
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelCheck:
    """
    Check statistical model by repeatedly
    fitting simulated data using different random seeds.
    """

    def __init__(self, truths=None, priors=None):
        if priors:
            self.priors = priors
            logger.info("Found priors")

        else:
            logger.info("Loading priors from config")
            prior_config = hnu_config["prior_config"]
            priors = self.make_priors(prior_config)
            self.priors = priors

        if truths:
            logger.info("Found true values")
            self.truths = truths

        else:
            logger.info("Loading true values from config")
            # Config
            file_config = hnu_config["file_config"]
            parameter_config = hnu_config["parameter_config"]
            _make_roi()
            share_L = hnu_config.parameter_config.share_L
            share_src_index = hnu_config.parameter_config.share_src_index

            asimov = parameter_config.asimov

            # Sources
            self._sources = _initialise_sources()
            f_arr = self._sources.f_arr().value
            f_arr_astro = self._sources.f_arr_astro().value

            # Detector
            self._detector_model_type = ModelCheck._get_dm_from_config(
                parameter_config["detector_model_type"]
            )
            self._obs_time = ModelCheck._get_obs_time_from_config(
                self._detector_model_type, parameter_config["obs_time"]
            )
            # self._nshards = parameter_config["nshards"]
            self._threads_per_chain = parameter_config["threads_per_chain"]

            if asimov:
                N = {}
                for dm in self._obs_time.keys():
                    N[dm] = [1] * self._sources.N
                sim = Simulation(
                    self._sources, self._detector_model_type, self._obs_time, N=N
                )
            else:
                sim = Simulation(
                    self._sources, self._detector_model_type, self._obs_time
                )
            self.sim = sim
            sim.precomputation()
            sim_inputs = sim._get_sim_inputs()
            Nex = sim._get_expected_Nnu(sim_inputs)
            Nex_per_comp = sim._expected_Nnu_per_comp
            self._Nex_et = sim._Nex_et
            if asimov:
                N = {}
                for c, dm in enumerate(self._obs_time.keys()):
                    N[dm] = np.rint(self._Nex_et[c]).astype(int).tolist()
                self._N = N

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
            try:
                self.truths["L"] = [
                    Parameter.get_parameter("luminosity").value.to(u.GeV / u.s).value
                ]
            except ValueError:
                self.truths["L"] = [
                    Parameter.get_parameter(f"ps_{_}_luminosity")
                    .value.to(u.GeV / u.s)
                    .value
                    for _ in range(len(self._sources.point_source))
                ]
            self.truths["f_arr"] = f_arr
            self.truths["f_arr_astro"] = f_arr_astro
            try:
                self.truths["src_index"] = [Parameter.get_parameter("src_index").value]
            except ValueError:
                self.truths["src_index"] = [
                    Parameter.get_parameter(f"ps_{_}_src_index").value
                    for _ in range(len(self._sources.point_source))
                ]
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
    def make_priors(prior_config):
        """
        Make priors from config file.
        Assumes default units specified in `hierarchical_nu.priors` for each quantity.
        """
        priors = Priors()

        for p, vals in prior_config.items():
            if vals.name == "NormalPrior":
                prior = NormalPrior
                mu = vals.mu
                sigma = vals.sigma
            elif vals.name == "LogNormalPrior":
                prior = LogNormalPrior
                mu = vals.mu
                sigma = vals.sigma
            elif vals.name == "ParetoPrior":
                prior = ParetoPrior
            else:
                raise NotImplementedError("Prior type not recognised.")

            if p == "src_index":
                priors.src_index = IndexPrior(prior, mu=mu, sigma=sigma)
            elif p == "diff_index":
                priors.diff_index = IndexPrior(prior, mu=mu, sigma=sigma)
            elif p == "L":
                if prior == NormalPrior:
                    priors.luminosity = LuminosityPrior(
                        prior,
                        mu=mu * LuminosityPrior.UNITS,
                        sigma=sigma * LuminosityPrior.UNITS,
                    )
                elif prior == LogNormalPrior:
                    priors.luminosity = LuminosityPrior(
                        prior, mu=mu * LuminosityPrior.UNITS, sigma=sigma
                    )
                elif prior == ParetoPrior:
                    raise NotImplementedError("Prior not recognised.")
                    # priors.luminosity = LuminosityPrior(prior,
            elif p == "diff_flux":
                if prior == NormalPrior:
                    priors.diffuse_flux = FluxPrior(
                        prior, mu=mu * FluxPrior.UNITS, sigma=sigma * FluxPrior.UNITS
                    )
                elif prior == LogNormalPrior:
                    priors.diffuse_flux = FluxPrior(
                        prior, mu=mu * FluxPrior.UNITS, sigma=sigma
                    )
                else:
                    raise NotImplementedError("Prior not recognised.")

            elif p == "atmo_flux":
                if prior == NormalPrior:
                    priors.atmospheric_flux = FluxPrior(
                        prior, mu=mu * FluxPrior.UNITS, sigma=sigma * FluxPrior.UNITS
                    )
                elif prior == LogNormalPrior:
                    priors.atmospheric = FluxPrior(
                        prior, mu=mu * FluxPrior.UNITS, sigma=sigma
                    )
                else:
                    raise NotImplementedError("Prior not recognised.")

        return priors

    @staticmethod
    def initialise_env(output_dir):
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

        asimov = parameter_config.asimov

        _make_roi()

        if not STAN_GEN_PATH in file_config["include_paths"]:
            file_config["include_paths"].append(STAN_GEN_PATH)

        # Run MCEq computation
        logger.info("Setting up MCEq run for AtmopshericNumuFlux")

        # Build necessary details to define simulation and fit code
        detector_model_type = ModelCheck._get_dm_from_config(
            parameter_config["detector_model_type"]
        )
        sources = _initialise_sources()
        # Generate sim Stan file
        sim_name = file_config["sim_filename"][:-5]
        stan_sim_interface = StanSimInterface(
            sim_name, sources, detector_model_type, force_N=asimov
        )

        stan_sim_interface.generate()
        logger.info(f"Generated sim_code Stan file at: {sim_name}")

        # Generate fit Stan file
        threads_per_chain = parameter_config["threads_per_chain"]
        nshards = threads_per_chain
        fit_name = file_config["fit_filename"][:-5]

        priors = ModelCheck.make_priors(prior_config)

        stan_fit_interface = StanFitInterface(
            fit_name,
            sources,
            detector_model_type,
            nshards=nshards,
            priors=priors,
        )

        stan_fit_interface.generate()
        logger.info(f"Generated fit Stan file at: {fit_name}")

        # Comilation of Stan models
        logger.info("Compile Stan models")
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

    def save(self, filename, save_events: bool = False):
        with h5py.File(filename, "w") as f:
            truths_folder = f.create_group("truths")
            for key, value in self.truths.items():
                truths_folder.create_dataset(key, data=value)

            sim_folder = f.create_group("sim")

            for i, res in enumerate(self._results):
                folder = f.create_group("results_%i" % i)

                for key, value in res.items():
                    if key == "events":
                        continue
                    elif key == "association_prob":
                        if not save_events:
                            continue
                        for c, prob in enumerate(value):
                            folder.create_dataset(f"association_prob_{c}", data=prob)
                    elif key != "Lambda" and key != "event_Lambda":
                        folder.create_dataset(key, data=value)
                    elif key == "Lambda":
                        sim_folder.create_dataset("sim_%i" % i, data=np.vstack(value))
                    elif key == "event_Lambda":
                        for c, data in enumerate(res["event_Lambda"]):
                            sim_folder.create_dataset(
                                f"event_Lambda_{i}_{c}", data=data
                            )

        if save_events and "events" in res.keys():
            for i, res in enumerate(self._results):
                for c, events in enumerate(res["events"]):
                    events.to_file(
                        filename, append=True, group_name=f"sim/events_{i}_{c}"
                    )

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

        sim = {}
        sim_N = []

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

                for i in range(n_jobs):
                    job_folder = f["results_%i" % i]
                    for res_key in job_folder:
                        results[res_key].extend(job_folder[res_key][()])
                    # sim["sim_%i_Lambda" % i] = sim_folder["sim_%i" % i][()]
                    sim_N.extend(sim_folder["sim_%i" % i][()])

        output = cls(truths, priors)
        output.results = results
        output.sim_Lambdas = sim
        output.sim_N = sim_N

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
                # or var_name == "F_atmo"
                or var_name == "Fs"
            ):
                log = True
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
                log = False
                prior_supp = np.linspace(
                    np.min(np.array(self.results[var_name])[~mask]),
                    np.max(np.array(self.results[var_name])[~mask]),
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
                    if log:
                        n, _ = np.histogram(
                            np.log10(self.results[var_name][i]),
                            np.log10(bins),
                            density=True,
                        )
                    else:
                        n, _ = np.histogram(
                            self.results[var_name][i], bins, density=True
                        )
                    n = np.hstack((np.array([0]), n, np.array([n[-1], 0])))
                    plot_bins = np.hstack(
                        (np.array([bins[0] - 1e-10]), bins, np.array([bins[-1]]))
                    )

                    low = np.nonzero(n)[0].min()
                    high = np.nonzero(n)[0].max()
                    ax[v].step(
                        plot_bins[low - 1 : high + 2],
                        n[low - 1 : high + 2],
                        color="#017B76",
                        alpha=alpha,
                        lw=1.0,
                        where="post",
                    )

                    if show_N and var_name == "Nex_src":
                        for val in self.sim_N:
                            # Overplot the actual number of events for each component
                            # for line in val:
                            ax[v].axvline(val[0], ls="-", c="red", lw=0.3)

                    elif show_N and var_name == "Nex_diff":
                        for val in self.sim_N:
                            # Overplot the actual number of events for each component
                            # for line in val:
                            ax[v].axvline(val[1], ls="-", c="red", lw=0.3)

                    elif show_N and var_name == "Nex_atmo":
                        for val in self.sim_N:
                            # Overplot the actual number of events for each component
                            # for line in val:
                            ax[v].axvline(val[2], ls="-", c="red", lw=0.3)

                    max_value = n.max() if n.max() > max_value else max_value

            if show_prior:
                N = len(self.results[var_name][0]) * 100

                # this case distinction is overly complicated
                if (
                    var_name == "L"
                    or var_name == "F_diff"
                    or var_name == "F_atmo"
                    or var_name == "Fs"
                ):
                    pass

                if "f_" in var_name and not "diff" in var_name:  # yikes
                    plot = False

                elif "Nex" in var_name:
                    plot = False

                elif "index" in var_name:
                    prior_density = self.priors.to_dict()[var_name].pdf(
                        prior_supp * self.priors.to_dict()[var_name].UNITS
                    )
                    plot = True

                elif not "Fs" in var_name:
                    # prior_samples = self.priors.to_dict()[var_name].sample(N)
                    prior_density = self.priors.to_dict()[var_name].pdf_logspace(
                        prior_supp * self.priors.to_dict()[var_name].UNITS
                    )
                    plot = True
                else:
                    plot = False

                if plot:
                    ax[v].plot(
                        prior_supp,
                        prior_density,
                        color="k",
                        alpha=0.5,
                        lw=2,
                        label="Prior",
                    )

            if var_name in self.truths.keys():
                ax[v].axvline(
                    self.truths[var_name], color="k", linestyle="-", label="Truth"
                )
                counts_hdi = 0
                counts_50_quantile = 0
                for d in self.results[var_name]:
                    hdi = av.hdi(d, 0.5)
                    quantile = np.quantile(d, [0.25, 0.75])
                    true_val = self.truths[var_name]
                    if true_val <= hdi[1] and true_val >= hdi[0]:
                        counts_hdi += 1
                    if true_val <= quantile[1] and true_val >= hdi[0]:
                        counts_50_quantile += 1
                length = len(self.results[var_name]) - mask_results.size
                text = [
                    f"fraction in 50% HDI: {counts_hdi / length:.2f}\n"
                    + f"fraction in 50% central interval: {counts_50_quantile / length:.2f}"
                ]
                handles = [
                    mpl_patches.Rectangle(
                        (0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0
                    )
                ]
                ax[v].legend(
                    handles=handles,
                    labels=text,
                    loc="best",
                    handlelength=0,
                    handletextpad=0,
                )

            ax[v].set_xlabel(var_labels[v], labelpad=10)
            ax[v].set_yticks([])

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
            logger.warning(
                "%.2f percent of fits have issues!"
                % (len(ind_not_ok) / len(diagnostics_array) * 100)
            )

        else:
            logger.info("No issues found")

        return ind_not_ok

    def _single_run(self, n_subjobs, seed, **kwargs):
        """
        Single run to be called using Parallel.
        """

        save_events = kwargs.pop("save_events", False)

        sys.stderr.write("Random seed: %i\n" % seed)

        roi = _make_roi()

        self._sources = _initialise_sources()

        file_config = hnu_config["file_config"]

        asimov = hnu_config.parameter_config.asimov

        subjob_seeds = [(seed + subjob) * 10 for subjob in range(n_subjobs)]

        outputs = {}
        for key in self._default_var_names:
            outputs[key] = []
        for key in self._diagnostic_names:
            outputs[key] = []

        outputs["diagnostics_ok"] = []
        outputs["run_time"] = []
        outputs["Lambda"] = []
        if save_events:
            outputs["events"] = []
            outputs["association_prob"] = []
            outputs["event_Lambda"] = []

        fit = None
        for i, s in enumerate(subjob_seeds):
            sys.stderr.write("Run %i\n" % i)
            # Simulation
            # Should reduce time consumption if only on first iteration model is compiled
            if i == 0:
                if asimov:
                    sim = Simulation(
                        self._sources,
                        self._detector_model_type,
                        self._obs_time,
                        N=self._N,
                    )
                else:
                    sim = Simulation(
                        self._sources,
                        self._detector_model_type,
                        self._obs_time,
                    )
                sim.precomputation()
                sim.setup_stan_sim(os.path.splitext(file_config["sim_filename"])[0])

            sim.run(seed=s, verbose=True)
            # self.sim = sim

            # Skip if no detected events
            if not sim.events:
                continue

            events = sim.events

            lambd = sim._sim_output.stan_variable("Lambda").squeeze()

            ps = np.sum(lambd == 1.0)
            diff = np.sum(lambd == 2.0)
            atmo = np.sum(lambd == 3.0)
            lam = np.array([ps, diff, atmo])

            # Skip if no detected events
            if not sim.events:
                continue

            # Fit
            # Same as above, save time
            # Also handle in case first sim has no events
            if not fit:
                fit = StanFit(
                    self._sources,
                    self._detector_model_type,
                    events,
                    self._obs_time,
                    priors=self.priors,
                    nshards=self._threads_per_chain,
                )
                fit.precomputation()
                fit.setup_stan_fit(os.path.splitext(file_config["fit_filename"])[0])

            else:
                fit.events = events

            start_time = time.time()

            share_L = hnu_config.parameter_config.share_L
            share_src_index = hnu_config.parameter_config.share_src_index

            if not share_L:
                L_init = [1e49] * len(self._sources.point_source)
            else:
                L_init = 1e49

            if not share_src_index:
                src_init = [2.3] * len(self._sources.point_source)
            else:
                src_init = 2.3
            inits = {
                "F_diff": 1e-4,
                "F_atmo": 0.3,
                "E": [1e5] * fit.events.N,
                "L": L_init,
                "src_index": src_init,
            }
            fit.run(
                seed=s,
                show_progress=True,
                inits=inits,
                **kwargs,
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

            if save_events:
                outputs["events"].append(events)
                outputs["association_prob"].append(
                    np.array(fit._get_event_classifications())
                )
                outputs["event_Lambda"].append(lambd)

            diagnostics_output_str = fit._fit_output.diagnose()

            if "no problems detected" in diagnostics_output_str:
                outputs["diagnostics_ok"].append(1)
            else:
                outputs["diagnostics_ok"].append(0)

        return outputs

    @staticmethod
    def _get_dm_from_config(dm_key):
        return [Refrigerator.python2dm(dm) for dm in dm_key]

    @staticmethod
    def _get_obs_time_from_config(dms, obs_time):
        return {dm: obs_time[c] * u.year for c, dm in enumerate(dms)}

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
    share_L = parameter_config["share_L"]
    share_src_index = parameter_config["share_src_index"]

    Parameter.clear_registry()
    indices = []
    if not share_src_index:
        for c, idx in enumerate(parameter_config["src_index"]):
            name = f"ps_{c}_src_index"
            indices.append(
                Parameter(
                    idx,
                    name,
                    fixed=False,
                    par_range=parameter_config["src_index_range"],
                )
            )
    else:
        indices.append(
            Parameter(
                parameter_config["src_index"][0],
                "src_index",
                fixed=False,
                par_range=parameter_config["src_index_range"],
            )
        )
    diff_index = Parameter(
        parameter_config["diff_index"],
        "diff_index",
        fixed=False,
        par_range=parameter_config["diff_index_range"],
    )
    L = []
    if not share_L:
        for c, Lumi in enumerate(parameter_config["L"]):
            name = f"ps_{c}_luminosity"
            L.append(
                Parameter(
                    Lumi * u.erg / u.s,
                    name,
                    fixed=True,
                    par_range=parameter_config["L_range"] * u.erg / u.s,
                )
            )
    else:
        L.append(
            Parameter(
                parameter_config["L"][0] * u.erg / u.s,
                "luminosity",
                fixed=False,
                par_range=parameter_config["L_range"] * u.erg / u.s,
            )
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
        for dm in Refrigerator.detectors:
            # Create a parameter for each detector
            # If the detector is not used, the parameter is disregarded
            _ = Parameter(
                parameter_config[f"Emin_det_{dm.P}"] * u.GeV,
                f"Emin_det_{dm.P}",
                fixed=True,
            )

    dec = np.deg2rad(parameter_config["src_dec"]) * u.rad
    ra = np.deg2rad(parameter_config["src_ra"]) * u.rad
    center = SkyCoord(ra=ra, dec=dec, frame="icrs")

    sources = Sources()

    for c in range(len(dec)):
        if share_L:
            Lumi = L[0]
        else:
            Lumi = L[c]

        if share_src_index:
            idx = indices[0]
        else:
            idx = indices[c]
        point_source = PointSource.make_powerlaw_source(
            f"ps_{c}",
            dec[c],
            ra[c],
            Lumi,
            idx,
            parameter_config["z"][c],
            Emin_src,
            Emax_src,
        )

        sources.add(point_source)
    sources.add_diffuse_component(
        diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff, 0.0
    )
    sources.add_atmospheric_component()

    return sources


def _make_roi():
    ROIList.clear_registry()
    parameter_config = hnu_config["parameter_config"]
    roi_config = hnu_config["roi_config"]
    dec = np.deg2rad(parameter_config["src_dec"]) * u.rad
    ra = np.deg2rad(parameter_config["src_ra"]) * u.rad
    center = SkyCoord(ra=ra, dec=dec, frame="icrs")

    roi_config = hnu_config["roi_config"]
    size = roi_config["size"] * u.deg
    apply_roi = roi_config["apply_roi"]

    if apply_roi and len(dec) > 1 and not roi_config["roi_type"] == "CircularROI":
        raise ValueError("Only CircularROIs can be stacked")
    if roi_config["roi_type"] == "CircularROI":
        for c in range(len(dec)):
            CircularROI(center[c], size, apply_roi=apply_roi)
    elif roi_config["roi_type"] == "RectangularROI":
        size = size.to(u.rad)
        RectangularROI(
            RA_min=ra[0] - size,
            RA_max=ra[0] + size,
            DEC_min=dec[0] - size,
            DEC_max=dec[0] + size,
            apply_roi=apply_roi,
        )
    elif roi_config["roi_type"] == "FullSkyROI":
        FullSkyROI()
    elif roi_config["roi_type"] == "NorthernSkyROI":
        NorthernSkyROI()
