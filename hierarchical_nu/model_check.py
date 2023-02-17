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
from hierarchical_nu.detector.r2021 import R2021DetectorModel

from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.utils.config import hnu_config
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior


class ModelCheck:
    """
    Check statistical model by repeatedly
    fitting simulated data using different random seeds.
    """

    def __init__(self, truths=None, priors=None):

        if priors:

            self.priors = priors

        else:

            self.priors = Priors()

        if truths:

            self.truths = truths

        else:

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
            self._nshards = parameter_config["nshards"]

            sim = Simulation(self._sources, self._detector_model_type, self._obs_time)
            sim.precomputation()
            self._exposure_integral = sim._exposure_integral
            sim.set_stan_filename(file_config["sim_filename"])
            sim.compile_stan_code(include_paths=list(file_config["include_paths"]))
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

            """
            self.priors.atmospheric_flux = NormalPrior(
                mu=atmo_bg.flux_model.total_flux_int.to(flux_unit),
                sigma=atmo_bg.flux_model.total_flux_int.to(flux_unit))
            self.priors.diffuse_flux = NormalPrior(
                mu=self._sources.diffuse.flux_model.total_flux_int.to(flux_unit).value, 
                sigma=self._sources.diffuse.flux_model.total_flux_int.to(flux_unit).value,
            )
            """
        self._default_var_names = [key for key in self.truths]

    @staticmethod
    def initialise_env(output_dir, priors: Priors = Priors()):
        """
        Script to set up enviroment for parallel
        model checking runs.

        * Runs MCEq for atmo flux if needed
        * Generates and compiles necessary Stan files
        Only need to run once before calling ModelCheck(...).run()
        """

        def make_prior(p):
            if p["name"] == "LogNormalPrior":
                prior = LogNormalPrior(mu=np.log(p["mu"]), sigma=p["sigma"])
            elif p["name"] == "NormalPrior":
                prior = NormalPrior(mu=p["mu"], sigma=p["sigma"])
            else:
                raise ValueError("Currently no other prior implemented")
            return prior
        

        # Config
        parameter_config = hnu_config["parameter_config"]
        file_config = hnu_config["file_config"]
        prior_config = hnu_config["prior_config"]

        # Run MCEq computation
        print("Setting up MCEq run for AtmopshericNumuFlux")
        Emin = parameter_config["Emin"] * u.GeV
        Emax = parameter_config["Emax"] * u.GeV
        atmo_flux_model = AtmosphericNuMuFlux(Emin, Emax)
        nshards = parameter_config["nshards"]

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
        fit_name = file_config["fit_filename"][:-5]
        
        priors = Priors()
        priors.luminosity = make_prior(prior_config["L"])
        priors.src_index = make_prior(prior_config["src_index"])
        priors.atmospheric_flux = make_prior(prior_config["atmospheric_flux"])
        priors.diffuse_flux = make_prior(prior_config["diffuse_flux"])
        priors.diff_index = make_prior(prior_config["diff_index"])


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

        _ = CmdStanModel(
            stan_file=file_config["sim_filename"],
            stanc_options=stanc_options,
        )
        _ = CmdStanModel(
            stan_file=file_config["fit_filename"], stanc_options=stanc_options
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

            for i, res in enumerate(self._results):
                folder = f.create_group("results_%i" % i)

                for key, value in res.items():
                    folder.create_dataset(key, data=value)

        self.priors.addto(filename, "priors")

    @classmethod
    def load(cls, filename_list):

        with h5py.File(filename_list[0], "r") as f:
            job_folder = f["results_0"]
            _default_var_names = [key for key in job_folder]

        truths = {}
        priors = None

        results = {}
        for key in _default_var_names:
            results[key] = []

        file_truths = {}
        for filename in filename_list:

            file_priors = Priors.from_group(filename, "priors")

            if not priors:

                priors = file_priors

            with h5py.File(filename, "r") as f:

                truths_folder = f["truths"]
                for key, value in truths_folder.items():
                    file_truths[key] = value[()]

                if not truths:
                    truths = file_truths

                if truths != file_truths:
                    raise ValueError(
                        "Files in list have different truth settings and should not be combined"
                    )

                n_jobs = len(
                    [key for key in f if (key != "truths" and key != "priors")]
                )
                for i in range(n_jobs):
                    job_folder = f["results_%i" % i]
                    for res_key in job_folder:
                        results[res_key].extend(job_folder[res_key][()])

        output = cls(truths, priors)
        output.results = results

        return output

    def compare(self, var_names=None, var_labels=None, show_prior=False, nbins=50):

        if not var_names:
            var_names = self._default_var_names

        if not var_labels:
            var_labels = self._default_var_names

        N = len(var_names)
        fig, ax = plt.subplots(N, figsize=(5, N * 3))

        for v, var_name in enumerate(var_names):

            if var_name == "L" or var_name == "F_diff" or var_name == "F_atmo":

                prior_supp = np.geomspace(
                    np.min(self.results[var_name]),
                    np.max(self.results[var_name]),
                    1000,
                )

                bins =  np.geomspace(
                    np.min(self.results[var_name]),
                    np.max(self.results[var_name]),
                    nbins,
                )
                ax[v].set_xscale("log")

            else:

                prior_supp = np.linspace(
                    np.min(self.results[var_name]),
                    np.max(self.results[var_name]),
                    1000,
                )

                bins = np.linspace(
                    np.min(self.results[var_name]),
                    np.max(self.results[var_name]),
                    nbins,
                )
            max_value = 0
            for i in range(len(self.results[var_name])):
                n, bins, _ = ax[v].hist(
                    self.results[var_name][i],
                    color="#017B76",
                    alpha=0.3,
                    histtype="step",
                    bins=bins,
                    lw=1.0,
                )

                max_value = n.max() if \
                    n.max() > max_value else max_value

            if show_prior:

                N = len(self.results[var_name][0]) * 100
                xmin, xmax = ax[v].get_xlim()

                if "f_" in var_name and not "diff" in var_name:   # yikes

                    prior_samples = uniform(0, 1).rvs(N)
                    prior_density = uniform(0, 1).pdf(prior_supp)
                    plot = True

                elif "Nex" in var_name:

                    plot = False

                elif "index" in var_name:
                    prior_density = self.priors.to_dict()[var_name].pdf(prior_supp)
                    plot = True

                else:

                    prior_samples = self.priors.to_dict()[var_name].sample(N)
                    prior_density = self.priors.to_dict()[var_name].pdf_logspace(prior_supp)
                    plot = True
                
                
                
                if plot:
                    ax[v].plot(
                        prior_supp,
                        prior_density / prior_density.max() * max_value / 2.,
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

            ax[v].axvline(
                self.truths[var_name], color="k", linestyle="-", label="Truth"
            )
            ax[v].set_xlabel(var_labels[v], labelpad=10)
            ax[v].legend(loc="best")

        fig.tight_layout()
        return fig, ax

    def _single_run(self, n_subjobs, seed, **kwargs):
        """
        Single run to be called using Parallel.
        """

        sys.stderr.write("Random seed: %i\n" % seed)

        self._sources = _initialise_sources()

        file_config = hnu_config["file_config"]

        subjob_seeds = [(seed + subjob) * 10 for subjob in range(n_subjobs)]

        start_time = time.time()

        outputs = {}
        for key in self._default_var_names:
            outputs[key] = []

        for i, s in enumerate(subjob_seeds):

            sys.stderr.write("Run %i\n" % i)

            # Simulation
            # Should reduce time consumption if only on first iteration model is compiled
            if i == 0:
                sim = Simulation(self._sources, self._detector_model_type, self._obs_time)
                sim.precomputation(self._exposure_integral)
                sim.set_stan_filename(file_config["sim_filename"])
                sim.compile_stan_code(include_paths=list(file_config["include_paths"]))
            sim.run(seed=s, verbose=True)
            self.sim = sim

            lam = sim._sim_output.stan_variable("Lambda")[0]
            sim_output = {}
            sim_output["Lambda"] = lam
            

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
                    nshards=self._nshards
                )
                fit.precomputation()
                fit.set_stan_filename(file_config["fit_filename"])
                fit.compile_stan_code(include_paths=list(file_config["include_paths"]))
            
            else:
                fit.events = sim.events
            fit.run(seed=s, show_progress=True, inits={"L": 1e52, "src_index": 2.0}, **kwargs)

            self.fit = fit

            # Store output
            for key in outputs:
                outputs[key].append(fit._fit_output.stan_variable(key))

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
        "test",
        np.deg2rad(parameter_config["src_dec"]) * u.rad,
        np.deg2rad(parameter_config["src_ra"]) * u.rad,
        L,
        src_index,
        0.43,
        Emin,
        Emax,
    )

    sources = Sources()
    sources.add(point_source)
    sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index)
    sources.add_atmospheric_component()

    return sources
