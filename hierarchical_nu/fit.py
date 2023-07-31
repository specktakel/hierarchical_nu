import numpy as np
import os
import h5py
import logging
import collections
from astropy import units as u
from typing import List, Union
import corner

from math import ceil, floor

from cmdstanpy import CmdStanModel

from hierarchical_nu.source.source import Sources, PointSource, icrs_to_uv
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import IsotropicDiffuseBG
from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.detector.detector_model import DetectorModel
from hierarchical_nu.detector.r2021 import R2021DetectorModel
from hierarchical_nu.precomputation import ExposureIntegral
from hierarchical_nu.events import Events
from hierarchical_nu.priors import Priors, NormalPrior, LogNormalPrior

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.stan.fit_interface import StanFitInterface


class StanFit:
    """
    To set up and run fits in Stan.
    """

    @u.quantity_input
    def __init__(
        self,
        sources: Sources,
        detector_model: DetectorModel,
        events: Events,
        observation_time: u.year,
        priors: Priors = Priors(),
        atmo_flux_energy_points: int = 100,
        atmo_flux_theta_points: int = 30,
        n_grid_points: int = 50,
        nshards: int = 0,
    ):
        """
        To set up and run fits in Stan.
        """

        self._sources = sources
        self._detector_model_type = detector_model
        self._events = events
        self._observation_time = observation_time
        self._n_grid_points = n_grid_points
        self._nshards = nshards
        self._priors = priors

        self._sources.organise()

        stan_file_name = os.path.join(STAN_GEN_PATH, "model_code")

        self._stan_interface = StanFitInterface(
            stan_file_name,
            self._sources,
            self._detector_model_type,
            priors=priors,
            nshards=nshards,
            atmo_flux_energy_points=atmo_flux_energy_points,
            atmo_flux_theta_points=atmo_flux_theta_points,
        )

        # Check for unsupported combinations
        if sources.atmospheric and detector_model.event_types == ["cascades"]:
            raise NotImplementedError(
                "AtmosphericNuMuFlux currently only implemented "
                + "for use with NorthernTracksDetectorModel or "
                + "IceCubeDetectorModel"
            )

        if (
            sources.atmospheric
            and sources.N == 1
            and "cascades" in detector_model.event_types
        ):
            raise NotImplementedError(
                "AtmosphericNuMuFlux as the only source component "
                + "for IceCubeDetectorModel is not implemented. Just use "
                + "NorthernTracksDetectorModel instead."
            )

        # Silence log output
        logger = logging.getLogger("hierarchical_nu.backend.code_generator")
        logger.propagate = False

        # For use with plot methods
        self._def_var_names = []

        if self._sources.point_source:
            self._def_var_names.append("L")
            self._def_var_names.append("src_index")

        if self._sources.diffuse:
            self._def_var_names.append("F_diff")
            self._def_var_names.append("diff_index")

        if self._sources.atmospheric:
            self._def_var_names.append("F_atmo")

        if self._sources._point_source and (
            self._sources.atmospheric or self._sources.diffuse
        ):
            self._def_var_names.append("f_arr")
            self._def_var_names.append("f_det")

        self._exposure_integral = collections.OrderedDict()

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, events: Events):
        if isinstance(events, Events):
            self._events = events
        else:
            raise ValueError("events must be instance of Events")

    def precomputation(
        self,
        exposure_integral: collections.OrderedDict = None,
    ):
        if not exposure_integral:
            for event_type in self._detector_model_type.event_types:
                self._exposure_integral[event_type] = ExposureIntegral(
                    self._sources,
                    self._detector_model_type,
                    n_grid_points=self._n_grid_points,
                    event_type=event_type,
                )

        else:
            self._exposure_integral = exposure_integral

    def generate_stan_code(self):
        self._fit_filename = self._stan_interface.generate()

    def set_stan_filename(self, fit_filename):
        self._fit_filename = fit_filename

    def compile_stan_code(self, include_paths=None):
        if not include_paths:
            include_paths = [STAN_PATH]
        if self._detector_model_type == R2021DetectorModel:
            r2021_path = os.path.join(os.getcwd(), ".stan_files")
            if not r2021_path in include_paths:
                include_paths.append(r2021_path)

        self._fit = CmdStanModel(
            stan_file=self._fit_filename,
            stanc_options={"include-paths": include_paths},
            cpp_options={"STAN_THREADS": True},
        )

    def setup_stan_fit(self, filename):
        """
        Create stan model from already compiled file
        """

        self._fit = CmdStanModel(exe_file=filename)

    def run(
        self,
        iterations: int = 1000,
        chains: int = 1,
        seed: int = None,
        show_progress: bool = False,
        threads_per_chain: Union[int, None] = None,
        **kwargs,
    ):
        # Use threads_per_chain = nshards as default
        if not threads_per_chain and self._nshards > 0:
            threads_per_chain = self._nshards

        self._fit_inputs = self._get_fit_inputs()

        self._fit_output = self._fit.sample(
            data=self._fit_inputs,
            iter_sampling=iterations,
            chains=chains,
            seed=seed,
            show_progress=show_progress,
            threads_per_chain=threads_per_chain,
            **kwargs,
        )

    def setup_and_run(
        self,
        iterations: int = 1000,
        chains: int = 1,
        seed: int = None,
        show_progress: bool = False,
        include_paths: List[str] = None,
        **kwargs,
    ):
        self.precomputation()
        self.generate_stan_code()
        self.compile_stan_code(include_paths=include_paths)
        self.run(
            iterations=iterations,
            chains=chains,
            seed=seed,
            show_progress=show_progress,
            **kwargs,
        )

    def plot_trace(self, var_names=None, **kwargs):
        """
        Trace plot using list of stan parameter keys.
        """

        import arviz

        if not var_names:
            var_names = self._def_var_names

        arviz.plot_trace(self._fit_output, var_names=var_names, **kwargs)

    def corner_plot(self, var_names=None, truths=None):
        """
        Corner plot using list of Stan parameter keys and optional
        true values if working with simulated data.
        """

        if not var_names:
            var_names = self._def_var_names

        chain = self._fit_output.stan_variables()

        # Organise samples
        samples_list = []
        label_list = []

        for key in var_names:
            if len(np.shape(chain[key])) > 1:
                for i, s in enumerate(chain[key].T):
                    samples_list.append(s)

                    if key == "L" or key == "src_index":
                        label = "ps_%i_" % i + key
                    else:
                        label = key

                    label_list.append(label)

            else:
                samples_list.append(chain[key])
                label_list.append(key)

        # Organise truths
        if truths:
            truths_list = []

            for key in var_names:
                if truths[key].size > 1:
                    for t in truths[key]:
                        truths_list.append(t)

                else:
                    truths_list.append(truths[key])

        else:
            truths_list = None

        samples = np.column_stack(samples_list)

        return corner.corner(samples, labels=label_list, truths=truths_list)

    def save(self, filename, overwrite: bool = False):
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"File {filename} already exists.")

        with h5py.File(filename, "w") as f:
            fit_folder = f.create_group("fit")
            inputs_folder = fit_folder.create_group("inputs")
            outputs_folder = fit_folder.create_group("outputs")

            for key, value in self._fit_inputs.items():
                inputs_folder.create_dataset(key, data=value)

            for key, value in self._fit_output.stan_variables().items():
                outputs_folder.create_dataset(key, data=value)

            outputs_folder.create_dataset("diagnose", data=self._fit_output.diagnose())

    @classmethod
    def from_file(cls, filename):
        """
        Load fit output from file. Allows to
        make plots and run classification check.
        """

        priors_dict = {}

        fit_inputs = {}

        with h5py.File(filename, "r") as f:
            if "fit" not in f.keys():
                raise ValueError("File is not a saved hierarchical_nu fit.")

            for k, v in f["fit/inputs"].items():
                if "mu" in k or "sigma" in k:
                    priors_dict[k] = v[()]

        priors = Priors()
        priors.luminosity = LogNormalPrior(
            mu=priors_dict["lumi_mu"], sigma=priors_dict["lumi_sigma"]
        )
        priors.src_index = NormalPrior(
            mu=priors_dict["src_index_mu"], sigma=priors_dict["src_index_sigma"]
        )
        priors.diff_index = NormalPrior(
            mu=priors_dict["diff_index_mu"], sigma=priors_dict["diff_index_sigma"]
        )
        priors.atmospheric_flux = LogNormalPrior(
            mu=priors_dict["f_atmo_mu"], sigma=priors_dict["f_atmo_sigma"]
        )
        priors.diffuse_flux = LogNormalPrior(
            mu=priors_dict["f_diff_mu"], sigma=priors_dict["f_diff_sigma"]
        )

    def check_classification(self, sim_outputs):
        """
        For the case of simulated data, check if
        events are correctly classified into the
        different source categories.
        """

        Ns = len([s for s in self._sources.sources if isinstance(s, PointSource)])

        event_labels = sim_outputs["Lambda"] - 1

        prob_each_src = self._get_event_classifications()

        source_labels = ["src%i" % src for src in range(Ns)]

        if self._sources.atmospheric and self._sources.diffuse:
            source_labels.append("diff")
            source_labels.append("atmo")

        elif self._sources.diffuse:
            source_labels.append("diff")

        elif self._sources.atmospheric:
            source_labels.append("atmo")

        wrong = []
        assumed = []
        correct = []

        for i in range(len(prob_each_src)):
            classified = np.where(prob_each_src[i] == np.max(prob_each_src[i]))[0][
                0
            ] == int(event_labels[i])

            if not classified:
                wrong.append(i)

                print("Event %i is misclassified" % i)

                for src in range(Ns):
                    print("P(src%i) = %.6f" % (src, prob_each_src[i][src]))

                if self._sources.atmospheric and self._sources.diffuse:
                    print("P(diff) = %.6f" % prob_each_src[i][Ns])
                    print("P(atmo) = %.6f" % prob_each_src[i][Ns + 1])

                elif self._sources.diffuse:
                    print("P(diff) = %.6f" % prob_each_src[i][Ns])

                elif self._sources.atmospheric:
                    print("P(atmo) = %.6f" % prob_each_src[i][Ns])

                print("The correct component is", source_labels[int(event_labels[i])])
                correct.append(source_labels[int(event_labels[i])])
                assumed.append(event_labels[i])

        if not wrong:
            print("All events are correctly classified")
        else:
            print(
                "A total of %i events out of %i are misclassified"
                % (len(wrong), len(event_labels))
            )

        return wrong, assumed, correct

    def _get_event_classifications(self):
        logprob = self._fit_output.stan_variable("lp").transpose(1, 2, 0)

        n_comps = np.shape(logprob)[1]

        prob_each_src = []
        for lp in logprob:
            lps = []
            ps = []
            for src in range(n_comps):
                lps.append(np.mean(np.exp(lp[src])))
            norm = sum(lps)

            for src in range(n_comps):
                ps.append(lps[src] / norm)

            prob_each_src.append(ps)

        return prob_each_src

    def _get_fit_inputs(self):
        fit_inputs = {}
        fit_inputs["N"] = self._events.N
        if self._nshards not in [0, 1]:
            # Number of shards and max. events per shards only used if multithreading is desired
            fit_inputs["N_shards"] = self._nshards
            fit_inputs["J"] = ceil(fit_inputs["N"] / fit_inputs["N_shards"])
        fit_inputs["Ns_tot"] = len([s for s in self._sources.sources])
        fit_inputs["Edet"] = self._events.energies.to(u.GeV).value
        fit_inputs["omega_det"] = self._events.unit_vectors
        fit_inputs["omega_det"] = [
            (_ / np.linalg.norm(_)).tolist() for _ in fit_inputs["omega_det"]
        ]
        fit_inputs["event_type"] = self._events.types
        fit_inputs["kappa"] = self._events.kappas
        fit_inputs["Ns"] = len(
            [s for s in self._sources.sources if isinstance(s, PointSource)]
        )

        redshift = [
            s.redshift
            for s in self._sources.sources
            if isinstance(s, PointSource)
            or isinstance(s.flux_model, IsotropicDiffuseBG)
        ]
        D = [
            luminosity_distance(s.redshift).value
            for s in self._sources.sources
            if isinstance(s, PointSource)
        ]
        src_pos = [
            icrs_to_uv(s.dec.value, s.ra.value)
            for s in self._sources.sources
            if isinstance(s, PointSource)
        ]

        fit_inputs["z"] = redshift
        fit_inputs["D"] = D
        fit_inputs["varpi"] = src_pos

        fit_inputs["Emin_src"] = (
            Parameter.get_parameter("Emin_src").value.to(u.GeV).value
        )
        fit_inputs["Emax_src"] = (
            Parameter.get_parameter("Emax_src").value.to(u.GeV).value
        )

        fit_inputs["Emin"] = Parameter.get_parameter("Emin").value.to(u.GeV).value
        fit_inputs["Emax"] = Parameter.get_parameter("Emax").value.to(u.GeV).value

        fit_inputs["Emin_diff"] = (
            Parameter.get_parameter("Emin_diff").value.to(u.GeV).value
        )
        fit_inputs["Emax_diff"] = (
            Parameter.get_parameter("Emax_diff").value.to(u.GeV).value
        )

        fit_inputs["T"] = self._observation_time.to(u.s).value

        event_type = self._detector_model_type.event_types[0]

        fit_inputs["Ngrid"] = self._exposure_integral[event_type]._n_grid_points

        if self._sources.point_source:
            try:
                Parameter.get_parameter("src_index")
                key = "src_index"
            except ValueError:
                key = "ps_0_src_index"

            fit_inputs["src_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids[key]

            # Inputs for priors of point sources
            fit_inputs["src_index_mu"] = self._priors.src_index.mu
            fit_inputs["src_index_sigma"] = self._priors.src_index.sigma
            if self._priors.luminosity.name in ["normal", "lognormal"]:
                fit_inputs["lumi_mu"] = self._priors.luminosity.mu
                fit_inputs["lumi_sigma"] = self._priors.luminosity.sigma
            elif self._priors.luminosity.name == "pareto":
                fit_inputs["lumi_xmin"] = self._priors.luminosity.xmin
                fit_inputs["lumi_alpha"] = self._priors.luminosity.alpha
            else:
                raise ValueError("No other prior type for luminosity implemented")

        if self._sources.diffuse:
            fit_inputs["diff_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids["diff_index"]

            # Priors for diffuse model
            fit_inputs["f_diff_mu"] = self._priors.diffuse_flux.mu
            fit_inputs["f_diff_sigma"] = self._priors.diffuse_flux.sigma
            fit_inputs["diff_index_mu"] = self._priors.diff_index.mu
            fit_inputs["diff_index_sigma"] = self._priors.diff_index.sigma

        if "tracks" in self._stan_interface._event_types:
            fit_inputs["integral_grid_t"] = [
                _.to(u.m**2).value.tolist()
                for _ in self._exposure_integral["tracks"].integral_grid
            ]

            if self._sources.point_source:
                fit_inputs["aeff_egrid_t"] = (
                    self._exposure_integral["tracks"]
                    .pdet_grid[0]
                    .to(u.GeV)
                    .value.tolist()
                )
                fit_inputs["aeff_slice_t"] = [
                    _.to(u.m**2).value.tolist()
                    for _ in self._exposure_integral["tracks"].pdet_grid[1:]
                ]
                fit_inputs["aeff_len_t"] = len(
                    self._exposure_integral["tracks"]
                    .pdet_grid[0]
                    .to(u.GeV)
                    .value.tolist()
                )

        if "cascades" in self._stan_interface._event_types:
            fit_inputs["integral_grid_c"] = [
                _.to(u.m**2).value.tolist()
                for _ in self._exposure_integral["cascades"].integral_grid
            ]

            if self._sources.point_source:
                fit_inputs["aeff_egrid_c"] = (
                    self._exposure_integral["cascades"]
                    .pdet_grid[0]
                    .to(u.GeV)
                    .value.tolist()
                )
                fit_inputs["aeff_slice_c"] = [
                    _.to(u.m**2).value.tolist()
                    for _ in self._exposure_integral["cascades"].pdet_grid[1:]
                ]
                fit_inputs["aeff_len_c"] = len(
                    self._exposure_integral["cascades"]
                    .pdet_grid[0]
                    .to(u.GeV)
                    .value.tolist()
                )

        if self._sources.atmospheric:
            fit_inputs["atmo_integ_val"] = (
                self._exposure_integral["tracks"]
                .integral_fixed_vals[0]
                .to(u.m**2)
                .value
            )

            fit_inputs[
                "atmo_integrated_flux"
            ] = self._sources.atmospheric.flux_model.total_flux_int.to(
                1 / (u.m**2 * u.s)
            ).value

            # Priors for atmo model
            fit_inputs["f_atmo_mu"] = self._priors.atmospheric_flux.mu
            fit_inputs["f_atmo_sigma"] = self._priors.atmospheric_flux.sigma

        # To work with cmdstanpy serialization
        fit_inputs = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in fit_inputs.items()
        }

        return fit_inputs
