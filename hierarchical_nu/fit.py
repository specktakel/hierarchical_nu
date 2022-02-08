import numpy as np
import os
import h5py
import logging
import collections
from astropy import units as u
import corner

from cmdstanpy import CmdStanModel

from hierarchical_nu.source.source import Sources, PointSource, icrs_to_uv
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import IsotropicDiffuseBG
from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.detector.detector_model import DetectorModel
from hierarchical_nu.precomputation import ExposureIntegral
from hierarchical_nu.events import Events
from hierarchical_nu.priors import Priors

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
    ):
        """
        To set up and run fits in Stan.
        """

        self._sources = sources
        self._detector_model_type = detector_model
        self._events = events
        self._observation_time = observation_time

        self._sources.organise()

        stan_file_name = os.path.join(STAN_GEN_PATH, "model_code")

        self._stan_interface = StanFitInterface(
            stan_file_name,
            self._sources,
            self._detector_model_type,
            priors=priors,
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

    def precomputation(
        self,
        exposure_integral: collections.OrderedDict = None,
    ):

        if not exposure_integral:

            for event_type in self._detector_model_type.event_types:

                self._exposure_integral[event_type] = ExposureIntegral(
                    self._sources,
                    self._detector_model_type,
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

        self._fit = CmdStanModel(
            stan_file=self._fit_filename, stanc_options={"include-paths": include_paths}
        )

    def run(self, iterations=1000, chains=1, seed=None, show_progress=False, **kwargs):

        self._fit_inputs = self._get_fit_inputs()

        self._fit_output = self._fit.sample(
            data=self._fit_inputs,
            iter_sampling=iterations,
            chains=chains,
            seed=seed,
            show_progress=show_progress,
            **kwargs
        )

    def setup_and_run(
        self,
        iterations=1000,
        chains=1,
        seed=None,
        show_progress=False,
        include_paths=None,
        **kwargs
    ):

        self.precomputation()
        self.generate_stan_code()
        self.compile_stan_code(include_paths=include_paths)
        self.run(
            iterations=iterations,
            chains=chains,
            seed=seed,
            show_progress=show_progress,
            **kwargs
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

    def save(self, filename):
        """
        TODO: Add overwrite check.
        """

        with h5py.File(filename, "w") as f:

            fit_folder = f.create_group("fit")
            inputs_folder = fit_folder.create_group("inputs")
            outputs_folder = fit_folder.create_group("outputs")

            for key, value in self._fit_inputs.items():
                inputs_folder.create_dataset(key, data=value)

            for key, value in self._fit_output.stan_variables().items():
                outputs_folder.create_dataset(key, data=value)

    @classmethod
    def from_file(cls, filename):
        """
        Load fit output from file. Allows to
        make plots and run classification check.
        """

        raise NotImplementedError()

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
                print()

        if not wrong:
            print("All events are correctly classified")
        else:
            print(
                "A total of %i events out of %i are misclassified"
                % (len(wrong), len(event_labels))
            )

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

        fit_inputs["Esrc_min"] = Parameter.get_parameter("Emin").value.to(u.GeV).value
        fit_inputs["Esrc_max"] = Parameter.get_parameter("Emax").value.to(u.GeV).value

        fit_inputs["T"] = self._observation_time.to(u.s).value

        event_type = self._detector_model_type.event_types[0]
        fit_inputs["E_grid"] = self._exposure_integral[event_type].energy_grid.value

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

        if self._sources.diffuse:

            fit_inputs["diff_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids["diff_index"]

        if "tracks" in self._stan_interface._event_types:

            fit_inputs["integral_grid_t"] = [
                _.value.tolist()
                for _ in self._exposure_integral["tracks"].integral_grid
            ]

            fit_inputs["Pdet_grid_t"] = np.array(
                self._exposure_integral["tracks"].pdet_grid
            )

        if "cascades" in self._stan_interface._event_types:

            fit_inputs["integral_grid_c"] = [
                _.value.tolist()
                for _ in self._exposure_integral["cascades"].integral_grid
            ]

            fit_inputs["Pdet_grid_c"] = np.array(
                self._exposure_integral["cascades"].pdet_grid
            )

        if self._sources.atmospheric:

            fit_inputs["atmo_integ_val"] = (
                self._exposure_integral["tracks"].integral_fixed_vals[0].value
            )

        # To work with cmdstanpy serialization
        fit_inputs = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in fit_inputs.items()
        }

        return fit_inputs
