import numpy as np
import os
import h5py
import logging
import collections
from astropy import units as u

from cmdstanpy import CmdStanModel

from .source.source import Sources, PointSource, icrs_to_uv
from .source.parameter import Parameter
from .source.flux_model import IsotropicDiffuseBG
from .source.cosmology import luminosity_distance
from .detector.detector_model import DetectorModel
from .detector.icecube import IceCubeDetectorModel
from .precomputation import ExposureIntegral
from .events import Events

from .stan_interface import (
    generate_stan_fit_code_,
    generate_stan_fit_code_hybrid_,
)


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
        output_dir="stan_files",
    ):
        """
        To set up and run fits in Stan.
        """

        self._sources = sources
        self._detector_model_type = detector_model
        self._events = events
        self._observation_time = observation_time
        self.output_dir = output_dir

        self._sources.organise()

        # Silence log output
        logger = logging.getLogger("python.backend.code_generator")
        logger.propagate = False

        # For use with plot methods
        if self._sources.atmo_component():
            self._def_var_names = ["L", "F_diff", "F_atmo", "f", "alpha"]
        else:
            self._def_var_names = ["L", "F_diff", "f", "alpha"]

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

        self._generate_stan_fit_code()

    def set_stan_filename(self, fit_filename):

        self._fit_filename = fit_filename

    def compile_stan_code(self, include_paths=None):

        if not include_paths:
            this_dir = os.path.abspath("")
            include_paths = [os.path.join(this_dir, self.output_dir)]

        self._fit = CmdStanModel(
            stan_file=self._fit_filename, stanc_options={"include_paths": include_paths}
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

        import corner

        if not var_names:
            var_names = self._def_var_names

        chain = self._fit_output.stan_variables()

        samples_list = [chain[key].values.T[0] for key in var_names]

        if truths:
            truths_list = [truths[key] for key in var_names]
        else:
            truths_list = None

        samples = np.column_stack(samples_list)

        return corner.corner(samples, labels=var_names, truths=truths_list)

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
                outputs_folder.create_dataset(key, data=value.values.T[0])

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
        source_labels.append("diff")

        if self._sources.atmo_component():
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
                print("P(diff) = %.6f" % prob_each_src[i][Ns])

                if self._sources.atmo_component():
                    print("P(atmo) = %.6f" % prob_each_src[i][Ns + 1])

                print("The correct component is", source_labels[int(event_labels[i])])

        if not wrong:
            print("All events are correctly classified")
        else:
            print(
                "A total of %i events out of %i are misclassified"
                % (len(wrong), len(event_labels))
            )

    def _get_event_classifications(self):

        N = self._fit_inputs["N"]
        Nsc = self._sources.N
        Nsamp = self._fit_output.chains * self._fit_output.num_draws

        logprob = self._fit_output.stan_variable("lp").values.reshape(Nsamp, Nsc, N)

        logprob = logprob.transpose(2, 1, 0)

        n_comps = np.shape(logprob)[1]
        if self._sources.atmo_component:
            Ns = n_comps - 2
        else:
            Ns = n_comps - 1

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
        fit_inputs["Edet_min"] = (
            Parameter.get_parameter("Emin_det").value.to(u.GeV).value
        )

        event_type = self._detector_model_type.event_types[0]
        fit_inputs["Ngrid"] = len(
            self._exposure_integral[event_type].par_grids["index"]
        )
        fit_inputs["alpha_grid"] = self._exposure_integral[event_type].par_grids[
            "index"
        ]

        if self._detector_model_type == IceCubeDetectorModel:

            fit_inputs["integral_grid_t"] = [
                _.value.tolist()
                for _ in self._exposure_integral["tracks"].integral_grid
            ]
            fit_inputs["integral_grid_c"] = [
                _.value.tolist()
                for _ in self._exposure_integral["cascades"].integral_grid
            ]

        else:

            fit_inputs["integral_grid"] = [
                _.value.tolist()
                for _ in self._exposure_integral[event_type].integral_grid
            ]

        fit_inputs["T"] = self._observation_time.to(u.s).value

        fit_inputs["E_grid"] = self._exposure_integral[event_type].energy_grid.value

        if self._detector_model_type == IceCubeDetectorModel:

            fit_inputs["Pdet_grid_t"] = np.array(
                self._exposure_integral["tracks"].pdet_grid
            )
            fit_inputs["Pdet_grid_c"] = np.array(
                self._exposure_integral["cascades"].pdet_grid
            )
        else:

            fit_inputs["Pdet_grid"] = np.array(
                self._exposure_integral[event_type].pdet_grid
            )

        fit_inputs["L_scale"] = (
            Parameter.get_parameter("luminosity").value.to(u.GeV / u.s).value
        )

        diffuse_bg = self._sources.diffuse_component()
        fit_inputs["F_diff_scale"] = diffuse_bg.flux_model.total_flux_int.value

        if self._sources.atmo_component():
            fit_inputs["atmo_integ_val"] = (
                self._exposure_integral["tracks"].integral_fixed_vals[0].value
            )

            atmo_bg = self._sources.atmo_component()
            fit_inputs["F_atmo_scale"] = atmo_bg.flux_model.total_flux_int.value

        fit_inputs["F_tot_scale"] = self._sources.total_flux_int().value

        # To work with cmdstanpy serialization
        fit_inputs = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in fit_inputs.items()
        }

        return fit_inputs

    def _generate_stan_fit_code(self):

        ps_spec_shape = self._sources.sources[0].flux_model.spectral_shape

        if self._sources.atmo_component():
            atmo_flux_model = self._sources.atmo_component().flux_model
        else:
            atmo_flux_model = None

        filename = self.output_dir + "/model_code"

        if self._detector_model_type == IceCubeDetectorModel:

            self._fit_filename = generate_stan_fit_code_hybrid_(
                filename,
                self._detector_model_type,
                ps_spec_shape,
                atmo_flux_model=atmo_flux_model,
                diffuse_bg_comp=self._sources.diffuse_component(),
                atmospheric_comp=self._sources.atmo_component(),
                theta_points=30,
                lumi_par_range=Parameter.get_parameter("luminosity").par_range,
                alpha_par_range=Parameter.get_parameter("index").par_range,
            )

        else:

            self._fit_filename = generate_stan_fit_code_(
                filename,
                self._detector_model_type,
                ps_spec_shape,
                atmo_flux_model=atmo_flux_model,
                diffuse_bg_comp=self._sources.diffuse_component(),
                atmospheric_comp=self._sources.atmo_component(),
                theta_points=30,
                lumi_par_range=Parameter.get_parameter("luminosity").par_range,
                alpha_par_range=Parameter.get_parameter("index").par_range,
            )
