import numpy as np
import os
import h5py
import logging
from astropy import units as u

from cmdstanpy import CmdStanModel

from .source.source import Sources, PointSource, icrs_to_uv
from .source.parameter import Parameter
from .source.flux_model import IsotropicDiffuseBG
from .source.cosmology import luminosity_distance
from .detector_model import DetectorModel
from .precomputation import ExposureIntegral
from .events import Events

from .backend.stan_generator import (
    StanFileGenerator,
    FunctionsContext,
    Include,
    DataContext,
    ParametersContext,
    TransformedParametersContext,
    ModelContext,
    ForLoopContext,
    IfBlockContext,
    ElseIfBlockContext,
    FunctionCall,
)
from .backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ParameterDef,
    ParameterVectorDef,
)
from .backend.expression import StringExpression


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

        # For use with plot_trace()
        self._def_var_names = ["L", "F_diff", "F_atmo", "f", "alpha"]

        self._sources = sources
        self._detector_model_type = detector_model
        self._events = events
        self._observation_time = observation_time
        self.output_dir = output_dir

        self._sources.organise()

        # Silence log output
        logger = logging.getLogger("python.backend.code_generator")
        logger.propagate = False

    def precomputation(self):

        self._exposure_integral = ExposureIntegral(
            self._sources, self._detector_model_type
        )

    def generate_stan_code(self):

        self._generate_stan_fit_code()

    def compile_stan_code(self):

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
            show_progress=False,
            **kwargs
        )

    def setup_and_run(
        self, iterations=1000, chains=1, seed=None, show_progress=False, **kwargs
    ):

        self.precomputation()
        self.generate_stan_code()
        self.compile_stan_code()
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

        Ns = np.shape(logprob)[1] - 2

        prob_each_src = []
        for lp in logprob:
            lps = []
            ps = []
            for src in range(Ns + 2):
                lps.append(np.mean(np.exp(lp[src])))
            norm = sum(lps)

            for src in range(Ns + 2):
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

        fit_inputs["Ngrid"] = len(self._exposure_integral.par_grids["index"])
        fit_inputs["alpha_grid"] = self._exposure_integral.par_grids["index"]
        fit_inputs["integral_grid"] = [
            _.value.tolist() for _ in self._exposure_integral.integral_grid
        ]
        fit_inputs["atmo_integ_val"] = self._exposure_integral.integral_fixed_vals[
            0
        ].value
        fit_inputs["T"] = self._observation_time.to(u.s).value

        fit_inputs["E_grid"] = self._exposure_integral.energy_grid.value
        fit_inputs["Pdet_grid"] = (
            np.array(self._exposure_integral.pdet_grid) + 1e-20
        )  # avoid log(0)

        fit_inputs["L_scale"] = (
            Parameter.get_parameter("luminosity").value.to(u.GeV / u.s).value
        )

        diffuse_bg = self._sources.diffuse_component()
        fit_inputs["F_diff_scale"] = diffuse_bg.flux_model.total_flux_int.value

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

        # TODO: make more flexible to different specified components?

        ps_spec_shape = self._sources.sources[0].flux_model.spectral_shape

        if self._sources.atmo_component():
            atmo_flux_model = self._sources.atmo_component().flux_model

        with StanFileGenerator(self.output_dir + "/model_code") as fit_gen:

            with FunctionsContext():
                _ = Include("utils.stan")
                _ = Include("vMF.stan")
                _ = Include("interpolation.stan")
                _ = Include("sim_functions.stan")
                dm = self._detector_model_type()

                spectrum_lpdf = ps_spec_shape.make_stan_lpdf_func("spectrum_logpdf")
                flux_fac = ps_spec_shape.make_stan_flux_conv_func("flux_conv")

                if self._sources.atmo_component:
                    atmu_nu_flux = atmo_flux_model.make_stan_function(theta_points=30)
                    atmo_flux_integral = atmo_flux_model.total_flux_int.value

            with DataContext():

                # Neutrinos
                N = ForwardVariableDef("N", "int")
                N_str = ["[", N, "]"]
                omega_det = ForwardArrayDef("omega_det", "unit_vector[3]", N_str)
                Edet = ForwardVariableDef("Edet", "vector[N]")
                Esrc_min = ForwardVariableDef("Esrc_min", "real")
                Esrc_max = ForwardVariableDef("Esrc_max", "real")

                # Sources
                Ns = ForwardVariableDef("Ns", "int")
                Ns_str = ["[", Ns, "]"]
                Ns_1p_str = ["[", Ns, "+1]"]
                Ns_2p_str = ["[", Ns, "+2]"]

                varpi = ForwardArrayDef("varpi", "unit_vector[3]", Ns_str)
                D = ForwardVariableDef("D", "vector[Ns]")
                z = ForwardVariableDef("z", "vector[Ns+1]")

                # Precomputed quantities
                Ngrid = ForwardVariableDef("Ngrid", "int")
                alpha_grid = ForwardVariableDef("alpha_grid", "vector[Ngrid]")
                integral_grid = ForwardArrayDef(
                    "integral_grid", "vector[Ngrid]", Ns_1p_str
                )
                if self._sources.atmo_component:
                    atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")
                Eg = ForwardVariableDef("E_grid", "vector[Ngrid]")
                Pg = ForwardArrayDef("Pdet_grid", "vector[Ngrid]", Ns_2p_str)

                # Inputs
                T = ForwardVariableDef("T", "real")

                # Priors
                L_scale = ForwardVariableDef("L_scale", "real")
                if self._sources.diffuse_component:
                    F_diff_scale = ForwardVariableDef("F_diff_scale", "real")
                if self._sources.atmo_component:
                    F_atmo_scale = ForwardVariableDef("F_atmo_scale", "real")
                F_tot_scale = ForwardVariableDef("F_tot_scale", "real")

            with ParametersContext():

                Lmin, Lmax = Parameter.get_parameter("luminosity").par_range
                alphamin, alphamax = Parameter.get_parameter("index").par_range

                L = ParameterDef("L", "real", Lmin, Lmax)
                F_diff = ParameterDef("F_diff", "real", 0.0, 1e-7)
                F_atmo = ParameterDef("F_atmo", "real", 0.0, 1e-7)

                alpha = ParameterDef("alpha", "real", alphamin, alphamax)

                Esrc = ParameterVectorDef("Esrc", "vector", N_str, Esrc_min, Esrc_max)

            with TransformedParametersContext():

                Fsrc = ForwardVariableDef("Fsrc", "real")
                F = ForwardVariableDef("F", "vector[Ns+2]")
                eps = ForwardVariableDef("eps", "vector[Ns+2]")

                f = ParameterDef("f", "real", 0, 1)
                Ftot = ParameterDef("Ftot", "real", 0)

                lp = ForwardArrayDef("lp", "vector[Ns+2]", N_str)
                logF = ForwardVariableDef("logF", "vector[Ns+2]")
                Nex = ForwardVariableDef("Nex", "real")
                E = ForwardVariableDef("E", "vector[N]")

                Fsrc << 0.0
                with ForLoopContext(1, Ns, "k") as k:
                    F[k] << StringExpression(
                        [L, "/ (4 * pi() * pow(", D[k], " * ", 3.086e22, ", 2))"]
                    )
                    StringExpression([F[k], "*=", flux_fac(alpha, Esrc_min, Esrc_max)])
                    StringExpression([Fsrc, "+=", F[k]])

                StringExpression("F[Ns+1]") << F_diff
                StringExpression("F[Ns+2]") << F_atmo

                Ftot << F_diff + F_atmo + Fsrc
                f << StringExpression([Fsrc, " / ", Ftot])
                logF << StringExpression(["log(", F, ")"])

                with ForLoopContext(1, N, "i") as i:
                    lp[i] << logF

                    with ForLoopContext(1, "Ns+2", "k") as k:

                        # Point source components
                        with IfBlockContext([StringExpression([k, " < ", Ns + 1])]):
                            StringExpression(
                                [
                                    lp[i][k],
                                    " += ",
                                    spectrum_lpdf(Esrc[i], alpha, Esrc_min, Esrc_max),
                                ]
                            )
                            E[i] << StringExpression([Esrc[i], " / (", 1 + z[k], ")"])
                            StringExpression(
                                [
                                    lp[i][k],
                                    " += ",
                                    dm.angular_resolution(E[i], varpi[k], omega_det[i]),
                                ]
                            )
                        # Diffuse component
                        with ElseIfBlockContext(
                            [StringExpression([k, " == ", Ns + 1])]
                        ):
                            StringExpression(
                                [
                                    lp[i][k],
                                    " += ",
                                    spectrum_lpdf(Esrc[i], alpha, Esrc_min, Esrc_max),
                                ]
                            )
                            E[i] << StringExpression([Esrc[i], " / (", 1 + z[k], ")"])
                            StringExpression(
                                [lp[i][k], " += ", np.log(1 / (4 * np.pi))]
                            )

                        # Atmospheric component
                        with ElseIfBlockContext(
                            [StringExpression([k, " == ", Ns + 2])]
                        ):
                            StringExpression(
                                [
                                    lp[i][k],
                                    " += ",
                                    FunctionCall(
                                        [
                                            atmu_nu_flux(Esrc[i], omega_det[i])
                                            / atmo_flux_integral
                                        ],
                                        "log",
                                    ),
                                ]
                            )

                            E[i] << Esrc[i]

                        # Detection effects
                        StringExpression(
                            [lp[i][k], " += ", dm.energy_resolution(E[i], Edet[i])]
                        )
                        StringExpression(
                            [
                                lp[i][k],
                                " += log(interpolate(",
                                Eg,
                                ", ",
                                Pg[k],
                                ", ",
                                E[i],
                                "))",
                            ]
                        )

                eps << FunctionCall(
                    [alpha, alpha_grid, integral_grid, atmo_integ_val, T, Ns],
                    "get_exposure_factor_atmo",
                )
                Nex << FunctionCall([F, eps], "get_Nex")

            with ModelContext():

                with ForLoopContext(1, N, "i") as i:
                    StringExpression(["target += log_sum_exp(", lp[i], ")"])
                StringExpression(["target += -", Nex])

                StringExpression([L, " ~ normal(0, ", L_scale, ")"])
                StringExpression([F_diff, " ~ normal(0, ", F_diff_scale, ")"])
                StringExpression(
                    [
                        F_atmo,
                        " ~ ",
                        FunctionCall([F_atmo_scale, 0.1 * F_atmo_scale], "normal"),
                    ]
                )
                StringExpression(
                    [
                        Ftot,
                        " ~ ",
                        FunctionCall([F_tot_scale, 0.5 * F_tot_scale], "normal"),
                    ]
                )
                StringExpression([alpha, " ~ normal(2.0, 2.0)"])

        fit_gen.generate_single_file()

        self._fit_filename = fit_gen.filename
