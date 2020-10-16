import numpy as np
import os

import stan_utility

from .detector_model import DetectorModel
from .precomputation import ExposureIntegral
from .source.source import Sources, PointSource
from .source.flux_model import IsotropicDiffuseBG
from .source.atmospheric_flux import AtmosphericNuMuFlux

from .backend.stan_generator import (
    StanFileGenerator,
    FunctionsContext,
    Include,
    DataContext,
    TransformedDataContext,
    ParametersContext,
    TransformedParametersContext,
    GeneratedQuantitiesContext,
    ForLoopContext,
    IfBlockContext,
    ElseIfBlockContext,
    ElseBlockContext,
    WhileLoopContext,
    ModelContext,
    FunctionCall,
)

from .backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ParameterDef,
    ParameterVectorDef,
)
from .backend.expression import StringExpression
from .backend.parameterizations import DistributionMode


class Simulation:
    """
    To set up and run simulations.
    """

    def __init__(self, sources: Sources, detector_model: DetectorModel):
        """
        To set up and run simulations.
        """

        self._sources = sources
        self._detector_model_type = detector_model

        # Check source components
        source_types = [type(s) for s in self._sources.sources]
        flux_types = [type(s.flux_model) for s in self._sources.sources]
        self._point_source_comp = PointSource in source_types
        self._diffuse_bg_comp = IsotropicDiffuseBG in flux_types
        self._atmospheric_comp = AtmosphericNuMuFlux in flux_types

        # TODO: Create stan cache if not done

        # TODO: Create output dir if not done
        self.output_directory = "stan_files/"

    def precomputation(self):
        """
        Run the necessary precomputation
        """

        self._exposure_integral = ExposureIntegral(
            self._sources, self._detector_model_type
        )

    def generate_stan_code(self):

        if self._atmospheric_comp:
            self._generate_atmospheric_sim_code()

        self._generate_main_sim_code()

    def compile_stan_code(self):

        this_dir = os.path.abspath("")
        include_paths = [os.path.join(this_dir, self.output_directory)]
        self._atmo_sim = stan_utility.compile_model(
            filename=self._atmo_sim_filename,
            include_paths=include_paths,
            model_name="atmo_sim",
        )
        self._main_sim = stan_utility.compile_model(
            filename=self._main_sim_filename,
            include_paths=include_paths,
            model_name="main_sim",
        )

    def _generate_atmospheric_sim_code(self):

        for s in self._sources.sources:
            if type(s.flux_model) == AtmosphericNuMuFlux:
                atmo_flux_model = s.flux_model

        with StanFileGenerator(self.output_directory + "atmo_gen") as atmo_gen:

            with FunctionsContext():
                _ = Include("utils.stan")
                _ = Include("interpolation.stan")

                # Increasing theta points too much makes compilation very slow
                # Could switch to passing array as data if problematic
                atmu_nu_flux = atmo_flux_model.make_stan_function(theta_points=30)

            with DataContext():
                Edet_min = ForwardVariableDef("Edet_min", "real")
                Esrc_max = ForwardVariableDef("Esrc_max", "real")

                cosz_min = ForwardVariableDef("cosz_min", "real")
                cosz_max = ForwardVariableDef("cosz_max", "real")

            with ParametersContext():
                # Simulate from Edet_min and cosz bounds for efficiency
                energy = ParameterDef("energy", "real", Edet_min, Esrc_max)
                coszen = ParameterDef("coszen", "real", cosz_min, cosz_max)
                phi = ParameterDef("phi", "real", 0, 2 * np.pi)

            with TransformedParametersContext():
                omega = ForwardVariableDef("omega", "vector[3]")
                zen = ForwardVariableDef("zen", "real")
                theta = ForwardVariableDef("theta", "real")

                zen << FunctionCall([coszen], "acos")
                theta << FunctionCall([], "pi") - zen

                omega[1] << FunctionCall([theta], "sin") * FunctionCall([phi], "cos")
                omega[2] << FunctionCall([theta], "sin") * FunctionCall([phi], "sin")
                omega[3] << FunctionCall([theta], "cos")

            with ModelContext():

                logflux = FunctionCall([atmu_nu_flux(energy, omega)], "log")
                StringExpression(["target += ", logflux])

        atmo_gen.generate_single_file()

        self._atmo_sim_filename = atmo_gen.filename

    def _generate_main_sim_code(self):

        ps_spec_shape = self._sources.sources[0].flux_model.spectral_shape

        with StanFileGenerator(self.output_directory + "sim_code") as sim_gen:

            with FunctionsContext():
                _ = Include("utils.stan")
                _ = Include("vMF.stan")
                _ = Include("interpolation.stan")
                _ = Include("sim_functions.stan")

                spectrum_rng = ps_spec_shape.make_stan_sampling_func("spectrum_rng")
                flux_fac = ps_spec_shape.make_stan_flux_conv_func("flux_conv")

            with DataContext():

                # Sources
                Ns = ForwardVariableDef("Ns", "int")
                Ns_str = ["[", Ns, "]"]
                Ns_1p_str = ["[", Ns, "+1]"]

                varpi = ForwardArrayDef("varpi", "unit_vector[3]", Ns_str)
                D = ForwardVariableDef("D", "vector[Ns]")
                z = ForwardVariableDef("z", "vector[Ns+1]")

                # Energies
                alpha = ForwardVariableDef("alpha", "real")
                Edet_min = ForwardVariableDef("Edet_min", "real")
                Esrc_min = ForwardVariableDef("Esrc_min", "real")
                Esrc_max = ForwardVariableDef("Esrc_max", "real")

                # Luminosity/ diffuse flux
                L = ForwardVariableDef("L", "real")
                F_diff = ForwardVariableDef("F_diff", "real")
                F_atmo = ForwardVariableDef("F_atmo", "real")

                # Precomputed quantities
                Ngrid = ForwardVariableDef("Ngrid", "int")
                alpha_grid = ForwardVariableDef("alpha_grid", "vector[Ngrid]")
                integral_grid = ForwardArrayDef(
                    "integral_grid", "vector[Ngrid]", Ns_1p_str
                )
                atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")
                aeff_max = ForwardVariableDef("aeff_max", "real")

                v_lim = ForwardVariableDef("v_lim", "real")
                T = ForwardVariableDef("T", "real")

                # Atmo samples
                N_atmo = ForwardVariableDef("N_atmo", "int")
                N_atmo_str = ["[", N_atmo, "]"]
                atmo_directions = ForwardArrayDef(
                    "atmo_directions", "unit_vector[3]", N_atmo_str
                )
                atmo_energies = ForwardVariableDef("atmo_energies", "vector[N_atmo]")
                atmo_weights = ForwardVariableDef("atmo_weights", "simplex[N_atmo]")

            with TransformedDataContext():
                F = ForwardVariableDef("F", "vector[Ns+2]")
                Ftot = ForwardVariableDef("Ftot", "real")
                Fsrc = ForwardVariableDef("Fs", "real")
                f = ForwardVariableDef("f", "real")
                w_exposure = ForwardVariableDef("w_exposure", "simplex[Ns+2]")
                Nex = ForwardVariableDef("Nex", "real")
                N = ForwardVariableDef("N", "int")
                eps = ForwardVariableDef("eps", "vector[Ns+2]")

                Fsrc << 0.0
                with ForLoopContext(1, Ns, "k") as k:
                    F[k] << StringExpression(
                        [L, "/ (4 * pi() * pow(", D[k], " * ", 3.086e22, ", 2))"]
                    )
                    StringExpression([F[k], "*=", flux_fac(alpha, Esrc_min, Esrc_max)])
                    StringExpression([Fsrc, " += ", F[k]])

                StringExpression("F[Ns+1]") << F_diff
                StringExpression("F[Ns+2]") << F_atmo

                Ftot << Fsrc + F_diff + F_atmo
                f << StringExpression([Fsrc, "/", Ftot])
                StringExpression(['print("f: ", ', f, ")"])

                eps << StringExpression(
                    [
                        "get_exposure_factor(",
                        alpha,
                        ", ",
                        alpha_grid,
                        ", ",
                        integral_grid,
                        ", ",
                        atmo_integ_val,
                        ", ",
                        T,
                        ", ",
                        Ns,
                        ")",
                    ]
                )
                Nex << StringExpression(["get_Nex(", F, ", ", eps, ")"])
                w_exposure << StringExpression(
                    ["get_exposure_weights(", F, ", ", eps, ")"]
                )
                N << StringExpression(["poisson_rng(", Nex, ")"])
                StringExpression(["print(", w_exposure, ")"])
                StringExpression(["print(", Ngrid, ")"])
                StringExpression(["print(", Nex, ")"])
                StringExpression(["print(", N, ")"])

            with GeneratedQuantitiesContext():
                dm_rng = self._detector_model_type(mode=DistributionMode.RNG)
                dm_pdf = self._detector_model_type(mode=DistributionMode.PDF)

                N_str = ["[", N, "]"]
                lam = ForwardArrayDef("Lambda", "int", N_str)
                omega = ForwardVariableDef("omega", "unit_vector[3]")

                Esrc = ForwardVariableDef("Esrc", "vector[N]")
                E = ForwardVariableDef("E", "vector[N]")
                Edet = ForwardVariableDef("Edet", "vector[N]")

                atmo_index = ForwardVariableDef("atmo_index", "int")
                cosz = ForwardArrayDef("cosz", "real", N_str)
                Pdet = ForwardArrayDef("Pdet", "real", N_str)
                accept = ForwardVariableDef("accept", "int")
                detected = ForwardVariableDef("detected", "int")
                ntrials = ForwardVariableDef("ntrials", "int")
                prob = ForwardVariableDef("prob", "simplex[2]")

                event = ForwardArrayDef("event", "unit_vector[3]", N_str)
                Nex_sim = ForwardVariableDef("Nex_sim", "real")

                Nex_sim << Nex

                with ForLoopContext(1, N, "i") as i:

                    lam[i] << FunctionCall([w_exposure], "categorical_rng")

                    accept << 0
                    detected << 0
                    ntrials << 0

                    with WhileLoopContext([StringExpression([accept != 1])]):

                        # Sample position
                        with IfBlockContext([StringExpression([lam[i], " <= ", Ns])]):
                            omega << varpi[lam[i]]
                        with ElseIfBlockContext(
                            [StringExpression([lam[i], " == ", Ns + 1])]
                        ):
                            omega << FunctionCall([1, v_lim], "sphere_lim_rng")
                        with ElseIfBlockContext(
                            [StringExpression([lam[i], " == ", Ns + 2])]
                        ):
                            atmo_index << FunctionCall(
                                [atmo_weights], "categorical_rng"
                            )
                            omega << atmo_directions[atmo_index]

                        cosz[i] << FunctionCall(
                            [FunctionCall([omega], "omega_to_zenith")], "cos"
                        )
                        # Sample energy
                        with IfBlockContext(
                            [StringExpression([lam[i], " <= ", Ns + 1])]
                        ):
                            Esrc[i] << spectrum_rng(alpha, Esrc_min, Esrc_max)
                            E[i] << Esrc[i] / (1 + z[lam[i]])
                        with ElseIfBlockContext(
                            [StringExpression([lam[i], " == ", Ns + 2])]
                        ):
                            E[i] << atmo_energies[atmo_index]

                        # Test against Aeff
                        with IfBlockContext([StringExpression([cosz[i], ">= 0.1"])]):
                            Pdet[i] << 0
                        with ElseBlockContext():
                            Pdet[i] << dm_pdf.effective_area(E[i], omega) / aeff_max

                        Edet[i] << 10 ** dm_rng.energy_resolution(E[i])

                        prob[1] << Pdet[i]
                        prob[2] << 1 - Pdet[i]
                        StringExpression([ntrials, " += ", 1])

                        with IfBlockContext([StringExpression([ntrials, "< 1000000"])]):
                            detected << FunctionCall([prob], "categorical_rng")
                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            "(",
                                            Edet[i],
                                            " >= ",
                                            Edet_min,
                                            ") && (",
                                            detected == 1,
                                            ")",
                                        ]
                                    )
                                ]
                            ):
                                accept << 1
                        with ElseBlockContext():
                            accept << 1
                            StringExpression(
                                ['print("problem component: ", ', lam[i], ");\n"]
                            )

                    # Detection effects
                    event[i] << dm_rng.angular_resolution(E[i], omega)

        sim_gen.generate_single_file()

        self._main_sim_filename = sim_gen.filename
