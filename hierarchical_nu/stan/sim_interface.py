from hierarchical_nu.stan.interface import StanInterface

from hierarchical_nu.backend.stan_generator import (
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

from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ParameterDef,
    ParameterVectorDef,
)

from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.events import TRACKS, CASCADES


class StanSimInterface(StanInterface):
    """
    For generating Stan sim code.
    """

    def __init__(
        self,
        output_file,
        sources,
        includes=["interpolation.stan", "utils.stan", "vMF.stan"],
    ):

        super().__init__(
            output_file=output_file,
            sources=sources,
            includes=includes,
        )

    def functions(self):

        with FunctionsContext():

            for include_file in self._includes:
                _ = Include(include_file)

            if self.sources.point_source:

                self._src_spectrum_rng = self._ps_spectrum.make_stan_sampling_func(
                    "src_spectrum_rng"
                )

                self._flux_conv = self._ps_spectrum.make_stan_flux_conv_func(
                    "flux_conv"
                )

            if self.sources.diffuse:

                self._diff_spectrum_rng = self._diff_spectrum.make_stan_sampling_func(
                    "diff_spectrum_rng"
                )

    def data(self):

        with DataContext():

            self._Ns = ForwardVariableDef("Ns", "int")
            self._Ns_str = ["[", self._Ns, "]"]
            self._Ns_1p_str = ["[", self._Ns, "+1]"]
            self._Ns_2p_str = ["[", self._Ns, "+2]"]

            if self.sources.diffuse:

                N_int_str = self._Ns_1p_str

            else:

                N_int_str = self._Ns_str

            self._varpi = ForwardArrayDef("varpi", "unit_vector[3]", self._Ns_str)
            self._D = ForwardVariableDef("D", "vector[Ns]")

            if self.sources.diffuse:

                self._z = ForwardVariableDef("z", "vector[Ns+1]")
                self._diff_index = ForwardVariableDef("diff_index", "real")

            else:

                self._z = ForwardVariableDef("z", "vector[Ns]")

            if self.sources.point_source:

                self._src_index = ForwardVariableDef("src_index", "real")

            self._Esrc_min = ForwardVariableDef("Esrc_min", "real")
            self._Esrc_max = ForwardVariableDef("Esrc_max", "real")

            if TRACKS in self._event_types:

                self._Emin_det_t = ForwardVariableDef("Emin_det_t", "real")
                self._aeff_t_max = ForwardVariableDef("aeff_t_max", "real")
                self._integral_grid_t = ForwardArrayDef(
                    "integral_grid_t", "vector[Ngrid]", N_int_str
                )

            if CASCADES in self._event_types:

                self._Emin_det_c = ForwardVariableDef("Emin_det_c", "real")
                self._aeff_c_max = ForwardVariableDef("aeff_c_max", "real")
                self._integral_grid_c = ForwardArrayDef(
                    "integral_grid_c", "vector[Ngrid]", N_int_str
                )

            if self.sources.point_source:

                self._L = ForwardVariableDef("L", "real")

            if self.sources.diffuse:

                self._F_diff = ForwardVariableDef("F_diff", "real")

            if self.sources.atmospheric:

                self._F_atmo = ForwardVariableDef("F_atmo", "real")

                self._atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")

                # Atmo samples
                self._N_atmo = ForwardVariableDef("N_atmo", "int")
                self._N_atmo_str = ["[", self._N_atmo, "]"]
                self._atmo_directions = ForwardArrayDef(
                    "atmo_directions", "unit_vector[3]", self._N_atmo_str
                )
                self._atmo_energies = ForwardVariableDef(
                    "atmo_energies", "vector[N_atmo]"
                )
                self._atmo_weights = ForwardVariableDef(
                    "atmo_weights", "simplex[N_atmo]"
                )

            self._v_lim = ForwardVariableDef("v_lim", "real")
            self._T = ForwardVariableDef("T", "real")

    def transformed_data(self):

        with TransformedDataContext():

            if self.sources.diffuse and self.sources.atmospheric:

                self._F = ForwardVariableDef("F", "vector[Ns+2]")

                N_tot_t = "[Ns+2]"
                N_tot_c = "[Ns+1]"

            elif self.sources.diffuse or self.sources.atmospheric:

                self._F = ForwardVariableDef("F", "vector[Ns+1]")

                N_tot_t = N_tot_c = "[Ns+1]"

            else:

                self._F = ForwardVariableDef("F", "vector[Ns]")

                N_tot_t = N_tot_c = "Ns"

            if TRACKS in self._event_types:

                self._track_type = ForwardVariableDef("track_type", "int")
                self._track_type << TRACKS

                self._w_exposure_t = ForwardVariableDef(
                    "w_exposure_t", "simplex" + N_tot_t
                )

                self._eps_t = ForwardVariableDef("eps_t", "vector" + N_tot_t)
                self._Nex_t = ForwardVariableDef("Nex_t", "real")
                self._N_t = ForwardVariableDef("N_t", "int")

            if CASCADES in self._event_types:

                self._cascade_type = ForwardVariableDef("cascade_type", "int")
                self._cascade_type << CASCADES

                self._w_exposure_c = ForwardVariableDef(
                    "w_exposure_c", "simplex" + N_tot_c
                )

                self._eps_c = ForwardVariableDef("eps_c", "vector" + N_tot_c)
                self._Nex_c = ForwardVariableDef("Nex_c", "real")
                self._N_c = ForwardVariableDef("N_c", "int")

            self._Ftot = ForwardVariableDef("Ftot", "real")
            self._Fsrc = ForwardVariableDef("Fs", "real")
            self._f = ForwardVariableDef("f", "real")
            self._N = ForwardVariableDef("N", "int")

            self._Fsrc << 0.0

            if self.sources.point_source:

                with ForLoopContext(1, self._Ns, "k") as k:
                    self._F[k] << StringExpression(
                        [
                            self._L,
                            "/ (4 * pi() * pow(",
                            self._D[k],
                            " * ",
                            3.086e22,
                            ", 2))",
                        ]
                    )
                    StringExpression(
                        [
                            self._F[k],
                            "*=",
                            self._flux_conv(
                                self._src_index, self._Esrc_min, self._Esrc_max
                            ),
                        ]
                    )
                    StringExpression([self._Fsrc, " += ", self._F[k]])

            if self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_diff

            if self.sources.atmospheric:
                StringExpression("F[Ns+2]") << self._F_atmo

            if self.sources.diffuse and self.sources.atmopheric:
                self._Ftot << self._Fsrc + self._F_diff + self._F_atmo

            elif self.sources.diffuse:
                self._Ftot << self._Fsrc + self._F_diff

            elif self.sources.atmospheric:
                self._Ftot << self._Fsrc + self._F_atmo

            else:
                self._Ftot << self._Fsrc

            self._f << StringExpression([self._Fsrc, "/", self._Ftot])
            StringExpression(['print("f: ", ', self._f, ")"])

            # Make flexible enough to handle optional source components
            if self.sources.atmospheric:
                self._eps_t << FunctionCall(
                    [
                        self._src_index,
                        self._diff_index,
                        self._src_index_grid,
                        self._diff_index_grid,
                        self._integral_grid_t,
                        self._atmo_integ_val,
                        self._T,
                        self._Ns,
                    ],
                    "get_exposure_factor_atmo",
                )


def generate_atmospheric_sim_code_(filename, atmo_flux_model, theta_points=50):

    with StanFileGenerator(filename) as atmo_gen:

        with FunctionsContext():
            _ = Include("interpolation.stan")
            _ = Include("utils.stan")

            # Increasing theta points too much makes compilation very slow
            # Could switch to passing array as data if problematic
            atmu_nu_flux = atmo_flux_model.make_stan_function(theta_points=theta_points)

        with DataContext():
            Esrc_min = ForwardVariableDef("Esrc_min", "real")
            Esrc_max = ForwardVariableDef("Esrc_max", "real")

            cosz_min = ForwardVariableDef("cosz_min", "real")
            cosz_max = ForwardVariableDef("cosz_max", "real")

        with ParametersContext():
            # Simulate from Emin and cosz bounds for efficiency
            energy = ParameterDef("energy", "real", Esrc_min, Esrc_max)
            coszen = ParameterDef("coszen", "real", cosz_min, cosz_max)
            phi = ParameterDef("phi", "real", 0, 2 * np.pi)

        with TransformedParametersContext():
            omega = ForwardVariableDef("omega", "unit_vector[3]")
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

        return atmo_gen.filename
