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

            self._varpi = ForwardArrayDef("varpi", "unit_vector[3]", self._Ns_str)
            self._D = ForwardVariableDef("D", "vector[Ns]")

            if self.sources.diffuse:

                self._z = ForwardVariableDef("z", "vector[Ns+1]")

            else:

                self._z = ForwardVariableDef("z", "vector[Ns]")

            # Energies
            self._src_index = ForwardVariableDef("src_index", "real")
            self._diff_index = ForwardVariableDef("diff_index", "real")
            self._Emin_det = ForwardVariableDef("Emin_det", "real")
            self._Esrc_min = ForwardVariableDef("Esrc_min", "real")
            self._Esrc_max = ForwardVariableDef("Esrc_max", "real")

            # Luminosity/diffuse fluxes
            if self.sources.point_source:

                self._L = ForwardVariableDef("L", "real")

            if self.sources.diffuse:

                self._F_diff = ForwardVariableDef("F_diff", "real")

            if self.sources.atmospheric:

                self._F_atmo = ForwardVariableDef("F_atmo", "real")


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


def generate_main_sim_code_hybrid_(
    filename,
    ps_spec_shape,
    diff_spec_shape,
    detector_model_type,
    diffuse_bg_comp=True,
    atmospheric_comp=True,
):

    with StanFileGenerator(filename) as sim_gen:

        with FunctionsContext():
            _ = Include("interpolation.stan")
            _ = Include("utils.stan")
            _ = Include("vMF.stan")

            src_spectrum_rng = ps_spec_shape.make_stan_sampling_func("src_spectrum_rng")
            diff_spectrum_rng = diff_spec_shape.make_stan_sampling_func(
                "diff_spectrum_rng"
            )
            flux_fac = ps_spec_shape.make_stan_flux_conv_func("flux_conv")

        with DataContext():

            # Sources
            Ns = ForwardVariableDef("Ns", "int")
            Ns_str = ["[", Ns, "]"]
            Ns_1p_str = ["[", Ns, "+1]"]

            varpi = ForwardArrayDef("varpi", "unit_vector[3]", Ns_str)
            D = ForwardVariableDef("D", "vector[Ns]")
            if diffuse_bg_comp:
                z = ForwardVariableDef("z", "vector[Ns+1]")
            else:
                z = ForwardVariableDef("z", "vector[Ns]")

            # Energies
            src_index = ForwardVariableDef("src_index", "real")
            diff_index = ForwardVariableDef("diff_index", "real")
            Emin_det_tracks = ForwardVariableDef("Emin_det_tracks", "real")
            Emin_det_cascades = ForwardVariableDef("Emin_det_cascades", "real")
            Esrc_min = ForwardVariableDef("Esrc_min", "real")
            Esrc_max = ForwardVariableDef("Esrc_max", "real")

            # Luminosity/ diffuse flux
            L = ForwardVariableDef("L", "real")
            F_diff = ForwardVariableDef("F_diff", "real")

            # Precomputed quantities
            Ngrid = ForwardVariableDef("Ngrid", "int")
            src_index_grid = ForwardVariableDef("src_index_grid", "vector[Ngrid]")
            diff_index_grid = ForwardVariableDef("diff_index_grid", "vector[Ngrid]")

            if diffuse_bg_comp:
                integral_grid_t = ForwardArrayDef(
                    "integral_grid_t", "vector[Ngrid]", Ns_1p_str
                )
                integral_grid_c = ForwardArrayDef(
                    "integral_grid_c", "vector[Ngrid]", Ns_1p_str
                )
            else:
                integral_grid_t = ForwardArrayDef(
                    "integral_grid_t", "vector[Ngrid]", Ns_str
                )
                integral_grid_c = ForwardArrayDef(
                    "integral_grid_c", "vector[Ngrid]", Ns_str
                )

            aeff_t_max = ForwardVariableDef("aeff_t_max", "real")
            aeff_c_max = ForwardVariableDef("aeff_c_max", "real")

            v_lim = ForwardVariableDef("v_lim", "real")
            T = ForwardVariableDef("T", "real")

            if atmospheric_comp:
                F_atmo = ForwardVariableDef("F_atmo", "real")
                atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")

                # Atmo samples
                N_atmo = ForwardVariableDef("N_atmo", "int")
                N_atmo_str = ["[", N_atmo, "]"]
                atmo_directions = ForwardArrayDef(
                    "atmo_directions", "unit_vector[3]", N_atmo_str
                )
                atmo_energies = ForwardVariableDef("atmo_energies", "vector[N_atmo]")
                atmo_weights = ForwardVariableDef("atmo_weights", "simplex[N_atmo]")

        with TransformedDataContext():

            if diffuse_bg_comp and atmospheric_comp:
                F = ForwardVariableDef("F", "vector[Ns+2]")
                w_exposure_t = ForwardVariableDef("w_exposure_t", "simplex[Ns+2]")
                w_exposure_c = ForwardVariableDef("w_exposure_c", "simplex[Ns+1]")
                eps_t = ForwardVariableDef("eps_t", "vector[Ns+2]")
                eps_c = ForwardVariableDef("eps_c", "vector[Ns+1]")
            elif diffuse_bg_comp or atmospheric_comp:
                F = ForwardVariableDef("F", "vector[Ns+1]")
                w_exposure_t = ForwardVariableDef("w_exposure_t", "simplex[Ns+1]")
                w_exposure_c = ForwardVariableDef("w_exposure_c", "simplex[Ns+1]")
                eps_t = ForwardVariableDef("eps_t", "vector[Ns+1]")
                eps_c = ForwardVariableDef("eps_c", "vector[Ns+1]")
            else:
                F = ForwardVariableDef("F", "vector[Ns]")
                w_exposure_t = ForwardVariableDef("w_exposure_t", "simplex[Ns]")
                w_exposure_c = ForwardVariableDef("w_exposure_c", "simplex[Ns]")
                eps_t = ForwardVariableDef("eps_t", "vector[Ns]")
                eps_c = ForwardVariableDef("eps_c", "vector[Ns]")

            track_type = ForwardVariableDef("track_type", "int")
            cascade_type = ForwardVariableDef("cascade_type", "int")

            track_type << TRACKS
            cascade_type << CASCADES

            Ftot = ForwardVariableDef("Ftot", "real")
            Fsrc = ForwardVariableDef("Fs", "real")
            f = ForwardVariableDef("f", "real")
            Nex_t = ForwardVariableDef("Nex_t", "real")
            Nex_c = ForwardVariableDef("Nex_c", "real")
            N_t = ForwardVariableDef("N_t", "int")
            N_c = ForwardVariableDef("N_c", "int")
            N = ForwardVariableDef("N", "int")

            Fsrc << 0.0
            with ForLoopContext(1, Ns, "k") as k:
                F[k] << StringExpression(
                    [L, "/ (4 * pi() * pow(", D[k], " * ", 3.086e22, ", 2))"]
                )
                StringExpression([F[k], "*=", flux_fac(src_index, Esrc_min, Esrc_max)])
                StringExpression([Fsrc, " += ", F[k]])

            if diffuse_bg_comp:
                StringExpression("F[Ns+1]") << F_diff

            if atmospheric_comp:
                StringExpression("F[Ns+2]") << F_atmo

            if diffuse_bg_comp and atmospheric_comp:
                Ftot << Fsrc + F_diff + F_atmo
            elif diffuse_bg_comp:
                Ftot << Fsrc + F_diff
            else:
                Ftot << Fsrc

            f << StringExpression([Fsrc, "/", Ftot])
            StringExpression(['print("f: ", ', f, ")"])

            if atmospheric_comp:
                eps_t << FunctionCall(
                    [
                        src_index,
                        diff_index,
                        src_index_grid,
                        diff_index_grid,
                        integral_grid_t,
                        atmo_integ_val,
                        T,
                        Ns,
                    ],
                    "get_exposure_factor_atmo",
                )
            else:
                eps_t << FunctionCall(
                    [
                        src_index,
                        diff_index,
                        src_index_grid,
                        diff_index_grid,
                        integral_grid_t,
                        T,
                        Ns,
                    ],
                    "get_exposure_factor",
                )
            eps_c << FunctionCall(
                [
                    src_index,
                    diff_index,
                    src_index_grid,
                    diff_index_grid,
                    integral_grid_c,
                    T,
                    Ns,
                ],
                "get_exposure_factor",
            )

            Nex_t << FunctionCall([F, eps_t], "get_Nex")
            w_exposure_t << FunctionCall([F, eps_t], "get_exposure_weights")

            Nex_c << FunctionCall([F, eps_c], "get_Nex")
            w_exposure_c << FunctionCall([F, eps_c], "get_exposure_weights")

            N_t << StringExpression(["poisson_rng(", Nex_t, ")"])
            N_c << StringExpression(["poisson_rng(", Nex_c, ")"])

            N << N_t + N_c
            StringExpression(['print("Ngrid: ", ', Ngrid, ")"])

        with GeneratedQuantitiesContext():

            # Load detector models
            dm_rng = collections.OrderedDict()
            dm_pdf = collections.OrderedDict()

            for event_type in detector_model_type.event_types:

                dm_rng[event_type] = detector_model_type(
                    mode=DistributionMode.RNG,
                    event_type=event_type,
                )
                dm_pdf[event_type] = detector_model_type(
                    mode=DistributionMode.PDF, event_type=event_type
                )

            N_str = ["[", N, "]"]
            lam = ForwardArrayDef("Lambda", "int", N_str)
            omega = ForwardVariableDef("omega", "unit_vector[3]")

            Esrc = ForwardVariableDef("Esrc", "vector[N]")
            E = ForwardVariableDef("E", "vector[N]")
            Edet = ForwardVariableDef("Edet", "vector[N]")

            if atmospheric_comp:
                atmo_index = ForwardVariableDef("atmo_index", "int")
            cosz = ForwardArrayDef("cosz", "real", N_str)
            Pdet = ForwardArrayDef("Pdet", "real", N_str)
            accept = ForwardVariableDef("accept", "int")
            detected = ForwardVariableDef("detected", "int")
            ntrials = ForwardVariableDef("ntrials", "int")
            prob = ForwardVariableDef("prob", "simplex[2]")

            event = ForwardArrayDef("event", "unit_vector[3]", N_str)
            Nex_t_sim = ForwardVariableDef("Nex_t_sim", "real")
            Nex_c_sim = ForwardVariableDef("Nex_c_sim", "real")

            event_type = ForwardVariableDef("event_type", "vector[N]")
            kappa = ForwardVariableDef("kappa", "vector[N]")

            Nex_t_sim << Nex_t
            Nex_c_sim << Nex_c

            # Tracks
            with ForLoopContext(1, N_t, "i") as i:

                event_type[i] << track_type

                lam[i] << FunctionCall([w_exposure_t], "categorical_rng")

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
                    if atmospheric_comp:
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
                    with IfBlockContext([StringExpression([lam[i], " <= ", Ns])]):
                        Esrc[i] << src_spectrum_rng(src_index, Esrc_min, Esrc_max)
                        E[i] << Esrc[i] / (1 + z[lam[i]])

                    if diffuse_bg_comp:
                        with ElseIfBlockContext(
                            [StringExpression([lam[i], " == ", Ns + 1])]
                        ):
                            Esrc[i] << diff_spectrum_rng(diff_index, Esrc_min, Esrc_max)
                            E[i] << Esrc[i] / (1 + z[lam[i]])

                    if atmospheric_comp:
                        with ElseIfBlockContext(
                            [StringExpression([lam[i], " == ", Ns + 2])]
                        ):
                            E[i] << atmo_energies[atmo_index]

                    # Test against Aeff
                    if detector_model_type == NorthernTracksDetectorModel:

                        with IfBlockContext([StringExpression([cosz[i], ">= 0.1"])]):
                            Pdet[i] << 0
                        with ElseBlockContext():
                            Pdet[i] << dm_pdf["tracks"].effective_area(
                                E[i], omega
                            ) / aeff_t_max

                    else:

                        Pdet[i] << dm_pdf["tracks"].effective_area(
                            E[i], omega
                        ) / aeff_t_max

                    Edet[i] << 10 ** dm_rng["tracks"].energy_resolution(E[i])

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
                                        Emin_det_tracks,
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
                event[i] << dm_rng["tracks"].angular_resolution(E[i], omega)
                kappa[i] << dm_rng["tracks"].angular_resolution.kappa()

            # Cascades
            with ForLoopContext("N_t+1", N, "i") as i:

                event_type[i] << cascade_type

                lam[i] << FunctionCall([w_exposure_c], "categorical_rng")

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
                        omega << FunctionCall([1, 0], "sphere_lim_rng")

                    cosz[i] << FunctionCall(
                        [FunctionCall([omega], "omega_to_zenith")], "cos"
                    )

                    # Sample energy
                    with IfBlockContext([StringExpression([lam[i], " <= ", Ns])]):
                        Esrc[i] << src_spectrum_rng(src_index, Esrc_min, Esrc_max)
                        E[i] << Esrc[i] / (1 + z[lam[i]])

                    if diffuse_bg_comp:
                        with ElseIfBlockContext(
                            [StringExpression([lam[i], " == ", Ns + 1])]
                        ):
                            Esrc[i] << diff_spectrum_rng(diff_index, Esrc_min, Esrc_max)
                            E[i] << Esrc[i] / (1 + z[lam[i]])

                    # Test against Aeff
                    Pdet[i] << dm_pdf["cascades"].effective_area(
                        E[i], omega
                    ) / aeff_c_max

                    Edet[i] << 10 ** dm_rng["cascades"].energy_resolution(E[i])

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
                                        Emin_det_cascades,
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
                event[i] << dm_rng["cascades"].angular_resolution(E[i], omega)
                kappa[i] << dm_rng["cascades"].angular_resolution.kappa()

    sim_gen.generate_single_file()

    return sim_gen.filename
