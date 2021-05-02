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


class StanFitInterface(StanInterface):
    """
    For generating Stan fit code.
    """

    def __init__(self, output_file):

        super().__init__(output_file=output_file)


def generate_stan_fit_code_hybrid_(
    filename,
    detector_model_type,
    ps_spec_shape,
    diff_spec_shape,
    atmo_flux_model=None,
    diffuse_bg_comp=True,
    atmospheric_comp=True,
    theta_points=30,
    lumi_par_range=(0, 1e60),
    src_index_par_range=(1.0, 4.0),
    diff_index_par_range=(1.0, 4.0),
):

    with StanFileGenerator(filename) as fit_gen:

        with FunctionsContext():
            _ = Include("interpolation.stan")
            _ = Include("utils.stan")
            _ = Include("vMF.stan")

            dm = collections.OrderedDict()

            for event_type in detector_model_type.event_types:
                dm[event_type] = detector_model_type(event_type=event_type)

            src_spectrum_lpdf = ps_spec_shape.make_stan_lpdf_func("src_spectrum_logpdf")
            diff_spectrum_lpdf = diff_spec_shape.make_stan_lpdf_func(
                "diff_spectrum_logpdf"
            )
            flux_fac = ps_spec_shape.make_stan_flux_conv_func("flux_conv")

            if atmospheric_comp:
                atmu_nu_flux = atmo_flux_model.make_stan_function(
                    theta_points=theta_points
                )
                atmo_flux_integral = atmo_flux_model.total_flux_int.value

        with DataContext():

            # Neutrinos
            N = ForwardVariableDef("N", "int")
            N_str = ["[", N, "]"]
            omega_det = ForwardArrayDef("omega_det", "unit_vector[3]", N_str)
            Edet = ForwardVariableDef("Edet", "vector[N]")
            event_type = ForwardVariableDef("event_type", "vector[N]")
            kappa = ForwardVariableDef("kappa", "vector[N]")
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
            src_index_grid = ForwardVariableDef("src_index_grid", "vector[Ngrid]")
            diff_index_grid = ForwardVariableDef("diff_index_grid", "vector[Ngrid]")

            integral_grid_t = ForwardArrayDef(
                "integral_grid_t", "vector[Ngrid]", Ns_1p_str
            )
            integral_grid_c = ForwardArrayDef(
                "integral_grid_c", "vector[Ngrid]", Ns_1p_str
            )

            Eg = ForwardVariableDef("E_grid", "vector[Ngrid]")

            if atmospheric_comp:
                Pg_t = ForwardArrayDef("Pdet_grid_t", "vector[Ngrid]", Ns_2p_str)
                Pg_c = ForwardArrayDef("Pdet_grid_c", "vector[Ngrid]", Ns_2p_str)
            else:
                Pg_t = ForwardArrayDef("Pdet_grid_t", "vector[Ngrid]", Ns_1p_str)
                Pg_c = ForwardArrayDef("Pdet_grid_c", "vector[Ngrid]", Ns_1p_str)

            # Inputs
            T = ForwardVariableDef("T", "real")

            # Priors
            L_scale = ForwardVariableDef("L_scale", "real")
            if diffuse_bg_comp:
                F_diff_scale = ForwardVariableDef("F_diff_scale", "real")
            if atmospheric_comp:
                atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")
                F_atmo_scale = ForwardVariableDef("F_atmo_scale", "real")
            F_tot_scale = ForwardVariableDef("F_tot_scale", "real")

        with TransformedDataContext():

            track_type = ForwardVariableDef("track_type", "int")
            cascade_type = ForwardVariableDef("cascade_type", "int")

            track_type << TRACKS
            cascade_type << CASCADES

        with ParametersContext():

            Lmin, Lmax = lumi_par_range
            src_index_min, src_index_max = src_index_par_range
            diff_index_min, diff_index_max = diff_index_par_range

            L = ParameterDef("L", "real", Lmin, Lmax)
            F_diff = ParameterDef("F_diff", "real", 0.0, 1e-6)

            if atmospheric_comp:
                F_atmo = ParameterDef("F_atmo", "real", 0.0, 1e-6)

            src_index = ParameterDef("src_index", "real", src_index_min, src_index_max)
            diff_index = ParameterDef(
                "diff_index", "real", diff_index_min, diff_index_max
            )

            Esrc = ParameterVectorDef("Esrc", "vector", N_str, Esrc_min, Esrc_max)

        with TransformedParametersContext():

            Fsrc = ForwardVariableDef("Fsrc", "real")

            if diffuse_bg_comp and atmospheric_comp:
                F = ForwardVariableDef("F", "vector[Ns+2]")
                eps_t = ForwardVariableDef("eps_t", "vector[Ns+2]")
                eps_c = ForwardVariableDef("eps_c", "vector[Ns+1]")
                lp = ForwardArrayDef("lp", "vector[Ns+2]", N_str)
                logF = ForwardVariableDef("logF", "vector[Ns+2]")
                n_comps_max = "Ns+2"
            elif diffuse_bg_comp or atmospheric_comp:
                F = ForwardVariableDef("F", "vector[Ns+1]")
                eps_t = ForwardVariableDef("eps_t", "vector[Ns+1]")
                eps_c = ForwardVariableDef("eps_c", "vector[Ns+1]")
                lp = ForwardArrayDef("lp", "vector[Ns+1]", N_str)
                logF = ForwardVariableDef("logF", "vector[Ns+1]")
                n_comps_max = "Ns+1"
            else:
                F = ForwardVariableDef("F", "vector[Ns]")
                eps_t = ForwardVariableDef("eps_t", "vector[Ns]")
                eps_c = ForwardVariableDef("eps_c", "vector[Ns]")
                lp = ForwardArrayDef("lp", "vector[Ns]", N_str)
                logF = ForwardVariableDef("logF", "vector[Ns]")
                n_comps_max = "Ns"

            f = ParameterDef("f", "real", 0, 1)
            Ftot = ParameterDef("Ftot", "real", 0)

            Nex_t = ForwardVariableDef("Nex_t", "real")
            Nex_c = ForwardVariableDef("Nex_c", "real")
            Nex = ForwardVariableDef("Nex", "real")
            E = ForwardVariableDef("E", "vector[N]")

            Fsrc << 0.0
            with ForLoopContext(1, Ns, "k") as k:
                F[k] << StringExpression(
                    [L, "/ (4 * pi() * pow(", D[k], " * ", 3.086e22, ", 2))"]
                )
                StringExpression([F[k], "*=", flux_fac(src_index, Esrc_min, Esrc_max)])
                StringExpression([Fsrc, "+=", F[k]])
            StringExpression("F[Ns+1]") << F_diff

            if atmospheric_comp:
                StringExpression("F[Ns+2]") << F_atmo

            if atmospheric_comp and diffuse_bg_comp:
                Ftot << F_diff + F_atmo + Fsrc
            if diffuse_bg_comp and not atmospheric_comp:
                Ftot << F_diff + Fsrc

            f << StringExpression([Fsrc, " / ", Ftot])
            logF << StringExpression(["log(", F, ")"])

            with ForLoopContext(1, N, "i") as i:
                lp[i] << logF

                with IfBlockContext(
                    [StringExpression([event_type[i], " == ", track_type])]
                ):

                    with ForLoopContext(1, n_comps_max, "k") as k:

                        # Point source components
                        with IfBlockContext([StringExpression([k, " < ", Ns + 1])]):
                            StringExpression(
                                [
                                    lp[i][k],
                                    " += ",
                                    src_spectrum_lpdf(
                                        Esrc[i], src_index, Esrc_min, Esrc_max
                                    ),
                                ]
                            )
                            E[i] << StringExpression([Esrc[i], " / (", 1 + z[k], ")"])

                            StringExpression(
                                [
                                    lp[i][k],
                                    " += vMF_lpdf(",
                                    omega_det[i],
                                    " | ",
                                    varpi[k],
                                    ", ",
                                    kappa[i],
                                    ")",
                                ]
                            )

                        if diffuse_bg_comp:
                            # Diffuse component
                            with ElseIfBlockContext(
                                [StringExpression([k, " == ", Ns + 1])]
                            ):
                                StringExpression(
                                    [
                                        lp[i][k],
                                        " += ",
                                        diff_spectrum_lpdf(
                                            Esrc[i], diff_index, Esrc_min, Esrc_max
                                        ),
                                    ]
                                )
                                E[i] << StringExpression(
                                    [Esrc[i], " / (", 1 + z[k], ")"]
                                )
                                StringExpression(
                                    [lp[i][k], " += ", np.log(1 / (4 * np.pi))]
                                )

                        if atmospheric_comp:
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
                            [
                                lp[i][k],
                                " += ",
                                dm["tracks"].energy_resolution(E[i], Edet[i]),
                            ]
                        )
                        StringExpression(
                            [
                                lp[i][k],
                                " += log(interpolate(",
                                Eg,
                                ", ",
                                Pg_t[k],
                                ", ",
                                E[i],
                                "))",
                            ]
                        )

                with ElseIfBlockContext(
                    [StringExpression([event_type[i], " == ", cascade_type])]
                ):

                    with ForLoopContext(1, n_comps_max, "k") as k:

                        # Point source components
                        with IfBlockContext([StringExpression([k, " < ", Ns + 1])]):
                            StringExpression(
                                [
                                    lp[i][k],
                                    " += ",
                                    src_spectrum_lpdf(
                                        Esrc[i], src_index, Esrc_min, Esrc_max
                                    ),
                                ]
                            )
                            E[i] << StringExpression([Esrc[i], " / (", 1 + z[k], ")"])

                            StringExpression(
                                [
                                    lp[i][k],
                                    " += vMF_lpdf(",
                                    omega_det[i],
                                    " | ",
                                    varpi[k],
                                    ", ",
                                    kappa[i],
                                    ")",
                                ]
                            )

                        if diffuse_bg_comp:
                            # Diffuse component
                            with ElseIfBlockContext(
                                [StringExpression([k, " == ", Ns + 1])]
                            ):
                                StringExpression(
                                    [
                                        lp[i][k],
                                        " += ",
                                        diff_spectrum_lpdf(
                                            Esrc[i], diff_index, Esrc_min, Esrc_max
                                        ),
                                    ]
                                )
                                E[i] << StringExpression(
                                    [Esrc[i], " / (", 1 + z[k], ")"]
                                )
                                StringExpression(
                                    [lp[i][k], " += ", np.log(1 / (4 * np.pi))]
                                )

                        if atmospheric_comp:
                            # Atmospheric component
                            with ElseIfBlockContext(
                                [StringExpression([k, " == ", Ns + 2])]
                            ):
                                StringExpression(
                                    [
                                        lp[i][k],
                                        " += negative_infinity()",
                                    ]
                                )
                                E[i] << Esrc[i]

                        # Detection effects
                        StringExpression(
                            [
                                lp[i][k],
                                " += ",
                                dm["cascades"].energy_resolution(E[i], Edet[i]),
                            ]
                        )
                        StringExpression(
                            [
                                lp[i][k],
                                " += log(interpolate(",
                                Eg,
                                ", ",
                                Pg_c[k],
                                ", ",
                                E[i],
                                "))",
                            ]
                        )

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
            Nex_c << FunctionCall([F, eps_c], "get_Nex")

            Nex << Nex_c + Nex_t

        with ModelContext():

            with ForLoopContext(1, N, "i") as i:
                StringExpression(["target += log_sum_exp(", lp[i], ")"])
            StringExpression(["target += -", Nex])

            # StringExpression([L, " ~ normal(", L_scale, "), 5)"])
            # StringExpression([F_diff, " ~ lognormal(log(", F_diff_scale, "), 5)"])

            StringExpression(
                [
                    L,
                    " ~ ",
                    FunctionCall([L_scale, 2 * L_scale], "normal"),
                ]
            )
            StringExpression(
                [
                    F_diff,
                    " ~ ",
                    FunctionCall([F_diff_scale, 2 * F_diff_scale], "normal"),
                ]
            )

            if atmospheric_comp:
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
            StringExpression([src_index, " ~ normal(2.0, 2.0)"])
            StringExpression([diff_index, " ~ normal(2.0, 2.0)"])

    fit_gen.generate_single_file()

    return fit_gen.filename


def generate_stan_fit_code_(
    filename,
    detector_model_type,
    ps_spec_shape,
    diff_spec_shape,
    atmo_flux_model=None,
    diffuse_bg_comp=True,
    atmospheric_comp=True,
    theta_points=30,
    lumi_par_range=(0, 1e60),
    src_index_par_range=(1.0, 4.0),
    diff_index_par_range=(1.0, 4.0),
):

    with StanFileGenerator(filename) as fit_gen:

        with FunctionsContext():
            _ = Include("interpolation.stan")
            _ = Include("utils.stan")
            _ = Include("vMF.stan")
            dm = detector_model_type()

            src_spectrum_lpdf = ps_spec_shape.make_stan_lpdf_func("src_spectrum_logpdf")
            diff_spectrum_lpdf = diff_spec_shape.make_stan_lpdf_func(
                "diff_spectrum_logpdf"
            )

            flux_fac = ps_spec_shape.make_stan_flux_conv_func("flux_conv")

            if atmospheric_comp:
                atmu_nu_flux = atmo_flux_model.make_stan_function(
                    theta_points=theta_points
                )
                atmo_flux_integral = atmo_flux_model.total_flux_int.value

        with DataContext():

            # Neutrinos
            N = ForwardVariableDef("N", "int")
            N_str = ["[", N, "]"]
            omega_det = ForwardArrayDef("omega_det", "unit_vector[3]", N_str)
            Edet = ForwardVariableDef("Edet", "vector[N]")
            kappa = ForwardVariableDef("kappa", "vector[N]")
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
            src_index_grid = ForwardVariableDef("src_index_grid", "vector[Ngrid]")
            diff_index_grid = ForwardVariableDef("diff_index_grid", "vector[Ngrid]")
            integral_grid = ForwardArrayDef("integral_grid", "vector[Ngrid]", Ns_1p_str)
            Eg = ForwardVariableDef("E_grid", "vector[Ngrid]")

            if atmospheric_comp:
                Pg = ForwardArrayDef("Pdet_grid", "vector[Ngrid]", Ns_2p_str)
            else:
                Pg = ForwardArrayDef("Pdet_grid", "vector[Ngrid]", Ns_1p_str)

            # Inputs
            T = ForwardVariableDef("T", "real")

            # Priors
            L_scale = ForwardVariableDef("L_scale", "real")
            if diffuse_bg_comp:
                F_diff_scale = ForwardVariableDef("F_diff_scale", "real")
            if atmospheric_comp:
                atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")
                F_atmo_scale = ForwardVariableDef("F_atmo_scale", "real")
            F_tot_scale = ForwardVariableDef("F_tot_scale", "real")

        with ParametersContext():

            Lmin, Lmax = lumi_par_range
            src_index_min, src_index_max = src_index_par_range
            diff_index_min, diff_index_max = diff_index_par_range

            L = ParameterDef("L", "real", Lmin, Lmax)
            F_diff = ParameterDef("F_diff", "real", 0.0, 1e-4)
            if atmospheric_comp:
                F_atmo = ParameterDef("F_atmo", "real", 0.0, 1e-4)

            src_index = ParameterDef("src_index", "real", src_index_min, src_index_max)

            diff_index = ParameterDef(
                "diff_index", "real", diff_index_min, diff_index_max
            )

            Esrc = ParameterVectorDef("Esrc", "vector", N_str, Esrc_min, Esrc_max)

        with TransformedParametersContext():

            Fsrc = ForwardVariableDef("Fsrc", "real")

            if diffuse_bg_comp and atmospheric_comp:
                F = ForwardVariableDef("F", "vector[Ns+2]")
                eps = ForwardVariableDef("eps", "vector[Ns+2]")
                lp = ForwardArrayDef("lp", "vector[Ns+2]", N_str)
                logF = ForwardVariableDef("logF", "vector[Ns+2]")
                n_comps = "Ns+2"
            elif diffuse_bg_comp or atmospheric_comp:
                F = ForwardVariableDef("F", "vector[Ns+1]")
                eps = ForwardVariableDef("eps", "vector[Ns+1]")
                lp = ForwardArrayDef("lp", "vector[Ns+1]", N_str)
                logF = ForwardVariableDef("logF", "vector[Ns+1]")
                n_comps = "Ns+1"
            else:
                F = ForwardVariableDef("F", "vector[Ns]")
                eps = ForwardVariableDef("eps", "vector[Ns]")
                lp = ForwardArrayDef("lp", "vector[Ns]", N_str)
                logF = ForwardVariableDef("logF", "vector[Ns]")
                n_comps = "Ns"

            f = ParameterDef("f", "real", 0, 1)
            Ftot = ParameterDef("Ftot", "real", 0)

            Nex = ForwardVariableDef("Nex", "real")
            E = ForwardVariableDef("E", "vector[N]")

            Fsrc << 0.0
            with ForLoopContext(1, Ns, "k") as k:
                F[k] << StringExpression(
                    [L, "/ (4 * pi() * pow(", D[k], " * ", 3.086e22, ", 2))"]
                )
                StringExpression([F[k], "*=", flux_fac(src_index, Esrc_min, Esrc_max)])
                StringExpression([Fsrc, "+=", F[k]])
            StringExpression("F[Ns+1]") << F_diff

            if atmospheric_comp:
                StringExpression("F[Ns+2]") << F_atmo

            if atmospheric_comp and diffuse_bg_comp:
                Ftot << F_diff + F_atmo + Fsrc
            if diffuse_bg_comp and not atmospheric_comp:
                Ftot << F_diff + Fsrc

            f << StringExpression([Fsrc, " / ", Ftot])
            logF << StringExpression(["log(", F, ")"])

            with ForLoopContext(1, N, "i") as i:
                lp[i] << logF

                with ForLoopContext(1, n_comps, "k") as k:

                    # Point source components
                    with IfBlockContext([StringExpression([k, " < ", Ns + 1])]):
                        StringExpression(
                            [
                                lp[i][k],
                                " += ",
                                src_spectrum_lpdf(
                                    Esrc[i], src_index, Esrc_min, Esrc_max
                                ),
                            ]
                        )
                        E[i] << StringExpression([Esrc[i], " / (", 1 + z[k], ")"])

                        StringExpression(
                            [
                                lp[i][k],
                                " += vMF_lpdf(",
                                omega_det[i],
                                " | ",
                                varpi[k],
                                ", ",
                                kappa[i],
                                ")",
                            ]
                        )

                    if diffuse_bg_comp:
                        # Diffuse component
                        with ElseIfBlockContext(
                            [StringExpression([k, " == ", Ns + 1])]
                        ):
                            StringExpression(
                                [
                                    lp[i][k],
                                    " += ",
                                    diff_spectrum_lpdf(
                                        Esrc[i], diff_index, Esrc_min, Esrc_max
                                    ),
                                ]
                            )
                            E[i] << StringExpression([Esrc[i], " / (", 1 + z[k], ")"])
                            StringExpression(
                                [lp[i][k], " += ", np.log(1 / (4 * np.pi))]
                            )

                    if atmospheric_comp:
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

            if atmospheric_comp:
                eps << FunctionCall(
                    [
                        src_index,
                        diff_index,
                        src_index_grid,
                        diff_index_grid,
                        integral_grid,
                        atmo_integ_val,
                        T,
                        Ns,
                    ],
                    "get_exposure_factor_atmo",
                )
            else:
                eps << FunctionCall(
                    [
                        src_index,
                        diff_index,
                        src_index_grid,
                        diff_index_grid,
                        integral_grid,
                        T,
                        Ns,
                    ],
                    "get_exposure_factor",
                )
            Nex << FunctionCall([F, eps], "get_Nex")

        with ModelContext():

            with ForLoopContext(1, N, "i") as i:
                StringExpression(["target += log_sum_exp(", lp[i], ")"])
            StringExpression(["target += -", Nex])

            StringExpression([L, " ~ normal(0, ", L_scale, ")"])
            StringExpression([F_diff, " ~ normal(0, ", F_diff_scale, ")"])

            if atmospheric_comp:
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
            StringExpression([src_index, " ~ normal(2.0, 2.0)"])
            StringExpression([diff_index, " ~ normal(2.0, 2.0)"])

    fit_gen.generate_single_file()

    return fit_gen.filename
