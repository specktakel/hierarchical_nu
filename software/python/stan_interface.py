import numpy as np
import collections

from .events import TRACKS, CASCADES

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

from .detector.northern_tracks import NorthernTracksDetectorModel
from .detector.cascades import CascadesDetectorModel


def generate_atmospheric_sim_code_(filename, atmo_flux_model, theta_points=50):

    with StanFileGenerator(filename) as atmo_gen:

        with FunctionsContext():
            _ = Include("utils.stan")
            _ = Include("interpolation.stan")

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


def generate_main_sim_code_(
    filename,
    ps_spec_shape,
    detector_model_type,
    diffuse_bg_comp=True,
    atmospheric_comp=True,
):

    with StanFileGenerator(filename) as sim_gen:

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
            if diffuse_bg_comp:
                z = ForwardVariableDef("z", "vector[Ns+1]")
            else:
                z = ForwardVariableDef("z", "vector[Ns]")

            # Energies
            alpha = ForwardVariableDef("alpha", "real")
            Emin_det = ForwardVariableDef("Emin_det", "real")
            Esrc_min = ForwardVariableDef("Esrc_min", "real")
            Esrc_max = ForwardVariableDef("Esrc_max", "real")

            # Luminosity/ diffuse flux
            L = ForwardVariableDef("L", "real")
            F_diff = ForwardVariableDef("F_diff", "real")

            # Precomputed quantities
            Ngrid = ForwardVariableDef("Ngrid", "int")
            alpha_grid = ForwardVariableDef("alpha_grid", "vector[Ngrid]")
            if diffuse_bg_comp:
                integral_grid = ForwardArrayDef(
                    "integral_grid", "vector[Ngrid]", Ns_1p_str
                )
            else:
                integral_grid = ForwardArrayDef(
                    "integral_grid", "vector[Ngrid]", Ns_str
                )

            aeff_max = ForwardVariableDef("aeff_max", "real")

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
                w_exposure = ForwardVariableDef("w_exposure", "simplex[Ns+2]")
                eps = ForwardVariableDef("eps", "vector[Ns+2]")
            elif diffuse_bg_comp or atmospheric_comp:
                F = ForwardVariableDef("F", "vector[Ns+1]")
                w_exposure = ForwardVariableDef("w_exposure", "simplex[Ns+1]")
                eps = ForwardVariableDef("eps", "vector[Ns+1]")
            else:
                F = ForwardVariableDef("F", "vector[Ns]")
                w_exposure = ForwardVariableDef("w_exposure", "simplex[Ns]")
                eps = ForwardVariableDef("eps", "vector[Ns]")

            track_type = ForwardVariableDef("track_type", "int")
            cascade_type = ForwardVariableDef("cascade_type", "int")

            track_type << TRACKS
            cascade_type << CASCADES

            Ftot = ForwardVariableDef("Ftot", "real")
            Fsrc = ForwardVariableDef("Fs", "real")
            f = ForwardVariableDef("f", "real")
            Nex = ForwardVariableDef("Nex", "real")
            N = ForwardVariableDef("N", "int")

            Fsrc << 0.0
            with ForLoopContext(1, Ns, "k") as k:
                F[k] << StringExpression(
                    [L, "/ (4 * pi() * pow(", D[k], " * ", 3.086e22, ", 2))"]
                )
                StringExpression([F[k], "*=", flux_fac(alpha, Esrc_min, Esrc_max)])
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
                eps << FunctionCall(
                    [alpha, alpha_grid, integral_grid, atmo_integ_val, T, Ns],
                    "get_exposure_factor_atmo",
                )
            else:
                eps << FunctionCall(
                    [alpha, alpha_grid, integral_grid, T, Ns], "get_exposure_factor"
                )

            Nex << StringExpression(["get_Nex(", F, ", ", eps, ")"])
            w_exposure << StringExpression(["get_exposure_weights(", F, ", ", eps, ")"])
            N << StringExpression(["poisson_rng(", Nex, ")"])
            StringExpression(["print(", w_exposure, ")"])
            StringExpression(["print(", Ngrid, ")"])
            StringExpression(["print(", Nex, ")"])
            StringExpression(["print(", N, ")"])

        with GeneratedQuantitiesContext():

            dm_rng = detector_model_type(mode=DistributionMode.RNG)
            dm_pdf = detector_model_type(mode=DistributionMode.PDF)

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
            Nex_sim = ForwardVariableDef("Nex_sim", "real")

            event_type = ForwardVariableDef("event_type", "vector[N]")
            kappa = ForwardVariableDef("kappa", "vector[N]")
            
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
                    with IfBlockContext([StringExpression([lam[i], " <= ", Ns + 1])]):
                        Esrc[i] << spectrum_rng(alpha, Esrc_min, Esrc_max)
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
                            Pdet[i] << dm_pdf.effective_area(E[i], omega) / aeff_max

                    if detector_model_type == CascadesDetectorModel:

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
                                        Emin_det,
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
                kappa[i] << dm_rng.angular_resolution.kappa()
              
                if detector_model_type == NorthernTracksDetectorModel:
                    event_type[i] << track_type
                if detector_model_type == CascadesDetectorModel:
                    event_type[i] << cascade_type

    sim_gen.generate_single_file()

    return sim_gen.filename


def generate_main_sim_code_hybrid_(
    filename,
    ps_spec_shape,
    detector_model_type,
    diffuse_bg_comp=True,
    atmospheric_comp=True,
):

    with StanFileGenerator(filename) as sim_gen:

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
            if diffuse_bg_comp:
                z = ForwardVariableDef("z", "vector[Ns+1]")
            else:
                z = ForwardVariableDef("z", "vector[Ns]")

            # Energies
            alpha = ForwardVariableDef("alpha", "real")
            Emin_det_tracks = ForwardVariableDef("Emin_det_tracks", "real")
            Emin_det_cascades = ForwardVariableDef("Emin_det_cascades", "real")
            Esrc_min = ForwardVariableDef("Esrc_min", "real")
            Esrc_max = ForwardVariableDef("Esrc_max", "real")

            # Luminosity/ diffuse flux
            L = ForwardVariableDef("L", "real")
            F_diff = ForwardVariableDef("F_diff", "real")

            # Precomputed quantities
            Ngrid = ForwardVariableDef("Ngrid", "int")
            alpha_grid = ForwardVariableDef("alpha_grid", "vector[Ngrid]")
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
                StringExpression([F[k], "*=", flux_fac(alpha, Esrc_min, Esrc_max)])
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
                    [alpha, alpha_grid, integral_grid_t, atmo_integ_val, T, Ns],
                    "get_exposure_factor_atmo",
                )
            else:
                eps_t << FunctionCall(
                    [alpha, alpha_grid, integral_grid_t, T, Ns],
                    "get_exposure_factor",
                )
            eps_c << FunctionCall(
                [alpha, alpha_grid, integral_grid_c, T, Ns],
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
                    with IfBlockContext([StringExpression([lam[i], " <= ", Ns + 1])]):
                        Esrc[i] << spectrum_rng(alpha, Esrc_min, Esrc_max)
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
                    with IfBlockContext([StringExpression([lam[i], " <= ", Ns + 1])]):
                        Esrc[i] << spectrum_rng(alpha, Esrc_min, Esrc_max)
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


def generate_stan_fit_code_hybrid_(
    filename,
    detector_model_type,
    ps_spec_shape,
    atmo_flux_model=None,
    diffuse_bg_comp=True,
    atmospheric_comp=True,
    theta_points=30,
    lumi_par_range=(0, 1e60),
    alpha_par_range=(1.0, 4.0),
):

    with StanFileGenerator(filename) as fit_gen:

        with FunctionsContext():
            _ = Include("utils.stan")
            _ = Include("vMF.stan")
            _ = Include("interpolation.stan")
            _ = Include("sim_functions.stan")

            dm = collections.OrderedDict()
            for event_type in detector_model_type.event_types:
                dm[event_type] = detector_model_type(event_type=event_type)

            spectrum_lpdf = ps_spec_shape.make_stan_lpdf_func("spectrum_logpdf")
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
            alpha_grid = ForwardVariableDef("alpha_grid", "vector[Ngrid]")

            integral_grid_t = ForwardArrayDef("integral_grid_t", "vector[Ngrid]", Ns_1p_str)
            integral_grid_c = ForwardArrayDef("integral_grid_c", "vector[Ngrid]", Ns_1p_str)

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
            alphamin, alphamax = alpha_par_range

            L = ParameterDef("L", "real", Lmin, Lmax)
            F_diff = ParameterDef("F_diff", "real", 0.0, 1e-6)

            if atmospheric_comp:
                F_atmo = ParameterDef("F_atmo", "real", 0.0, 1e-6)

            alpha = ParameterDef("alpha", "real", alphamin, alphamax)

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
                StringExpression([F[k], "*=", flux_fac(alpha, Esrc_min, Esrc_max)])
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

                with IfBlockContext([StringExpression([event_type[i], " == ", track_type])]):

                    with ForLoopContext(1, n_comps_max, "k") as k:

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
                            # StringExpression(
                            #     [
                            #         lp[i][k],
                            #         " += ",
                            #         dm["tracks"].angular_resolution(E[i], varpi[k], omega_det[i]),
                            #     ]
                            # )
                            StringExpression([lp[i][k], " += vMF_lpdf(", omega_det[i], " | ", varpi[k], ", ", kappa[i], ")"])
                            

                        if diffuse_bg_comp:
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
                            [lp[i][k], " += ", dm["tracks"].energy_resolution(E[i], Edet[i])]
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

                with ElseIfBlockContext([StringExpression([event_type[i], " == ", cascade_type])]):

                    with ForLoopContext(1, n_comps_max, "k") as k:

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
                            # StringExpression(
                            #    [
                            #        lp[i][k],
                            #        " += ",
                            #        dm["cascades"].angular_resolution(E[i], varpi[k], omega_det[i]),
                            #    ]
                            # )
                            StringExpression([lp[i][k], " += vMF_lpdf(", omega_det[i], " | ", varpi[k], ", ", kappa[i], ")"])

                        if diffuse_bg_comp:
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
                            [lp[i][k], " += ", dm["cascades"].energy_resolution(E[i], Edet[i])]
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
                    [alpha, alpha_grid, integral_grid_t, atmo_integ_val, T, Ns],
                    "get_exposure_factor_atmo",
                )
            else:
                eps_t << FunctionCall(
                    [alpha, alpha_grid, integral_grid_t, T, Ns],
                    "get_exposure_factor",
                )
            eps_c << FunctionCall(
                    [alpha, alpha_grid, integral_grid_c, T, Ns],
                    "get_exposure_factor",
                )

            Nex_t << FunctionCall([F, eps_t], "get_Nex")
            Nex_c << FunctionCall([F, eps_c], "get_Nex")

            Nex << Nex_c + Nex_t

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
            StringExpression([alpha, " ~ normal(2.0, 2.0)"])

    fit_gen.generate_single_file()

    return fit_gen.filename

def generate_stan_fit_code_(
    filename,
    detector_model_type,
    ps_spec_shape,
    atmo_flux_model=None,
    diffuse_bg_comp=True,
    atmospheric_comp=True,
    theta_points=30,
    lumi_par_range=(0, 1e60),
    alpha_par_range=(1.0, 4.0),
):

    with StanFileGenerator(filename) as fit_gen:

        with FunctionsContext():
            _ = Include("utils.stan")
            _ = Include("vMF.stan")
            _ = Include("interpolation.stan")
            _ = Include("sim_functions.stan")
            dm = detector_model_type()

            spectrum_lpdf = ps_spec_shape.make_stan_lpdf_func("spectrum_logpdf")
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
            alpha_grid = ForwardVariableDef("alpha_grid", "vector[Ngrid]")
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
            alphamin, alphamax = alpha_par_range

            L = ParameterDef("L", "real", Lmin, Lmax)
            F_diff = ParameterDef("F_diff", "real", 0.0, 1e-6)
            if atmospheric_comp:
                F_atmo = ParameterDef("F_atmo", "real", 0.0, 1e-6)

            alpha = ParameterDef("alpha", "real", alphamin, alphamax)

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
                StringExpression([F[k], "*=", flux_fac(alpha, Esrc_min, Esrc_max)])
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
                                spectrum_lpdf(Esrc[i], alpha, Esrc_min, Esrc_max),
                            ]
                        )
                        E[i] << StringExpression([Esrc[i], " / (", 1 + z[k], ")"])
                        # StringExpression(
                        #     [
                        #         lp[i][k],
                        #         " += ",
                        #         dm.angular_resolution(E[i], varpi[k], omega_det[i]),
                        #     ]
                        # )
                        StringExpression([lp[i][k], " += vMF_lpdf(", omega_det[i], " | ", varpi[k], ", ", kappa[i], ")"])

                    if diffuse_bg_comp:
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
                    [alpha, alpha_grid, integral_grid, atmo_integ_val, T, Ns],
                    "get_exposure_factor_atmo",
                )
            else:
                eps << FunctionCall(
                    [alpha, alpha_grid, integral_grid, T, Ns],
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
            StringExpression([alpha, " ~ normal(2.0, 2.0)"])

    fit_gen.generate_single_file()

    return fit_gen.filename
