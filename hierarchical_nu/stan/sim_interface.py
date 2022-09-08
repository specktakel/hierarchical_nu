import numpy as np
import os
from collections import OrderedDict

from hierarchical_nu.stan.interface import StanInterface

from hierarchical_nu.backend.stan_generator import (
    FunctionsContext,
    Include,
    DataContext,
    TransformedDataContext,
    GeneratedQuantitiesContext,
    ForLoopContext,
    IfBlockContext,
    ElseIfBlockContext,
    ElseBlockContext,
    WhileLoopContext,
    FunctionCall,
)

from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
)

from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.events import TRACKS, CASCADES
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel


class StanSimInterface(StanInterface):
    """
    For generating Stan sim code.
    """

    def __init__(
        self,
        output_file,
        sources,
        detector_model_type,
        atmo_flux_theta_points=30,
        includes=[
            "interpolation.stan",
            "utils.stan",
            "vMF.stan",
            "rejection_sampling.stan",
        ],
    ):

        super().__init__(
            output_file=output_file,
            sources=sources,
            detector_model_type=detector_model_type,
            includes=includes,
        )

        self._atmo_flux_theta_points = atmo_flux_theta_points

    def _functions(self):

        with FunctionsContext():

            for include_file in self._includes:
                _ = Include(include_file)

            if self.sources.point_source:

                self._src_spectrum_lpdf = self._ps_spectrum.make_stan_lpdf_func(
                    "src_spectrum_logpdf"
                )

                self._flux_conv = self._ps_spectrum.make_stan_flux_conv_func(
                    "flux_conv"
                )

            if self.sources.diffuse:

                self._diff_spectrum_lpdf = self._diff_spectrum.make_stan_lpdf_func(
                    "diff_spectrum_logpdf"
                )

            if self.sources.atmospheric:

                atmo_flux_model = self.sources.atmospheric.flux_model

                # Increasing theta points too much makes compilation very slow
                # Could switch to passing array as data if problematic
                self._atmo_flux = atmo_flux_model.make_stan_function(
                    theta_points=self._atmo_flux_theta_points
                )

    def _data(self):

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

            if self.sources.diffuse or self.sources.point_source:

                self._Ngrid = ForwardVariableDef("Ngrid", "int")

            if self.sources.diffuse:

                self._z = ForwardVariableDef("z", "vector[Ns+1]")
                self._diff_index = ForwardVariableDef("diff_index", "real")
                self._diff_index_grid = ForwardVariableDef(
                    "diff_index_grid", "vector[Ngrid]"
                )

            else:

                self._z = ForwardVariableDef("z", "vector[Ns]")

            if self.sources.point_source:

                if self._shared_src_index:
                    self._src_index = ForwardVariableDef("src_index", "real")
                else:
                    self._src_index = ForwardVariableDef("src_index", "vector[Ns]")

                self._src_index_grid = ForwardVariableDef(
                    "src_index_grid", "vector[Ngrid]"
                )

            self._Esrc_min = ForwardVariableDef("Esrc_min", "real")
            self._Esrc_max = ForwardVariableDef("Esrc_max", "real")

            if "tracks" in self._event_types:

                self._Emin_det_t = ForwardVariableDef("Emin_det_t", "real")
                self._rs_bbpl_Eth_t = ForwardVariableDef("rs_bpl_Eth_t", "real")
                self._rs_bbpl_gamma1_t = ForwardVariableDef("rs_bpl_gamma1_t", "real")
                self._rs_bbpl_gamma2_t = ForwardVariableDef("rs_bpl_gamma2_t", "real")
                self._rs_N_cosz_bins_t = ForwardVariableDef("rs_N_cosz_bins_t", "int")
                self._rs_cvals_t = ForwardArrayDef(
                    "rs_cvals_t", "vector[rs_N_cosz_bins_t]", [f"[{self.sources.N}]"]
                )
                self._rs_cosz_bin_edges_t = ForwardArrayDef(
                    "rs_cosz_bin_edges_t", "real", ["[rs_N_cosz_bins_t + 1]"]
                )

                if self.sources.diffuse or self.sources.point_source:
                    self._integral_grid_t = ForwardArrayDef(
                        "integral_grid_t", "vector[Ngrid]", N_int_str
                    )

            if "cascades" in self._event_types:

                self._Emin_det_c = ForwardVariableDef("Emin_det_c", "real")
                self._rs_bbpl_Eth_c = ForwardVariableDef("rs_bpl_Eth_c", "real")
                self._rs_bbpl_gamma1_c = ForwardVariableDef("rs_bpl_gamma1_c", "real")
                self._rs_bbpl_gamma2_c = ForwardVariableDef("rs_bpl_gamma2_c", "real")
                self._rs_N_cosz_bins_c = ForwardVariableDef("rs_N_cosz_bins_c", "int")
                self._rs_cvals_c = ForwardArrayDef(
                    "rs_cvals_c", "vector[rs_N_cosz_bins_c]", [f"[{self.sources.N}]"]
                )
                self._rs_cosz_bin_edges_c = ForwardArrayDef(
                    "rs_cosz_bin_edges_c", "real", ["[rs_N_cosz_bins_c + 1]"]
                )

                if self.sources.diffuse or self.sources.point_source:
                    self._integral_grid_c = ForwardArrayDef(
                        "integral_grid_c", "vector[Ngrid]", N_int_str
                    )

            if self.sources.point_source:

                if self._shared_luminosity:
                    self._L = ForwardVariableDef("L", "real")
                else:
                    self._L = ForwardVariableDef("L", "vector[Ns]")

            if self.sources.diffuse:

                self._F_diff = ForwardVariableDef("F_diff", "real")

            if self.sources.atmospheric:

                self._F_atmo = ForwardVariableDef("F_atmo", "real")

                self._atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")

            self._v_lim = ForwardVariableDef("v_lim", "real")
            self._T = ForwardVariableDef("T", "real")

    def _transformed_data(self):

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

                N_tot_t = N_tot_c = "[Ns]"

            if "tracks" in self._event_types:

                self._track_type = ForwardVariableDef("track_type", "int")
                self._track_type << TRACKS

                self._w_exposure_t = ForwardVariableDef(
                    "w_exposure_t", "simplex" + N_tot_t
                )

                self._eps_t = ForwardVariableDef("eps_t", "vector" + N_tot_t)
                self._Nex_t = ForwardVariableDef("Nex_t", "real")
                self._N_t = ForwardVariableDef("N_t", "int")

            if "cascades" in self._event_types:

                self._cascade_type = ForwardVariableDef("cascade_type", "int")
                self._cascade_type << CASCADES

                self._w_exposure_c = ForwardVariableDef(
                    "w_exposure_c", "simplex" + N_tot_c
                )

                self._eps_c = ForwardVariableDef("eps_c", "vector" + N_tot_c)
                self._Nex_c = ForwardVariableDef("Nex_c", "real")
                self._N_c = ForwardVariableDef("N_c", "int")

            self._Ftot = ForwardVariableDef("Ftot", "real")
            self._F_src = ForwardVariableDef("Fs", "real")

            self._f_arr_ = ForwardVariableDef("f_arr_", "real")
            self._f_arr_astro_ = ForwardVariableDef("f_arr_astro_", "real")
            self._f_det_ = ForwardVariableDef("f_det_", "real")
            self._f_det_astro_ = ForwardVariableDef("f_det_astro_", "real")

            self._Nex_src_t = ForwardVariableDef("Nex_src_t", "real")
            self._Nex_src_c = ForwardVariableDef("Nex_src_c", "real")
            self._Nex_src = ForwardVariableDef("Nex_src", "real")
            self._Nex_diff_t = ForwardVariableDef("Nex_diff_t", "real")
            self._Nex_diff_c = ForwardVariableDef("Nex_diff_c", "real")
            self._Nex_diff = ForwardVariableDef("Nex_diff", "real")
            self._Nex_atmo = ForwardVariableDef("Nex_atmo", "real")
            self._N = ForwardVariableDef("N", "int")

            self._F_src << 0.0
            self._Nex_src_t << 0.0
            self._Nex_src_c << 0.0
            self._Nex_src << 0.0

            if self.sources.point_source:

                with ForLoopContext(1, self._Ns, "k") as k:

                    if self._shared_luminosity:
                        L_ref = self._L
                    else:
                        L_ref = self._L[k]

                    if self._shared_src_index:
                        src_index_ref = self._src_index
                    else:
                        src_index_ref = self._src_index[k]

                    self._F[k] << StringExpression(
                        [
                            L_ref,
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
                                src_index_ref, self._Esrc_min, self._Esrc_max
                            ),
                        ]
                    )
                    StringExpression([self._F_src, " += ", self._F[k]])

            if self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_diff

            if self.sources.atmospheric and not self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_atmo

            if self.sources.atmospheric and self.sources.diffuse:
                StringExpression("F[Ns+2]") << self._F_atmo

            if self.sources.point_source:

                with ForLoopContext(1, self._Ns, "k") as k:

                    if self._shared_src_index:
                        src_index_ref = self._src_index
                    else:
                        src_index_ref = self._src_index[k]

                    if "tracks" in self._event_types:

                        (
                            self._eps_t[k]
                            << FunctionCall(
                                [
                                    self._src_index_grid,
                                    self._integral_grid_t[k],
                                    src_index_ref,
                                ],
                                "interpolate",
                            )
                            * self._T
                        )

                        StringExpression(
                            [self._Nex_src_t, "+=", self._F[k] * self._eps_t[k]]
                        )

                    if "cascades" in self._event_types:

                        (
                            self._eps_c[k]
                            << FunctionCall(
                                [
                                    self._src_index_grid,
                                    self._integral_grid_c[k],
                                    src_index_ref,
                                ],
                                "interpolate",
                            )
                            * self._T
                        )

                        StringExpression(
                            [self._Nex_src_c, "+=", self._F[k] * self._eps_c[k]]
                        )

            if self.sources.diffuse and self.sources.atmospheric:

                if "tracks" in self._event_types:

                    (
                        self._eps_t[self._Ns + 1]
                        << FunctionCall(
                            [
                                self._diff_index_grid,
                                self._integral_grid_t[self._Ns + 1],
                                self._diff_index,
                            ],
                            "interpolate",
                        )
                        * self._T
                    )

                    (
                        self._Nex_diff_t
                        << self._F[self._Ns + 1] * self._eps_t[self._Ns + 1]
                    )

                    self._eps_t[self._Ns + 2] << self._atmo_integ_val * self._T

                    self._Nex_atmo << self._F[self._Ns + 2] * self._eps_t[self._Ns + 2]

                if "cascades" in self._event_types:

                    (
                        self._eps_c[self._Ns + 1]
                        << FunctionCall(
                            [
                                self._diff_index_grid,
                                self._integral_grid_c[self._Ns + 1],
                                self._diff_index,
                            ],
                            "interpolate",
                        )
                        * self._T
                    )

                    (
                        self._Nex_diff_c
                        << self._F[self._Ns + 1] * self._eps_c[self._Ns + 1]
                    )

            elif self.sources.diffuse:

                if "tracks" in self._event_types:

                    (
                        self._eps_t[self._Ns + 1]
                        << FunctionCall(
                            [
                                self._diff_index_grid,
                                self._integral_grid_t[self._Ns + 1],
                                self._diff_index,
                            ],
                            "interpolate",
                        )
                        * self._T
                    )

                    (
                        self._Nex_diff_t
                        << self._F[self._Ns + 1] * self._eps_t[self._Ns + 1]
                    )

                if "cascades" in self._event_types:

                    (
                        self._eps_c[self._Ns + 1]
                        << FunctionCall(
                            [
                                self._diff_index_grid,
                                self._integral_grid_c[self._Ns + 1],
                                self._diff_index,
                            ],
                            "interpolate",
                        )
                        * self._T
                    )

                    (
                        self._Nex_diff_c
                        << self._F[self._Ns + 1] * self._eps_c[self._Ns + 1]
                    )

            elif self.sources.atmospheric and "tracks" in self._event_types:

                self._eps_t[self._Ns + 1] << self._atmo_integ_val * self._T

                self._Nex_atmo << self._F[self._Ns + 1] * self._eps_t[self._Ns + 1]

                if "cascades" in self._event_types:

                    self._eps_c[self._Ns + 1] << 0.0

            if "tracks" in self._event_types:

                self._Nex_t << FunctionCall([self._F, self._eps_t], "get_Nex")
                self._w_exposure_t << FunctionCall(
                    [self._F, self._eps_t], "get_exposure_weights"
                )
                self._N_t << StringExpression(["poisson_rng(", self._Nex_t, ")"])

            if "cascades" in self._event_types:

                self._Nex_c << FunctionCall([self._F, self._eps_c], "get_Nex")
                self._w_exposure_c << FunctionCall(
                    [self._F, self._eps_c], "get_exposure_weights"
                )
                self._N_c << StringExpression(["poisson_rng(", self._Nex_c, ")"])

            if "tracks" in self._event_types and "cascades" in self._event_types:

                self._Nex_src << self._Nex_src_t + self._Nex_src_c
                self._Nex_diff << self._Nex_diff_t + self._Nex_diff_c
                self._N << self._N_t + self._N_c

            elif "tracks" in self._event_types:

                self._Nex_src << self._Nex_src_t
                self._Nex_diff << self._Nex_diff_t
                self._N << self._N_t

            elif "cascades" in self._event_types:

                self._Nex_src << self._Nex_src_c
                self._Nex_diff << self._Nex_diff_c
                self._Nex_atmo << 0.0
                self._N << self._N_c

            if self.sources.diffuse and self.sources.atmospheric:
                self._Ftot << self._F_src + self._F_diff + self._F_atmo
                self._f_arr_astro_ << StringExpression(
                    [self._F_src, "/", self._F_src + self._F_diff]
                )
                self._f_det_ << self._Nex_src / (
                    self._Nex_src + self._Nex_diff + self._Nex_atmo
                )
                self._f_det_astro_ << self._Nex_src / (self._Nex_src + self._Nex_diff)

            elif self.sources.diffuse:
                self._Ftot << self._F_src + self._F_diff
                self._f_arr_astro_ << StringExpression(
                    [self._F_src, "/", self._F_src + self._F_diff]
                )
                self._f_det_ << self._Nex_src / (self._Nex_src + self._Nex_diff)
                self._f_det_astro_ << self._f_det_

            elif self.sources.atmospheric:
                self._Ftot << self._F_src + self._F_atmo
                self._f_arr_astro_ << 1.0
                self._f_det_ << self._Nex_src / (self._Nex_src + self._Nex_atmo)
                self._f_det_astro_ << 1.0

            else:
                self._Ftot << self._F_src
                self._f_arr_astro_ << 1.0
                self._f_det_ << 1.0
                self._f_det_astro_ << 1.0

            self._f_arr_ << StringExpression([self._F_src, "/", self._Ftot])

            # StringExpression(['print("f_arr: ", ', self._f_arr_, ")"])

    def _generated_quantities(self):

        with GeneratedQuantitiesContext():

            self._dm_rng = OrderedDict()
            self._dm_pdf = OrderedDict()

            for event_type in self._event_types:

                self._dm_rng[event_type] = self.detector_model_type(
                    mode=DistributionMode.RNG,
                    event_type=event_type,
                )
                self._dm_pdf[event_type] = self.detector_model_type(
                    mode=DistributionMode.PDF,
                    event_type=event_type,
                )

            self._f_arr = ForwardVariableDef("f_arr", "real")
            self._f_arr_astro = ForwardVariableDef("f_arr_astro", "real")
            self._f_det = ForwardVariableDef("f_det", "real")
            self._f_det_astro = ForwardVariableDef("f_det_astro", "real")
            self._f_arr << self._f_arr_
            self._f_arr_astro << self._f_arr_astro_
            self._f_det << self._f_det_
            self._f_det_astro << self._f_det_astro_

            self._N_str = ["[", self._N, "]"]
            self._lam = ForwardArrayDef("Lambda", "int", self._N_str)
            self._omega = ForwardVariableDef("omega", "unit_vector[3]")

            self._Esrc = ForwardVariableDef("Esrc", "vector[N]")
            self._E = ForwardVariableDef("E", "vector[N]")
            self._Edet = ForwardVariableDef("Edet", "vector[N]")

            self._cosz = ForwardArrayDef("cosz", "real", self._N_str)
            self._Pdet = ForwardArrayDef("Pdet", "real", self._N_str)
            self._accept = ForwardVariableDef("accept", "int")
            self._detected = ForwardVariableDef("detected", "int")
            self._ntrials = ForwardVariableDef("ntrials", "int")
            self._prob = ForwardVariableDef("prob", "simplex[2]")

            self._event = ForwardArrayDef("event", "unit_vector[3]", self._N_str)

            # For rejection sampling
            self._u_samp = ForwardVariableDef("u_samp", "real")
            self._aeff_factor = ForwardVariableDef("aeff_factor", "real")
            self._src_factor = ForwardVariableDef("src_factor", "real")
            self._f_value = ForwardVariableDef("f_value", "real")
            self._g_value = ForwardVariableDef("g_value", "real")
            self._c_value = ForwardVariableDef("c_value", "real")
            self._idx_cosz = ForwardVariableDef("idx_cosz", "int")

            if "tracks" in self._event_types:
                Nex_t_sim = ForwardVariableDef("Nex_t_sim", "real")
                Nex_t_sim << self._Nex_t

            if "cascades" in self._event_types:
                Nex_c_sim = ForwardVariableDef("Nex_c_sim", "real")
                Nex_c_sim << self._Nex_c

            self._event_type = ForwardVariableDef("event_type", "vector[N]")
            self._kappa = ForwardVariableDef("kappa", "vector[N]")

            if "tracks" in self._event_types:

                with ForLoopContext(1, self._N_t, "i") as i:

                    self._event_type[i] << self._track_type

                    self._lam[i] << FunctionCall(
                        [self._w_exposure_t], "categorical_rng"
                    )

                    self._accept << 0
                    self._detected << 0
                    self._ntrials << 0

                    with WhileLoopContext([StringExpression([self._accept != 1])]):

                        self._u_samp << FunctionCall([0.0, 1.0], "uniform_rng")
                        self._E[i] << FunctionCall(
                            [
                                self._Esrc_min / (1 + self._z[self._lam[i]]),
                                self._rs_bbpl_Eth_t,
                                self._Esrc_max / (1 + self._z[self._lam[i]]),
                                self._rs_bbpl_gamma1_t,
                                self._rs_bbpl_gamma2_t,
                            ],
                            "bbpl_rng",
                        )

                        with IfBlockContext(
                            [StringExpression([self._lam[i], " <= ", self._Ns])]
                        ):

                            self._omega << self._varpi[self._lam[i]]

                        if self.sources.atmospheric and not self.sources.diffuse:

                            with ElseIfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                self._omega << FunctionCall(
                                    [1, self._v_lim], "sphere_lim_rng"
                                )

                        elif self.sources.diffuse:

                            with ElseIfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                self._omega << FunctionCall(
                                    [1, self._v_lim], "sphere_lim_rng"
                                )

                        if self.sources.atmospheric and self._sources.diffuse:

                            with ElseIfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 2])]
                            ):

                                self._omega << FunctionCall(
                                    [1, self._v_lim], "sphere_lim_rng"
                                )

                        self._cosz[i] << FunctionCall(
                            [FunctionCall([self._omega], "omega_to_zenith")], "cos"
                        )

                        self._aeff_factor << self._dm_pdf["tracks"].effective_area(
                            self._E[i], self._omega
                        )

                        # Energy spectrum
                        if self.sources.point_source:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " <= ", self._Ns])]
                            ):

                                if self._shared_src_index:
                                    src_index_ref = self._src_index
                                else:
                                    src_index_ref = self._src_index[self._lam[i]]

                                self._src_factor << self._src_spectrum_lpdf(
                                    self._E[i],
                                    src_index_ref,
                                    self._Esrc_min / (1 + self._z[self._lam[i]]),
                                    self._Esrc_max / (1 + self._z[self._lam[i]]),
                                )
                                self._src_factor << FunctionCall(
                                    [self._src_factor], "exp"
                                )

                                self._Esrc[i] << self._E[i] * (
                                    1 + self._z[self._lam[i]]
                                )

                        if self.sources.atmospheric and not self.sources.diffuse:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                (
                                    self._src_factor
                                    << self._atmo_flux(self._E[i], self._omega) * 1e7
                                )  # Scale for reasonable c_values (see precomputation)
                                self._Esrc[i] << self._E[i]

                        elif self.sources.diffuse:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                self._src_factor << self._diff_spectrum_lpdf(
                                    self._E[i],
                                    self._diff_index,
                                    self._Esrc_min / (1 + self._z[self._lam[i]]),
                                    self._Esrc_max / (1 + self._z[self._lam[i]]),
                                )
                                self._src_factor << FunctionCall(
                                    [self._src_factor], "exp"
                                )

                                self._Esrc[i] << self._E[i] * (
                                    1 + self._z[self._lam[i]]
                                )

                        if self.sources.diffuse and self.sources.atmospheric:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 2])]
                            ):

                                (
                                    self._src_factor
                                    << self._atmo_flux(self._E[i], self._omega) * 1e7
                                )  # Scale for reasonable c_values (see precomputation)
                                self._Esrc[i] << self._E[i]

                        # Calculate quantities for rejection sampling
                        if (
                            self.detector_model_type == NorthernTracksDetectorModel
                            or self.detector_model_type == IceCubeDetectorModel
                        ):

                            with IfBlockContext(
                                [StringExpression([self._cosz[i], ">= 0.1"])]
                            ):

                                self._aeff_factor << 0

                        self._Edet[i] << 10 ** self._dm_rng["tracks"].energy_resolution(
                            self._E[i]
                        )

                        self._f_value << self._src_factor * self._aeff_factor
                        self._g_value << FunctionCall(
                            [
                                self._E[i],
                                self._Esrc_min / (1 + self._z[self._lam[i]]),
                                self._rs_bbpl_Eth_t,
                                self._Esrc_max / (1 + self._z[self._lam[i]]),
                                self._rs_bbpl_gamma1_t,
                                self._rs_bbpl_gamma2_t,
                            ],
                            "bbpl_pdf",
                        )
                        self._idx_cosz << FunctionCall(
                            [self._cosz[i], self._rs_cosz_bin_edges_t], "binary_search"
                        )
                        self._c_value << self._rs_cvals_t[self._lam[i]][self._idx_cosz]

                        # Debugging when sampling gets stuck
                        StringExpression([self._ntrials, " += ", 1])

                        with IfBlockContext(
                            [StringExpression([self._ntrials, "< 1000000"])]
                        ):

                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            self._u_samp,
                                            " < ",
                                            self._f_value
                                            / (self._c_value * self._g_value),
                                        ]
                                    )
                                ]
                            ):

                                self._detected << 1

                            with ElseBlockContext():

                                self._detected << 0

                            # Energy threshold
                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            "(",
                                            self._Edet[i],
                                            " >= ",
                                            self._Emin_det_t,
                                            ") && (",
                                            self._detected == 1,
                                            ")",
                                        ]
                                    )
                                ]
                            ):
                                self._accept << 1

                        # Debugging
                        with ElseBlockContext():

                            self._accept << 1

                            StringExpression(
                                ['print("problem component: ", ', self._lam[i], ");\n"]
                            )

                    # Detection effects
                    self._event[i] << self._dm_rng["tracks"].angular_resolution(
                        self._E[i], self._omega
                    )
                    self._kappa[i] << self._dm_rng["tracks"].angular_resolution.kappa()

            if "cascades" in self._event_types:

                if "tracks" in self._event_types:

                    N_start = "N_t+1"

                else:

                    N_start = 1

                with ForLoopContext(N_start, self._N, "i") as i:

                    self._event_type[i] << self._cascade_type

                    self._lam[i] << FunctionCall(
                        [self._w_exposure_c], "categorical_rng"
                    )

                    self._accept << 0
                    self._detected << 0
                    self._ntrials << 0

                    with WhileLoopContext([StringExpression([self._accept != 1])]):

                        self._u_samp << FunctionCall([0.0, 1.0], "uniform_rng")
                        self._E[i] << FunctionCall(
                            [
                                self._Esrc_min / (1 + self._z[self._lam[i]]),
                                self._rs_bbpl_Eth_c,
                                self._Esrc_max / (1 + self._z[self._lam[i]]),
                                self._rs_bbpl_gamma1_c,
                                self._rs_bbpl_gamma2_c,
                            ],
                            "bbpl_rng",
                        )

                        # Sample position
                        with IfBlockContext(
                            [StringExpression([self._lam[i], " <= ", self._Ns])]
                        ):

                            self._omega << self._varpi[self._lam[i]]

                        with ElseIfBlockContext(
                            [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                        ):
                            self._omega << FunctionCall([1, 0], "sphere_lim_rng")

                        self._cosz[i] << FunctionCall(
                            [FunctionCall([self._omega], "omega_to_zenith")], "cos"
                        )

                        self._aeff_factor << self._dm_pdf["cascades"].effective_area(
                            self._E[i], self._omega
                        )

                        # Energy spectrum
                        if self.sources.point_source:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " <= ", self._Ns])]
                            ):

                                if self._shared_src_index:
                                    src_index_ref = self._src_index
                                else:
                                    src_index_ref = self._src_index[self._lam[i]]

                                self._src_factor << self._src_spectrum_lpdf(
                                    self._E[i],
                                    src_index_ref,
                                    self._Esrc_min / (1 + self._z[self._lam[i]]),
                                    self._Esrc_max / (1 + self._z[self._lam[i]]),
                                )
                                self._src_factor << FunctionCall(
                                    [self._src_factor], "exp"
                                )

                                self._Esrc[i] << self._E[i] * (
                                    1 + self._z[self._lam[i]]
                                )

                        if self.sources.diffuse:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                self._src_factor << self._diff_spectrum_lpdf(
                                    self._E[i],
                                    self._diff_index,
                                    self._Esrc_min / (1 + self._z[self._lam[i]]),
                                    self._Esrc_max / (1 + self._z[self._lam[i]]),
                                )
                                self._src_factor << FunctionCall(
                                    [self._src_factor], "exp"
                                )

                                self._Esrc[i] << self._E[i] * (
                                    1 + self._z[self._lam[i]]
                                )

                        # Calculate quantities for rejection sampling
                        self._Edet[i] << 10 ** self._dm_rng[
                            "cascades"
                        ].energy_resolution(self._E[i])

                        self._f_value = self._src_factor * self._aeff_factor
                        self._g_value << FunctionCall(
                            [
                                self._E[i],
                                self._Esrc_min / (1 + self._z[self._lam[i]]),
                                self._rs_bbpl_Eth_c,
                                self._Esrc_max / (1 + self._z[self._lam[i]]),
                                self._rs_bbpl_gamma1_c,
                                self._rs_bbpl_gamma2_c,
                            ],
                            "bbpl_pdf",
                        )
                        self._idx_cosz << FunctionCall(
                            [self._cosz[i], self._rs_cosz_bin_edges_c], "binary_search"
                        )
                        self._c_value << self._rs_cvals_c[self._lam[i]][self._idx_cosz]

                        # Debugging when sampling gets stuck
                        StringExpression([self._ntrials, " += ", 1])

                        with IfBlockContext(
                            [StringExpression([self._ntrials, "< 1000000"])]
                        ):

                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            self._u_samp,
                                            " < ",
                                            self._f_value
                                            / (self._c_value * self._g_value),
                                        ]
                                    )
                                ]
                            ):

                                self._detected << 1

                            with ElseBlockContext():

                                self._detected << 0

                            # Energy threshold
                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            "(",
                                            self._Edet[i],
                                            " >= ",
                                            self._Emin_det_c,
                                            ") && (",
                                            self._detected == 1,
                                            ")",
                                        ]
                                    )
                                ]
                            ):

                                self._accept << 1

                        # Debugging
                        with ElseBlockContext():

                            self._accept << 1

                            StringExpression(
                                ['print("problem component: ", ', self._lam[i], ");\n"]
                            )

                    # Detection effects
                    self._event[i] << self._dm_rng["cascades"].angular_resolution(
                        self._E[i], self._omega
                    )
                    (
                        self._kappa[i]
                        << self._dm_rng["cascades"].angular_resolution.kappa()
                    )
