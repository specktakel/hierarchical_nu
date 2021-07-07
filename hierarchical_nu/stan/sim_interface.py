import numpy as np
import os
from collections import OrderedDict

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
)

from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.events import TRACKS, CASCADES
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel

from hierarchical_nu.stan.interface import STAN_GEN_PATH


class StanSimInterface(StanInterface):
    """
    For generating Stan sim code.
    """

    def __init__(
        self,
        output_file,
        sources,
        detector_model_type,
        includes=["interpolation.stan", "utils.stan", "vMF.stan"],
    ):

        super().__init__(
            output_file=output_file,
            sources=sources,
            detector_model_type=detector_model_type,
            includes=includes,
        )

    def generate_atmo(self):

        atmo_flux_model = self.sources.atmospheric.flux_model

        filename = os.path.join(STAN_GEN_PATH, "atmo_gen")

        return generate_atmospheric_sim_code_(
            filename, atmo_flux_model, theta_points=30
        )

    def _functions(self):

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
                self._aeff_t_max = ForwardVariableDef("aeff_t_max", "real")

                if self.sources.diffuse or self.sources.point_source:
                    self._integral_grid_t = ForwardArrayDef(
                        "integral_grid_t", "vector[Ngrid]", N_int_str
                    )

            if "cascades" in self._event_types:

                self._Emin_det_c = ForwardVariableDef("Emin_det_c", "real")
                self._aeff_c_max = ForwardVariableDef("aeff_c_max", "real")

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
            self._Fsrc = ForwardVariableDef("Fs", "real")

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

            self._Fsrc << 0.0
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
                    StringExpression([self._Fsrc, " += ", self._F[k]])

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

                        self._eps_t[k] << FunctionCall(
                            [
                                self._src_index_grid,
                                self._integral_grid_t[k],
                                src_index_ref,
                            ],
                            "interpolate",
                        ) * self._T

                        StringExpression(
                            [self._Nex_src_t, "+=", self._F[k] * self._eps_t[k]]
                        )

                    if "cascades" in self._event_types:

                        self._eps_c[k] << FunctionCall(
                            [
                                self._src_index_grid,
                                self._integral_grid_c[k],
                                src_index_ref,
                            ],
                            "interpolate",
                        ) * self._T

                        StringExpression(
                            [self._Nex_src_c, "+=", self._F[k] * self._eps_c[k]]
                        )

            if self.sources.diffuse and self.sources.atmospheric:

                if "tracks" in self._event_types:

                    self._eps_t[self._Ns + 1] << FunctionCall(
                        [
                            self._diff_index_grid,
                            self._integral_grid_t[self._Ns + 1],
                            self._diff_index,
                        ],
                        "interpolate",
                    ) * self._T

                    self._Nex_diff_t << self._F[self._Ns + 1] * self._eps_t[
                        self._Ns + 1
                    ]

                    self._eps_t[self._Ns + 2] << self._atmo_integ_val * self._T

                    self._Nex_atmo << self._F[self._Ns + 2] * self._eps_t[self._Ns + 2]

                if "cascades" in self._event_types:

                    self._eps_c[self._Ns + 1] << FunctionCall(
                        [
                            self._diff_index_grid,
                            self._integral_grid_c[self._Ns + 1],
                            self._diff_index,
                        ],
                        "interpolate",
                    ) * self._T

                    self._Nex_diff_c << self._F[self._Ns + 1] * self._eps_c[
                        self._Ns + 1
                    ]

            elif self.sources.diffuse:

                if "tracks" in self._event_types:

                    self._eps_t[self._Ns + 1] << FunctionCall(
                        [
                            self._diff_index_grid,
                            self._integral_grid_t[self._Ns + 1],
                            self._diff_index,
                        ],
                        "interpolate",
                    ) * self._T

                    self._Nex_diff_t << self._F[self._Ns + 1] * self._eps_t[
                        self._Ns + 1
                    ]

                if "cascades" in self._event_types:

                    self._eps_c[self._Ns + 1] << FunctionCall(
                        [
                            self._diff_index_grid,
                            self._integral_grid_c[self._Ns + 1],
                            self._diff_index,
                        ],
                        "interpolate",
                    ) * self._T

                    self._Nex_diff_c << self._F[self._Ns + 1] * self._eps_c[
                        self._Ns + 1
                    ]

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
                self._Ftot << self._Fsrc + self._F_diff + self._F_atmo
                self._f_arr_astro_ << StringExpression(
                    [self._Fsrc, "/", self._Fsrc + self._F_diff]
                )
                self._f_det_ << self._Nex_src / (
                    self._Nex_src + self._Nex_diff + self._Nex_atmo
                )
                self._f_det_astro_ << self._Nex_src / (self._Nex_src + self._Nex_diff)

            elif self.sources.diffuse:
                self._Ftot << self._Fsrc + self._F_diff
                self._f_arr_astro_ << StringExpression(
                    [self._Fsrc, "/", self._Fsrc + self._F_diff]
                )
                self._f_det_ << self._Nex_src / (self._Nex_src + self._Nex_diff)
                self._f_det_astro_ << self._f_det_

            elif self.sources.atmospheric:
                self._Ftot << self._Fsrc + self._F_atmo
                self._f_arr_astro_ << 1.0
                self._f_det_ << self._Nex_src / (self._Nex_src + self._Nex_atmo)
                self._f_det_astro_ << 1.0

            else:
                self._Ftot << self._Fsrc
                self._f_arr_astro_ << 1.0
                self._f_det_ << 1.0
                self._f_det_astro << 1.0

            self._f_arr_ << StringExpression([self._Fsrc, "/", self._Ftot])

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

            if self.sources.atmospheric:

                self._atmo_index = ForwardVariableDef("atmo_index", "int")

            self._cosz = ForwardArrayDef("cosz", "real", self._N_str)
            self._Pdet = ForwardArrayDef("Pdet", "real", self._N_str)
            self._accept = ForwardVariableDef("accept", "int")
            self._detected = ForwardVariableDef("detected", "int")
            self._ntrials = ForwardVariableDef("ntrials", "int")
            self._prob = ForwardVariableDef("prob", "simplex[2]")

            self._event = ForwardArrayDef("event", "unit_vector[3]", self._N_str)

            if "tracks" in self._event_types:
                Nex_t_sim = ForwardVariableDef("Nex_t_sim", "real")
                Nex_t_sim << self._Nex_t

            if "cascades" in self._event_types:
                Nex_c_sim = ForwardVariableDef("Nex_c_sim", "real")
                Nex_c_sim << self._Nex_c

            self._event_type = ForwardVariableDef("event_type", "vector[N]")
            self._kappa = ForwardVariableDef("kappa", "vector[N]")

            if self.sources.atmospheric:
                self._atmo_weights_p = ForwardVariableDef(
                    "atmo_weights_p", "simplex[N_atmo]"
                )
                self._atmo_weights_v = ForwardVariableDef(
                    "atmo_weights_v", "vector[N_atmo]"
                )

                self._atmo_weights_p << self._atmo_weights
                self._atmo_weights_v << self._atmo_weights

            if "tracks" in self._event_types:

                with ForLoopContext(1, self._N_t, "i") as i:

                    self._event_type[i] << self._track_type

                    self._lam[i] << FunctionCall(
                        [self._w_exposure_t], "categorical_rng"
                    )

                    self._accept << 0
                    self._detected << 0
                    self._ntrials << 0

                    if self._sources.atmospheric:
                        self._atmo_index << 1

                    with WhileLoopContext([StringExpression([self._accept != 1])]):

                        with IfBlockContext(
                            [StringExpression([self._lam[i], " <= ", self._Ns])]
                        ):

                            self._omega << self._varpi[self._lam[i]]

                        if self.sources.atmospheric and not self.sources.diffuse:

                            with ElseIfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                self._atmo_index << FunctionCall(
                                    [self._atmo_weights_p], "categorical_rng"
                                )
                                self._omega << self._atmo_directions[self._atmo_index]

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
                                self._atmo_index << FunctionCall(
                                    [self._atmo_weights_p], "categorical_rng"
                                )
                                self._omega << self._atmo_directions[self._atmo_index]

                        self._cosz[i] << FunctionCall(
                            [FunctionCall([self._omega], "omega_to_zenith")], "cos"
                        )

                        # Energy
                        if self.sources.point_source:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " <= ", self._Ns])]
                            ):

                                if self._shared_src_index:
                                    src_index_ref = self._src_index
                                else:
                                    src_index_ref = self._src_index[self._lam[i]]

                                self._Esrc[i] << self._src_spectrum_rng(
                                    src_index_ref,
                                    self._Esrc_min,
                                    self._Esrc_max,
                                )
                                self._E[i] << self._Esrc[i] / (
                                    1 + self._z[self._lam[i]]
                                )

                        if self.sources.atmospheric and not self.sources.diffuse:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                self._Esrc[i] << self._atmo_energies[self._atmo_index]
                                self._E[i] << self._Esrc[i]

                        elif self.sources.diffuse:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                self._Esrc[i] << self._diff_spectrum_rng(
                                    self._diff_index, self._Esrc_min, self._Esrc_max
                                )
                                self._E[i] << self._Esrc[i] / (
                                    1 + self._z[self._lam[i]]
                                )

                        if self.sources.diffuse and self.sources.atmospheric:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 2])]
                            ):

                                self._Esrc[i] << self._atmo_energies[self._atmo_index]
                                self._E[i] << self._Esrc[i]

                        # Test against Aeff
                        if (
                            self.detector_model_type == NorthernTracksDetectorModel
                            or self.detector_model_type == IceCubeDetectorModel
                        ):

                            with IfBlockContext(
                                [StringExpression([self._cosz[i], ">= 0.1"])]
                            ):

                                self._Pdet[i] << 0

                            with ElseBlockContext():

                                self._Pdet[i] << self._dm_pdf["tracks"].effective_area(
                                    self._E[i], self._omega
                                ) / self._aeff_t_max

                        else:

                            self._Pdet[i] << self._dm_pdf["tracks"].effective_area(
                                self._E[i], self._omega
                            ) / self._aeff_t_max

                        self._Edet[i] << 10 ** self._dm_rng["tracks"].energy_resolution(
                            self._E[i]
                        )

                        self._prob[1] << self._Pdet[i]
                        self._prob[2] << 1 - self._Pdet[i]
                        StringExpression([self._ntrials, " += ", 1])

                        with IfBlockContext(
                            [StringExpression([self._ntrials, "< 1000000"])]
                        ):

                            self._detected << FunctionCall(
                                [self._prob], "categorical_rng"
                            )

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

                                # Stop same atmo events being sampled
                                # multiple times
                                if self.sources.atmospheric:

                                    self._atmo_weights_v << self._atmo_weights_p
                                    self._atmo_weights_v[self._atmo_index] << 0.0
                                    self._atmo_weights_v << self._atmo_weights_v / FunctionCall(
                                        [self._atmo_weights_v], "sum"
                                    )
                                    self._atmo_weights_p << self._atmo_weights_v

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

                        # Sample energy
                        if self.sources.point_source:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " <= ", self._Ns])]
                            ):

                                if self._shared_src_index:
                                    src_index_ref = self._src_index
                                else:
                                    src_index_ref = self._src_index[self._lam[i]]

                                self._Esrc[i] << self._src_spectrum_rng(
                                    src_index_ref,
                                    self._Esrc_min,
                                    self._Esrc_max,
                                )
                                self._E[i] << self._Esrc[i] / (
                                    1 + self._z[self._lam[i]]
                                )

                        if self.sources.diffuse:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):
                                self._Esrc[i] << self._diff_spectrum_rng(
                                    self._diff_index, self._Esrc_min, self._Esrc_max
                                )
                                self._E[i] << self._Esrc[i] / (
                                    1 + self._z[self._lam[i]]
                                )

                        # Test against Aeff
                        self._Pdet[i] << self._dm_pdf["cascades"].effective_area(
                            self._E[i], self._omega
                        ) / self._aeff_c_max

                        self._Edet[i] << 10 ** self._dm_rng[
                            "cascades"
                        ].energy_resolution(self._E[i])

                        self._prob[1] << self._Pdet[i]
                        self._prob[2] << 1 - self._Pdet[i]
                        StringExpression([self._ntrials, " += ", 1])

                        with IfBlockContext(
                            [StringExpression([self._ntrials, "< 1000000"])]
                        ):

                            self._detected << FunctionCall(
                                [self._prob], "categorical_rng"
                            )

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

                        with ElseBlockContext():

                            self._accept << 1

                            StringExpression(
                                ['print("problem component: ", ', self._lam[i], ");\n"]
                            )

                    # Detection effects
                    self._event[i] << self._dm_rng["cascades"].angular_resolution(
                        self._E[i], self._omega
                    )
                    self._kappa[i] << self._dm_rng[
                        "cascades"
                    ].angular_resolution.kappa()


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
