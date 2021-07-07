import numpy as np
from astropy import units as u
from collections import OrderedDict

from hierarchical_nu.priors import Priors
from hierarchical_nu.stan.interface import StanInterface

from hierarchical_nu.backend.stan_generator import (
    FunctionsContext,
    Include,
    DataContext,
    TransformedDataContext,
    ParametersContext,
    TransformedParametersContext,
    ForLoopContext,
    IfBlockContext,
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
from hierarchical_nu.source.parameter import Parameter


class StanFitInterface(StanInterface):
    """
    For generating Stan fit code.
    """

    def __init__(
        self,
        output_file,
        sources,
        detector_model_type,
        priors=Priors(),
        includes=["interpolation.stan", "utils.stan", "vMF.stan"],
        theta_points=30,
    ):

        super().__init__(
            output_file=output_file,
            sources=sources,
            detector_model_type=detector_model_type,
            includes=includes,
        )

        self._priors = priors

        self._theta_points = theta_points

        self._get_par_ranges()

    def _get_par_ranges(self):

        if self.sources.point_source:

            if self._shared_luminosity:
                key = "luminosity"
            else:
                key = "ps_0_luminosity"

            self._lumi_par_range = Parameter.get_parameter(key).par_range
            self._lumi_par_range = self._lumi_par_range.to(u.GeV / u.s).value

            if self._shared_src_index:
                key = "src_index"
            else:
                key = "ps_0_src_index"

            self._src_index_par_range = Parameter.get_parameter(key).par_range

        if self.sources.diffuse:

            self._diff_index_par_range = Parameter.get_parameter("diff_index").par_range

    def _functions(self):

        with FunctionsContext():

            for include_file in self._includes:
                _ = Include(include_file)

            self._dm = OrderedDict()

            for event_type in self._event_types:

                self._dm[event_type] = self._detector_model_type(
                    event_type=event_type,
                    mode=DistributionMode.PDF,
                )

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

                self._atmo_flux_func = self._atmo_flux.make_stan_function(
                    theta_points=self._theta_points
                )

                self._atmo_flux_integral = self._atmo_flux.total_flux_int.to(
                    1 / (u.m ** 2 * u.s)
                ).value

    def _data(self):

        with DataContext():

            self._N = ForwardVariableDef("N", "int")
            self._N_str = ["[", self._N, "]"]
            self._omega_det = ForwardArrayDef(
                "omega_det", "unit_vector[3]", self._N_str
            )
            self._Edet = ForwardVariableDef("Edet", "vector[N]")
            self._event_type = ForwardVariableDef("event_type", "vector[N]")
            self._kappa = ForwardVariableDef("kappa", "vector[N]")
            self._Esrc_min = ForwardVariableDef("Esrc_min", "real")
            self._Esrc_max = ForwardVariableDef("Esrc_max", "real")

            self._Ns = ForwardVariableDef("Ns", "int")
            self._Ns_str = ["[", self._Ns, "]"]
            self._Ns_1p_str = ["[", self._Ns, "+1]"]
            self._Ns_2p_str = ["[", self._Ns, "+2]"]

            self._varpi = ForwardArrayDef("varpi", "unit_vector[3]", self._Ns_str)
            self._D = ForwardVariableDef("D", "vector[Ns]")

            self._Ngrid = ForwardVariableDef("Ngrid", "int")
            self._Eg = ForwardVariableDef("E_grid", "vector[Ngrid]")

            self._T = ForwardVariableDef("T", "real")

            if self.sources.diffuse:

                N_int_str = self._Ns_1p_str
                self._z = ForwardVariableDef("z", "vector[Ns+1]")

            else:

                N_int_str = self._Ns_str
                self._z = ForwardVariableDef("z", "vector[Ns]")

            if self.sources.point_source or self.sources.diffuse:

                if self.sources.point_source:

                    self._src_index_grid = ForwardVariableDef(
                        "src_index_grid", "vector[Ngrid]"
                    )

                if self.sources.diffuse:

                    self._diff_index_grid = ForwardVariableDef(
                        "diff_index_grid", "vector[Ngrid]"
                    )

                if "tracks" in self._event_types:

                    self._integral_grid_t = ForwardArrayDef(
                        "integral_grid_t", "vector[Ngrid]", N_int_str
                    )

                if "cascades" in self._event_types:

                    self._integral_grid_c = ForwardArrayDef(
                        "integral_grid_c", "vector[Ngrid]", N_int_str
                    )

            if self.sources.diffuse and self.sources.atmospheric:

                N_pdet_str = self._Ns_2p_str

            elif self.sources.diffuse or self.sources.atmospheric:

                N_pdet_str = self._Ns_1p_str

            else:

                N_pdet_str = self._Ns_str

            if "tracks" in self._event_types:

                self._Pg_t = ForwardArrayDef("Pdet_grid_t", "vector[Ngrid]", N_pdet_str)

            if "cascades" in self._event_types:

                self._Pg_c = ForwardArrayDef("Pdet_grid_c", "vector[Ngrid]", N_pdet_str)

            if self.sources.atmospheric:

                self._atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")

    def _transformed_data(self):

        with TransformedDataContext():

            if "tracks" in self._event_types:

                self._track_type = ForwardVariableDef("track_type", "int")
                self._track_type << TRACKS

            if "cascades" in self._event_types:

                self._cascade_type = ForwardVariableDef("cascade_type", "int")
                self._cascade_type << CASCADES

    def _parameters(self):

        with ParametersContext():

            if self.sources.point_source:

                Lmin, Lmax = self._lumi_par_range
                src_index_min, src_index_max = self._src_index_par_range

                if self._shared_luminosity:

                    self._L = ParameterDef("L", "real", Lmin, Lmax)

                else:

                    self._L = ParameterVectorDef(
                        "L",
                        "vector",
                        self._Ns_str,
                        Lmin,
                        Lmax,
                    )

                if self._shared_src_index:

                    self._src_index = ParameterDef(
                        "src_index",
                        "real",
                        src_index_min,
                        src_index_max,
                    )

                else:

                    self._src_index = ParameterVectorDef(
                        "src_index",
                        "vector",
                        self._Ns_str,
                        src_index_min,
                        src_index_max,
                    )

            if self.sources.diffuse:

                diff_index_min, diff_index_max = self._diff_index_par_range

                self._F_diff = ParameterDef("F_diff", "real", 0, 1e-6)
                self._diff_index = ParameterDef(
                    "diff_index", "real", diff_index_min, diff_index_max
                )

            if self.sources.atmospheric:

                self._F_atmo = ParameterDef("F_atmo", "real", 0.0, 1e-6)

            self._Esrc = ParameterVectorDef(
                "Esrc", "vector", self._N_str, self._Esrc_min, self._Esrc_max
            )

    def _transformed_parameters(self):

        with TransformedParametersContext():

            self._Nex = ForwardVariableDef("Nex", "real")
            self._Nex_atmo = ForwardVariableDef("Nex_atmo", "real")
            self._Nex_src = ForwardVariableDef("Nex_src", "real")
            self._Nex_diff = ForwardVariableDef("Nex_diff", "real")
            self._Ftot = ForwardVariableDef("Ftot", "real")
            self._F_src = ForwardVariableDef("Fs", "real")
            self._f_arr = ForwardVariableDef("f_arr", "real")
            self._f_det = ForwardVariableDef("f_det", "real")
            self._f_arr_astro = ForwardVariableDef("f_arr_astro", "real")
            self._f_det_astro = ForwardVariableDef("f_det_astro", "real")
            self._E = ForwardVariableDef("E", "vector[N]")

            if self.sources.diffuse and self.sources.atmospheric:

                self._F = ForwardVariableDef("F", "vector[Ns+2]")
                self._logF = ForwardVariableDef("logF", "vector[Ns+2]")

                self._lp = ForwardArrayDef("lp", "vector[Ns+2]", self._N_str)

                n_comps_max = "Ns+2"
                N_tot_t = "[Ns+2]"
                N_tot_c = "[Ns+1]"

            elif self.sources.diffuse or self.sources.atmospheric:

                self._F = ForwardVariableDef("F", "vector[Ns+1]")
                self._logF = ForwardVariableDef("logF", "vector[Ns+1]")

                self._lp = ForwardArrayDef("lp", "vector[Ns+1]", self._N_str)

                n_comps_max = "Ns+1"
                N_tot_t = N_tot_c = "[Ns+1]"

            else:

                self._F = ForwardVariableDef("F", "vector[Ns]")
                self._logF = ForwardVariableDef("logF", "vector[Ns]")

                self._lp = ForwardArrayDef("lp", "vector[Ns]", self._N_str)

                n_comps_max = "Ns"
                N_tot_t = N_tot_c = "[Ns]"

            if "tracks" in self._event_types:

                self._eps_t = ForwardVariableDef("eps_t", "vector" + N_tot_t)
                self._Nex_src_t = ForwardVariableDef("Nex_src_t", "real")
                self._Nex_diff_t = ForwardVariableDef("Nex_diff_t", "real")
                self._Nex_t = ForwardVariableDef("Nex_t", "real")
                self._Nex_src_t << 0.0

            if "cascades" in self._event_types:

                self._eps_c = ForwardVariableDef("eps_c", "vector" + N_tot_c)
                self._Nex_src_c = ForwardVariableDef("Nex_src_c", "real")
                self._Nex_diff_c = ForwardVariableDef("Nex_diff_c", "real")
                self._Nex_c = ForwardVariableDef("Nex_c", "real")
                self._Nex_src_c << 0.0

            self._F_src << 0.0
            self._Nex_src << 0.0
            self._Nex_atmo << 0.0

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

            if "cascades" in self._event_types:

                self._Nex_c << FunctionCall([self._F, self._eps_c], "get_Nex")

            if "tracks" in self._event_types and "cascades" in self._event_types:

                self._Nex_src << self._Nex_src_t + self._Nex_src_c
                self._Nex_diff << self._Nex_diff_t + self._Nex_diff_c
                self._Nex << self._Nex_t + self._Nex_c

            elif "tracks" in self._event_types:

                self._Nex_src << self._Nex_src_t
                self._Nex_diff << self._Nex_diff_t
                self._Nex << self._Nex_t

            elif "cascades" in self._event_types:

                self._Nex_src << self._Nex_src_c
                self._Nex_diff << self._Nex_diff_c
                self._Nex << self._Nex_c

            if self.sources.diffuse and self.sources.atmospheric:
                self._Ftot << self._F_src + self._F_diff + self._F_atmo
                self._f_arr_astro << self._F_src / (self._F_src + self._F_diff)
                self._f_det << self._Nex_src / self._Nex
                self._f_det_astro << self._Nex_src / (self._Nex_src + self._Nex_diff)

            elif self.sources.diffuse:
                self._Ftot << self._F_src + self._F_diff
                self._f_arr_astro << self._F_src / (self._F_src + self._F_diff)
                self._f_det << self._Nex_src / self._Nex
                self._f_det_astro << self._f_det

            elif self.sources.atmospheric:
                self._Ftot << self._F_src + self._F_atmo
                self._f_arr_astro << 1.0
                self._f_det << self._Nex_src / (self._Nex_src + self._Nex_atmo)
                self._f_det_astro << 1.0

            else:
                self._Ftot << self._F_src
                self._f_arr_astro << 1.0
                self._f_det << 1.0
                self._f_det_astro << 1.0

            self._f_arr << StringExpression([self._F_src, "/", self._Ftot])

            if self.sources.diffuse and self.sources.atmospheric:

                k_diff = "Ns + 1"
                k_atmo = "Ns + 2"

            elif self.sources.diffuse:

                k_diff = "Ns + 1"

            elif self.sources.atmospheric:

                k_atmo = "Ns + 1"

            # Main model loop
            self._logF << StringExpression(["log(", self._F, ")"])

            with ForLoopContext(1, self._N, "i") as i:

                self._lp[i] << self._logF

                # Tracks
                if "tracks" in self._event_types:

                    with IfBlockContext(
                        [
                            StringExpression(
                                [self._event_type[i], " == ", self._track_type]
                            )
                        ]
                    ):

                        with ForLoopContext(1, n_comps_max, "k") as k:

                            # Point source components
                            if self.sources.point_source:

                                with IfBlockContext(
                                    [StringExpression([k, " < ", self._Ns + 1])]
                                ):

                                    if self._shared_src_index:
                                        src_index_ref = self._src_index
                                    else:
                                        src_index_ref = self._src_index[k]

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            self._src_spectrum_lpdf(
                                                self._Esrc[i],
                                                src_index_ref,
                                                self._Esrc_min,
                                                self._Esrc_max,
                                            ),
                                        ]
                                    )
                                    self._E[i] << StringExpression(
                                        [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                    )

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += vMF_lpdf(",
                                            self._omega_det[i],
                                            " | ",
                                            self._varpi[k],
                                            ", ",
                                            self._kappa[i],
                                            ")",
                                        ]
                                    )

                            # Diffuse component
                            if self.sources.diffuse:

                                with IfBlockContext(
                                    [StringExpression([k, " == ", k_diff])]
                                ):

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            self._diff_spectrum_lpdf(
                                                self._Esrc[i],
                                                self._diff_index,
                                                self._Esrc_min,
                                                self._Esrc_max,
                                            ),
                                        ]
                                    )
                                    self._E[i] << StringExpression(
                                        [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                    )

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            np.log(1 / (4 * np.pi)),
                                        ]
                                    )

                            # Atmospheric component
                            if self.sources.atmospheric:

                                with IfBlockContext(
                                    [StringExpression([k, " == ", k_atmo])]
                                ):

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            FunctionCall(
                                                [
                                                    self._atmo_flux_func(
                                                        self._Esrc[i],
                                                        self._omega_det[i],
                                                    )
                                                    / self._atmo_flux_integral
                                                ],
                                                "log",
                                            ),
                                        ]
                                    )
                                    self._E[i] << self._Esrc[i]

                            # Detection effects
                            StringExpression(
                                [
                                    self._lp[i][k],
                                    " += ",
                                    self._dm["tracks"].energy_resolution(
                                        self._E[i], self._Edet[i]
                                    ),
                                ]
                            )
                            StringExpression(
                                [
                                    self._lp[i][k],
                                    " += log(interpolate(",
                                    self._Eg,
                                    ", ",
                                    self._Pg_t[k],
                                    ", ",
                                    self._E[i],
                                    "))",
                                ]
                            )

                # Cascades
                if "cascades" in self._event_types:

                    with IfBlockContext(
                        [
                            StringExpression(
                                [self._event_type[i], " == ", self._cascade_type]
                            )
                        ]
                    ):

                        with ForLoopContext(1, n_comps_max, "k") as k:

                            # Point source components
                            if self.sources.point_source:

                                with IfBlockContext(
                                    [StringExpression([k, " < ", self._Ns + 1])]
                                ):

                                    if self._shared_src_index:
                                        src_index_ref = self._src_index
                                    else:
                                        src_index_ref = self._src_index[k]

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            self._src_spectrum_lpdf(
                                                self._Esrc[i],
                                                src_index_ref,
                                                self._Esrc_min,
                                                self._Esrc_max,
                                            ),
                                        ]
                                    )
                                    self._E[i] << StringExpression(
                                        [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                    )

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += vMF_lpdf(",
                                            self._omega_det[i],
                                            " | ",
                                            self._varpi[k],
                                            ", ",
                                            self._kappa[i],
                                            ")",
                                        ]
                                    )

                            # Diffuse component
                            if self.sources.diffuse:

                                with IfBlockContext(
                                    [StringExpression([k, " == ", self._Ns + 1])]
                                ):

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            self._diff_spectrum_lpdf(
                                                self._Esrc[i],
                                                self._diff_index,
                                                self._Esrc_min,
                                                self._Esrc_max,
                                            ),
                                        ]
                                    )
                                    self._E[i] << StringExpression(
                                        [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                    )
                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            np.log(1 / (4 * np.pi)),
                                        ]
                                    )

                            # Atmospheric component
                            if self.sources.atmospheric:

                                with IfBlockContext(
                                    [StringExpression([k, " == ", self._Ns + 2])]
                                ):

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += negative_infinity()",
                                        ]
                                    )
                                    self._E[i] << self._Esrc[i]

                            # Detection effects
                            StringExpression(
                                [
                                    self._lp[i][k],
                                    " += ",
                                    self._dm["cascades"].energy_resolution(
                                        self._E[i], self._Edet[i]
                                    ),
                                ]
                            )
                            StringExpression(
                                [
                                    self._lp[i][k],
                                    " += log(interpolate(",
                                    self._Eg,
                                    ", ",
                                    self._Pg_c[k],
                                    ", ",
                                    self._E[i],
                                    "))",
                                ]
                            )

    def _model(self):

        with ModelContext():

            # Likelihood
            with ForLoopContext(1, self._N, "i") as i:

                StringExpression(["target += log_sum_exp(", self._lp[i], ")"])

            StringExpression(["target += -", self._Nex])

            # Priors
            if self.sources.point_source:

                if self._priors.luminosity.name in ["normal", "lognormal"]:

                    StringExpression(
                        [
                            self._L,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._priors.luminosity.mu,
                                    self._priors.luminosity.sigma,
                                ],
                                self._priors.luminosity.name,
                            ),
                        ]
                    )

                elif self._priors.luminosity.name == "pareto":

                    StringExpression(
                        [
                            self._L,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._priors.luminosity.xmin,
                                    self._priors.luminosity.alpha,
                                ],
                                self._priors.luminosity.name,
                            ),
                        ]
                    )

                else:

                    raise NotImplementedError(
                        "Luminosity prior distribution not recognised."
                    )

                StringExpression(
                    [
                        self._src_index,
                        " ~ ",
                        FunctionCall(
                            [self._priors.src_index.mu, self._priors.src_index.sigma],
                            self._priors.src_index.name,
                        ),
                    ]
                )

            if self.sources.diffuse:

                StringExpression(
                    [
                        self._F_diff,
                        " ~ ",
                        FunctionCall(
                            [
                                self._priors.diffuse_flux.mu,
                                self._priors.diffuse_flux.sigma,
                            ],
                            self._priors.diffuse_flux.name,
                        ),
                    ]
                )

                StringExpression(
                    [
                        self._diff_index,
                        " ~ ",
                        FunctionCall(
                            [self._priors.diff_index.mu, self._priors.diff_index.sigma],
                            self._priors.diff_index.name,
                        ),
                    ]
                )

            if self.sources.atmospheric:

                StringExpression(
                    [
                        self._F_atmo,
                        " ~ ",
                        FunctionCall(
                            [
                                self._priors.atmospheric_flux.mu,
                                self._priors.atmospheric_flux.sigma,
                            ],
                            self._priors.atmospheric_flux.name,
                        ),
                    ]
                )
