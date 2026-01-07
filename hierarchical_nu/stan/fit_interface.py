import numpy as np
from typing import List
from collections import OrderedDict

from ..source.parameter import Parameter
from ..priors import Priors, MultiSourcePrior
from ..stan.interface import StanInterface

from ..backend.stan_generator import (
    FunctionsContext,
    Include,
    DataContext,
    DummyContext,
    TransformedDataContext,
    ParametersContext,
    TransformedParametersContext,
    GeneratedQuantitiesContext,
    ForLoopContext,
    IfBlockContext,
    ElseBlockContext,
    ElseIfBlockContext,
    ModelContext,
    FunctionCall,
    UserDefinedFunction,
)

from ..backend.expression import (
    ReturnStatement,
    StringExpression,
)

from ..backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ParameterDef,
    ParameterVectorDef,
    InstantVariableDef,
)

from ..backend.expression import StringExpression
from ..backend.parameterizations import DistributionMode

from ..source.flux_model import LogParabolaSpectrum
from ..source.source import Sources
from ..source.parameter import Parameter
from ..detector.icecube import EventType, NT, CAS
from ..detector.detector_model import (
    GridInterpolationEnergyResolution,
)
from ..detector.r2021 import R2021EnergyResolution


class StanFitInterface(StanInterface):
    """
    An interface for generating the Stan fit code.
    """

    def __init__(
        self,
        output_file: str,
        sources: Sources,
        event_types: List[EventType],
        atmo_flux_energy_points: int = 100,
        atmo_flux_theta_points: int = 30,
        includes: List[str] = [
            "interpolation.stan",
            "utils.stan",
            "vMF.stan",
            "power_law.stan",
        ],
        priors: Priors = Priors(),
        nshards: int = 1,
        use_event_tag: bool = False,
        debug: bool = False,
        bg: bool = False,
    ):
        """
        An interface for generating Stan fit code.

        :param output_file: Name of the file to write to
        :param sources: Sources object containing sources to be fit
        :param event_types: Type of the detector model to be used
        :param atmo_flux_theta_points: Number of points to use for the grid of
        atmospheric flux
        :param includes: List of names of stan files to include into the
        functions block of the generated file
        :param priors: Priors object detailing the priors to use
        :param nshards: Number of shards for multithreading, defaults to zero
        :param use_event_tag: if True, only consider the closest PS for each event
        :param debug: if True, add function calls for debugging and tests
        :param bg: if True, use data to construct background likelihood
        """

        super().__init__(
            output_file=output_file,
            sources=sources,
            event_types=event_types,
            includes=includes,
        )

        self._priors = priors

        if self._shared_luminosity and isinstance(
            self._priors.luminosity, MultiSourcePrior
        ):
            raise ValueError("Shared luminosity requires a single prior")
        if self._shared_src_index:
            if isinstance(self._priors.src_index, MultiSourcePrior):
                raise ValueError("Shared src_index requires a single prior")
            if isinstance(self._priors.beta_index, MultiSourcePrior):
                raise ValueError("Shared beta_index requires a single prior")
            if isinstance(self._priors.E0_src, MultiSourcePrior):
                raise ValueError("Shared E0_src requires a single prior")

        self._atmo_flux_energy_points = atmo_flux_energy_points

        self._atmo_flux_theta_points = atmo_flux_theta_points

        if not isinstance(nshards, int):
            raise ValueError("nshards must be an integer")

        if not nshards >= 0:
            raise ValueError("nshards must not be negative")

        self._nshards = nshards
        self._use_event_tag = use_event_tag
        self._debug = debug
        self._bg = bg

        n_params = 0
        n_params += 1 if self._fit_index else 0
        n_params += 1 if self._fit_beta else 0
        n_params += 1 if self._fit_Enorm else 0
        n_params += 1 if self._fit_eta else 0

        self._n_params = n_params

        self._dm = OrderedDict()

        for et in self._event_types:
            detector_model_type = et.model

            if detector_model_type.PDF_FILENAME not in self._includes:
                self._includes.append(detector_model_type.PDF_FILENAME)
            detector_model_type.generate_code(
                DistributionMode.PDF,
                rewrite=False,
            )

        try:
            ang_sys = Parameter.get_parameter("ang_sys_add")
            self._ang_sys = True
            self._fit_ang_sys = not ang_sys.fixed
        except ValueError:
            self._ang_sys = False
            self._fit_ang_sys = False
        # In case of pgamma create dummy references to hardcoded parameters
        # which are otherwise needed in for-loops
        # TODO: find better solution, possibly to actually use these but set fixed=True
        if self._pgamma:
            self._src_index = None
            self._beta_index = None

    def _model_likelihood(self):
        """
        Write the likelihood part of the model.
        Is reused three times in the python code and up to two times
        when generating, depending on the configutation.
        """

        with ForLoopContext(1, self._N, "i") as i:
            if self._bg and not self._use_event_tag:
                self._lp[i, 1 : self._Ns] << self._logF
                self._lp[i, self._k_bg] << self._log_N_bg + self._bg_llh[
                    i
                ] - FunctionCall([self._E[i]], "log")

            elif not self._use_event_tag:
                # TODO fix this case later for use with data-background llh
                self._lp[i] << self._logF

            elif self._bg:
                self._lp[i, 2] << self._log_N_bg + self._bg_llh[i] - FunctionCall(
                    [self._E[i]], "log"
                )

            for c, event_type in enumerate(self._event_types):
                if c == 0:
                    context = IfBlockContext
                else:
                    context = ElseIfBlockContext
                with context(
                    [
                        StringExpression(
                            [
                                self._event_type[i],
                                " == ",
                                event_type.S,
                            ]
                        )
                    ]
                ):
                    # Detection effects
                    if self._use_event_tag:
                        ps_pos = self._varpi[self._event_tag[i]]
                    else:
                        ps_pos = self._varpi
                    if event_type in [NT, CAS]:
                        # patch this out?
                        self._irf_return << self._dm[event_type](
                            self._E[i],
                            self._Edet[i],
                            self._omega_det[i],
                            ps_pos,
                        )
                    elif isinstance(
                        self._dm[event_type].energy_resolution,
                        GridInterpolationEnergyResolution,
                    ):
                        if not (self.sources.diffuse or self.sources.atmospheric):
                            self._irf_return << self._dm[event_type](
                                self._E[i],
                                ps_pos,
                                FunctionCall([self._ereco_grid[i]], "to_vector"),
                            )
                        else:
                            self._irf_return << self._dm[event_type](
                                self._E[i],
                                self._omega_det[i],
                                ps_pos,
                                FunctionCall([self._ereco_grid[i]], "to_vector"),
                            )

                    """
                    # TODO: fix later...
                    # remind me, what was I to do later?
                    else:
                        self._irf_return << self._dm[event_type](
                            self._E[i],
                            self._omega_det[i],
                            ps_pos,
                            self._Edet[i],
                        )
                    """
                    if self.sources.point_source and self._fit_ang_sys:
                        # Calculate spatial likelihood if fitting ang_sys_add(_squared)
                        with ForLoopContext(1, self._Ns, "k") as k:
                            self._spatial_loglike[k, i] << FunctionCall(
                                [
                                    self._angular_separation[k, i],
                                    self._ang_errs_squared[i]
                                    + self._ang_sys_add_squared,
                                    # self._kappa[i],
                                ],
                                event_type.F + "AngularResolution",
                            )

            self._eres_src << StringExpression(["irf_return.1"])
            self._aeff_src << StringExpression(["irf_return.2"])
            if self.sources.diffuse or self.sources.atmospheric:
                self._eres_diff << StringExpression(["irf_return.3"])
                self._aeff_diff << StringExpression(["irf_return.4"])

            # Sum over sources => evaluate and store components
            with ForLoopContext(1, self._Ns_tot, "k") as k:
                # Point source components
                if self.sources.point_source:
                    with IfBlockContext([StringExpression([k, " < ", self._Ns + 1])]):
                        if self._use_event_tag:
                            # Create new reference to proper entry in lp to reuse more code
                            _lp = self._lp[i][1]
                            _eres_src = self._eres_src
                            _aeff_src = self._aeff_src
                            # If the source label k does not match the tag, continue
                            # with condition k < Ns + 1 this does not interfere with diffuse components
                            with IfBlockContext([k, "!=", self._event_tag[i]]):
                                StringExpression(["continue"])
                            with ElseBlockContext():
                                _lp << self._logF[k]
                        else:
                            _lp = self._lp[i][k]
                            _eres_src = self._eres_src[k]
                            _aeff_src = self._aeff_src[k]

                        StringExpression(
                            [
                                _lp,
                                " += ",
                                self._spatial_loglike[k, i],
                            ]
                        )
                        StringExpression([_lp, " += ", _aeff_src])
                        StringExpression([_lp, " += ", _eres_src])

                        # Create references only if they are definitely needed
                        # Either if param is fitted, or if we need all references, even to data
                        # (case for generated quantities, i.e. _x_r_idxs does not exist)

                        if self._logparabola or self._pgamma:

                            # create even more references
                            # go through all three params
                            self._refs = [
                                self._src_index,
                                self._beta_index,
                                self._E0_src,
                            ]

                            first_param = True
                            theta = ["{"]
                            for f, r in zip(self._fit, self._refs):
                                if f:
                                    if not first_param:
                                        theta.append(",")
                                    theta.append(r[k])
                                    first_param = False
                            theta.append("}")
                            theta = StringExpression(theta)

                            """
                            logic:
                            if x_r_idx is present, the not-fitted param may not be referenced using indexing

                            if x_r_idx is not present, create all references
                            """
                            try:
                                # If this works, we are coming from lp_reduce
                                # use self._x_r_idxs to get k-th entry of Emin/max_src, E0
                                # should work the same for fitting E0
                                # because we substitute the same number of entries in real data
                                x_r = ["real_data[{"]
                                for l in range(1, self._x_r_len + 1):
                                    x_r.append(self._x_r_idxs[l] + k - 1)
                                    if l < self._x_r_len:
                                        x_r.append(",")
                                x_r.append("}]")
                                x_r = StringExpression(x_r)

                                """
                                x_r = StringExpression(
                                    ["real_data[", self._x_r_idxs, "]"]
                                )
                                """
                                del self._x_r_idxs
                            except AttributeError as e:
                                data = ["{"]
                                first_data = True
                                for f, r in zip(self._fit, self._refs):
                                    if not f and self._logparabola:
                                        if not first_data:
                                            data.append(",")
                                        data.append(r[k])
                                        first_data = False
                                # Otherwise single thread or generated quantities
                                if self._logparabola:
                                    data.append(",")
                                data += [
                                    self._Emin_src[k],
                                    ",",
                                    self._Emax_src[k],
                                    "}",
                                ]
                                x_r = StringExpression(data)
                            x_i = StringExpression(
                                [
                                    "{",
                                    0,
                                    "}",
                                ]
                            )
                            StringExpression(
                                [
                                    _lp,
                                    " += ",
                                    self._src_spectrum_lpdf(
                                        self._E[i],
                                        theta,
                                        x_r,
                                        x_i,
                                    ),
                                ]
                            )
                        elif self._seyfert:
                            if len(self.sources.point_source) == 1:
                                StringExpression(
                                    [
                                        _lp,
                                        " += ",
                                        self._src_spectrum_lpdf[0](
                                            self._E[i],
                                            self._eta[k],
                                        ),
                                    ]
                                )
                            else:
                                for j in range(1, len(self.sources.point_source) + 1):
                                    if j == 1:
                                        context = IfBlockContext
                                    else:
                                        context = ElseIfBlockContext
                                    with context([k, " == ", j]):
                                        StringExpression(
                                            [
                                                _lp,
                                                " += ",
                                                self._src_spectrum_lpdf[j - 1](
                                                    self._E[i],
                                                    self._eta[k],
                                                ),
                                            ]
                                        )
                        else:
                            StringExpression(
                                [
                                    _lp,
                                    " += ",
                                    self._src_spectrum_lpdf(
                                        self._E[i],
                                        self._src_index[k],
                                        self._Emin_src[k],
                                        self._Emax_src[k],
                                    ),
                                ]
                            )

                # Diffuse component
                if self.sources.diffuse:
                    with IfBlockContext([StringExpression([k, " == ", self._k_diff])]):
                        if self._use_event_tag:
                            _lp = self._lp[i][2]
                            StringExpression([_lp, "=", self._logF[k]])
                        else:
                            _lp = self._lp[i][k]
                        StringExpression(
                            [
                                _lp,
                                " += ",
                                self._eres_diff,
                            ]
                        )

                        StringExpression(
                            [
                                _lp,
                                " += ",
                                self._aeff_diff,
                            ]
                        )

                        # log_prob += log(p(Esrc|diff_index))
                        StringExpression(
                            [
                                _lp,
                                " += ",
                                self._diff_spectrum_lpdf(
                                    self._E[i],
                                    self._diff_index,
                                    self._Emin_diff,
                                    self._Emax_diff,
                                ),
                            ]
                        )

                        # log_prob += log(1/4pi)
                        StringExpression(
                            [
                                _lp,
                                " += ",
                                np.log(1 / (4 * np.pi)),
                            ]
                        )

                # Atmospheric component
                if self.sources.atmospheric:
                    with IfBlockContext([StringExpression([k, " == ", self._k_atmo])]):
                        if self._use_event_tag:
                            if self.sources.diffuse:
                                _lp = self._lp[i][3]
                            else:
                                _lp = self._lp[i][2]
                            _lp << self._logF[k]
                        else:
                            _lp = self._lp[i][k]
                        StringExpression(
                            [
                                _lp,
                                " += ",
                                self._eres_diff,
                            ]
                        )

                        if CAS in self._event_types:
                            # The amount of indentation is too damn high
                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            self._event_type[i],
                                            " == ",
                                            CAS.S,
                                        ]
                                    )
                                ]
                            ):
                                StringExpression(
                                    [
                                        _lp,
                                        " += negative_infinity()",
                                    ]
                                )
                            with ElseBlockContext():
                                StringExpression(
                                    [
                                        _lp,
                                        " +=",
                                        self._aeff_diff,
                                    ]
                                )
                        else:
                            StringExpression(
                                [
                                    _lp,
                                    " += ",
                                    self._aeff_diff,
                                ]
                            )

                        # log_prob += log(p(Esrc, omega | atmospheric source))
                        StringExpression(
                            [
                                _lp,
                                " += ",
                                FunctionCall(
                                    [
                                        self._atmo_flux_func(
                                            self._E[i],
                                            self._omega_det[i],
                                        )
                                        / self._atmo_integrated_flux
                                    ],
                                    "log",
                                ),
                            ]
                        )
            """
            if hasattr(self, "_log_lik"):
                # If self._log_lik exists, fill it with data
                self._log_lik[i] << FunctionCall(
                    [self._lp[i]], "log_sum_exp"
                ) - FunctionCall([self._Nex], "log")
            """

    def _functions(self):
        """
        Write the functions section of the Stan file.
        """

        with FunctionsContext():
            # Include all the specified files
            for include_file in self._includes:
                _ = Include(include_file)

            for et in self._event_types:
                self._dm[et] = et.model(mode=DistributionMode.PDF)
                # Event tag is equivalent to only requesting one point source
                diffuse = self.sources.diffuse or self.sources.atmospheric
                self._dm[et].generate_pdf_function_code(
                    single_ps=self._use_event_tag, diffuse=diffuse
                )
            # If we have point sources, include the shape of their PDF
            # and how to convert from energy to number flux
            if self.sources.point_source:
                if self._logparabola or self._pgamma:
                    self._ps_spectrum.make_stan_utility_func(
                        self._fit_index, self._fit_beta, self._fit_Enorm
                    )
                if self._seyfert:
                    self._src_spectrum_lpdf, self._src_flux_table, self._flux_conv = (
                        self._sources.make_seyfert_functions()
                    )
                else:
                    self._src_spectrum_lpdf = self._ps_spectrum.make_stan_lpdf_func(
                        "src_spectrum_logpdf",
                        self._fit_index,
                        self._fit_beta,
                        self._fit_Enorm,
                    )

                    self._flux_conv = self._ps_spectrum.make_stan_flux_conv_func(
                        "flux_conv",
                        self._fit_index,
                        self._fit_beta,
                        self._fit_Enorm,
                    )

            # If we have diffuse sources, include the shape of their PDF
            if self.sources.diffuse:
                self._diff_spectrum_lpdf = self._diff_spectrum.make_stan_lpdf_func(
                    "diff_spectrum_logpdf"
                )
                self._diff_spectrum_flux_conv = (
                    self._diff_spectrum.make_stan_diff_flux_conv_func("diff_flux_conv")
                )

            # If we have atmospheric sources, include the atmospheric flux table
            # the density of the grid in theta (i.e. declination) is specified here
            if self.sources.atmospheric:
                # Increasing theta points too much makes compilation very slow
                # Could switch to passing array as data if problematic
                self._atmo_flux_func = self._atmo_flux.make_stan_function(
                    energy_points=self._atmo_flux_energy_points,
                    theta_points=self._atmo_flux_theta_points,
                )

            if self._nshards not in [0, 1]:
                # Create a function to be used in map_rect in the model block
                # Signature is determined by stan's `map_rect` function
                lp_reduce = UserDefinedFunction(
                    "lp_reduce",
                    ["global", "local", "real_data", "int_data"],
                    ["vector", "vector", "data array[] real", "data array[] int"],
                    "vector",
                )

                with lp_reduce:
                    # Unpack integer data, needed to interpret real data
                    # Use InstantVariableDef to save on lines
                    self._N = InstantVariableDef("N", "int", ["int_data[1]"])
                    self._Ns = InstantVariableDef("Ns", "int", ["int_data[2]"])
                    glob = StringExpression(["global"])
                    loc = StringExpression(["local"])
                    int_data = StringExpression(["int_data"])
                    real_data = StringExpression(["real_data"])

                    if self.sources.diffuse and self.sources.atmospheric:
                        self._Ns_tot = "Ns+2"
                    elif (
                        self.sources.diffuse
                        or self.sources.atmospheric
                        or self.sources.background
                    ):
                        self._Ns_tot = "Ns+1"
                    else:
                        self._Ns_tot = "Ns"

                    if self._bg:
                        self._Ns_tot_flux = "Ns"
                    else:
                        self._Ns_tot_flux = self._Ns_tot

                    start = InstantVariableDef("start", "int", [1])
                    end = InstantVariableDef("end", "int", [0])
                    length = ForwardVariableDef("length", "int")
                    self._x_r_len = 2
                    if self._logparabola:
                        self._x_r_len += 1 if not self._fit_index else 0
                        self._x_r_len += 1 if not self._fit_beta else 0
                        self._x_r_len += 1 if not self._fit_Enorm else 0
                    self._x_r_idxs = ForwardArrayDef(
                        "x_r_idxs", "int", [f"[{self._x_r_len}]"]
                    )
                    # Get global parameters
                    if self._fit_index:
                        end << end + self._Ns
                        self._src_index = ForwardVariableDef("src_index", "vector[Ns]")
                        self._src_index << glob[start:end]
                        start << start + self._Ns
                    if self._fit_beta:
                        end << end + self._Ns
                        self._beta_index = ForwardVariableDef(
                            "beta_index", "vector[Ns]"
                        )
                        self._beta_index << glob[start:end]
                        start << start + self._Ns
                    if self._fit_Enorm:
                        end << end + self._Ns
                        self._E0_src = ForwardVariableDef("E0_src", "vector[Ns]")
                        self._E0_src << glob[start:end]
                        start << start + self._Ns
                    if self._fit_eta:
                        end << end + self._Ns
                        self._eta = ForwardVariableDef("eta", "vector[Ns]")
                        self._eta << glob[start:end]
                        start << start + self._Ns

                    # Get diffuse index
                    if self.sources.diffuse:
                        end << end + 1
                        self._diff_index = ForwardVariableDef("diff_index", "real")
                        self._diff_index << glob[start]
                        start << start + 1
                    if self._fit_ang_sys:
                        self._ang_sys_add_squared = ForwardVariableDef(
                            "ang_sys_add_squared", "real"
                        )
                        end << end + 1
                        self._ang_sys_add_squared << glob[start]
                        start << start + 1
                    end << end + self._Ns_tot_flux
                    self._logF = ForwardVariableDef(
                        "logF", "vector[" + self._Ns_tot_flux + "]"
                    )
                    self._logF << glob[start:end]
                    start << start + self._Ns_tot_flux
                    if self._bg:
                        end << end + 1
                        self._log_N_bg = ForwardVariableDef("log_N_bg", "real")
                        self._log_N_bg << glob[start]
                        start << start + 1

                    # Local pars are only neutrino energies
                    self._E = ForwardVariableDef("E", "vector[N]")
                    self._E << loc[1 : self._N]

                    # Define variable to store loglikelihood
                    if self._use_event_tag:
                        size = 1
                        if self._sources.diffuse:
                            size += 1
                        if self._sources.atmospheric:
                            size += 1
                        # reduce lp to 3 components per event since we only allow for one PS association
                        self._lp = ForwardArrayDef(
                            "lp", "vector[" + str(size) + "]", ["[N]"]
                        )
                    else:
                        self._lp = ForwardArrayDef(
                            "lp", "vector[" + self._Ns_tot + "]", ["[N]"]
                        )

                    # Unpack event types (Tracks, cascades, IC40...)
                    self._event_type = ForwardArrayDef("event_type", "int", ["[N]"])

                    # Define indices for unpacking of real_data
                    start << 3
                    end << 2 + self._N

                    self._event_type << int_data[start:end]

                    if self._use_event_tag:
                        self._event_tag = ForwardArrayDef("event_tag", "int", ["[N]"])
                        start << start + self._N
                        end << end + self._N
                        self._event_tag << StringExpression(["int_data[3+N:2+2*N]"])

                    # self._ereco_idx = ForwardArrayDef("ereco_idx", "int", ["[N]"])
                    # self._ereco_idx << StringExpression("int_data[3+N:2+2*N]")

                    # Define indices for unpacking of real_data
                    start << 1
                    length << self._N
                    end << self._N

                    self._Edet = ForwardVariableDef("Edet", "vector[N]")
                    self._Edet << FunctionCall([real_data[start:end]], "to_vector")
                    # Shift indices appropriate amount for next batch of data
                    start << start + length
                    grid_size = R2021EnergyResolution._log_tE_grid.size
                    self._ereco_grid = ForwardArrayDef(
                        "ereco_grid", "real", ["[N, ", str(grid_size), "]"]
                    )

                    with ForLoopContext(1, "N", "f") as f:
                        end << end + grid_size
                        self._ereco_grid[f] << real_data[start:end]
                        start << start + grid_size
                    self._omega_det = ForwardArrayDef("omega_det", "vector[3]", ["[N]"])
                    # Loop over events to unpack reconstructed direction
                    with ForLoopContext(1, self._N, "i") as i:
                        end << end + 3
                        self._omega_det[i] << FunctionCall(
                            [real_data[start:end]], "to_vector"
                        )
                        start << start + 3

                    if self._bg:
                        end << end + self._N
                        self._bg_llh = ForwardVariableDef("bg_llh", "vector[N]")
                        self._bg_llh << FunctionCall(
                            [real_data[start:end]], "to_vector"
                        )
                        start << start + self._N

                    self._varpi = ForwardArrayDef("varpi", "vector[3]", ["[Ns]"])
                    # Loop over sources to unpack source direction (for point sources only)
                    with ForLoopContext(1, self._Ns, "i") as i:
                        end << end + 3
                        self._varpi[i] << FunctionCall(
                            [real_data[start:end]], "to_vector"
                        )
                        start << start + 3
                    # If diffuse source, z is longer by 1 element
                    if self.sources.diffuse:
                        end << end + self._Ns + 1
                        self._z = ForwardVariableDef("z", "vector[Ns+1]")
                        self._z << FunctionCall([real_data[start:end]], "to_vector")
                        start << start + self._Ns + 1
                    else:
                        end << end + self._Ns
                        self._z = ForwardVariableDef("z", "vector[Ns]")
                        self._z << FunctionCall([real_data[start:end]], "to_vector")
                        start << start + self._Ns

                    if self.sources.atmospheric:
                        self._atmo_integrated_flux = ForwardVariableDef(
                            "atmo_integrated_flux", "real"
                        )
                        end << end + 1
                        (self._atmo_integrated_flux << real_data[start])
                        start << start + 1

                    if self.sources.point_source:
                        self._spatial_loglike = ForwardArrayDef(
                            "spatial_loglike", "real", ["[Ns, N]"]
                        )
                        if not self._fit_ang_sys:
                            with ForLoopContext(1, self._Ns, "k") as k:
                                end << end + length
                                (self._spatial_loglike[k] << real_data[start:end])
                                start << start + length
                        else:
                            self._ang_errs_squared = ForwardArrayDef(
                                "ang_errs_squared", "real", ["[N]"]
                            )
                            end << end + length
                            self._ang_errs_squared << real_data[start:end]
                            start << start + length

                            self._angular_separation = ForwardArrayDef(
                                "angular_separation", "real", ["[Ns, N]"]
                            )
                            with ForLoopContext(1, self._Ns, "k") as k:
                                end << end + length
                                self._angular_separation[k] << real_data[start:end]
                                start << start + length

                    self._Emin_src = ForwardArrayDef("Emin_src", "real", ["[Ns]"])
                    self._Emax_src = ForwardArrayDef("Emax_src", "real", ["[Ns]"])
                    self._Emin = ForwardVariableDef("Emin", "real")
                    self._Emax = ForwardVariableDef("Emax", "real")
                    if self.sources.diffuse:
                        self._Emin_diff = ForwardVariableDef("Emin_diff", "real")
                        self._Emax_diff = ForwardVariableDef("Emax_diff", "real")

                    # Insert Emin_src
                    end << end + self._Ns
                    self._Emin_src << StringExpression(["real_data[start:end]"])
                    if self._logparabola or self._pgamma:
                        self._x_r_idxs[self._x_r_len - 1] << start
                    start << start + self._Ns

                    # Insert Emax_src
                    end << end + self._Ns
                    self._Emax_src << StringExpression(["real_data[start:end]"])
                    if self._logparabola or self._pgamma:
                        self._x_r_idxs[self._x_r_len] << start
                    start << start + self._Ns

                    if self.sources.diffuse:
                        end << end + 1
                        (self._Emin_diff << real_data[start])
                        start << start + 1

                        end << end + 1
                        (self._Emax_diff << real_data[start])
                        start << start + 1

                    end << end + 1
                    (self._Emin << real_data[start])
                    start << start + 1

                    end << end + 1
                    (self._Emax << real_data[start])
                    start << start + 1

                    data_idx = 1
                    if (self._power_law or self._logparabola) and not self._fit_index:
                        end << end + self._Ns
                        self._src_index = ForwardArrayDef("src_index", "real", ["[Ns]"])
                        self._src_index << real_data[start:end]
                        self._x_r_idxs[data_idx] << start
                        data_idx += 1
                        start << start + self._Ns
                    if self._logparabola and not self._fit_beta:
                        end << end + self._Ns
                        self._beta_index = ForwardArrayDef(
                            "beta_index", "real", ["[Ns]"]
                        )
                        self._beta_index << real_data[start:end]
                        self._x_r_idxs[data_idx] << start
                        data_idx += 1
                        start << start + self._Ns
                    if self._logparabola and not self._fit_Enorm:
                        end << end + self._Ns
                        self._E0_src = ForwardArrayDef("E0_src", "real", ["[Ns]"])
                        self._E0_src << real_data[start:end]
                        self._x_r_idxs[data_idx] << start
                        data_idx += 1
                        start << start + self._Ns

                    # Define tracks and cascades to sort events into correct detector response
                    if self._use_event_tag and not (
                        self.sources.diffuse or self.sources.atmospheric
                    ):
                        self._irf_return = ForwardVariableDef(
                            "irf_return",
                            "tuple(real, real)",
                        )
                    elif self._use_event_tag:
                        self._irf_return = ForwardVariableDef(
                            "irf_return",
                            "tuple(real, real, real, real)",
                        )
                    elif self.sources.diffuse or self.sources.atmospheric:
                        self._irf_return = ForwardVariableDef(
                            "irf_return",
                            "tuple(array[Ns] real, array[Ns] real, real, real)",
                        )
                    else:
                        self._irf_return = ForwardVariableDef(
                            "irf_return",
                            "tuple(array[Ns] real, array[Ns] real)",
                        )

                    if not self._use_event_tag:
                        self._eres_src = ForwardArrayDef("eres_src", "real", ["[Ns]"])
                        self._aeff_src = ForwardArrayDef("aeff_src", "real", ["[Ns]"])
                    else:
                        self._eres_src = ForwardVariableDef("eres_src", "real")
                        self._aeff_src = ForwardVariableDef("aeff_src", "real")

                    if self.sources.diffuse or self.sources.atmospheric:
                        self._eres_diff = ForwardVariableDef("eres_diff", "real")
                        self._aeff_diff = ForwardVariableDef("aeff_diff", "real")

                    if self.sources.diffuse and self.sources.atmospheric:
                        self._k_diff = "Ns + 1"
                        self._k_atmo = "Ns + 2"

                    elif self.sources.diffuse:
                        self._k_diff = "Ns + 1"

                    elif self.sources.atmospheric:
                        self._k_atmo = "Ns + 1"

                    elif self.sources.background:
                        self._k_bg = "Ns + 1"

                    self._model_likelihood()
                    results = ForwardArrayDef("results", "real", ["[N]"])
                    with ForLoopContext(1, self._N, "i") as i:
                        results[i] << FunctionCall([self._lp[i]], "log_sum_exp")
                    if self._debug:
                        # Only for debugging purposes, this makes fits for large N really slow
                        ReturnStatement([FunctionCall([results], "to_vector")])
                    else:
                        ReturnStatement(["[sum(results)]'"])

    def _data(self):
        with DataContext():
            # Total number of detected events
            self._N = ForwardVariableDef("N", "int")
            self._N_str = ["[", self._N, "]"]

            # Number of point sources
            self._Ns = ForwardVariableDef("Ns", "int")
            self._Ns_str = ["[", self._Ns, "]"]
            self._Ns_1p_str = ["[", self._Ns, "+1]"]
            self._Ns_2p_str = ["[", self._Ns, "+2]"]

            # Total number of sources
            self._Ns_tot = ForwardVariableDef("Ns_tot", "int")

            if self.sources.diffuse and self._n_params == 1:
                self._Ns_string_int_grid = "Ns+1"
            elif self.sources.diffuse:
                self._Ns_string_int_grid = "1"
            elif self._n_params == 1:
                self._Ns_string_int_grid = "Ns"

            if self._nshards not in [0, 1]:
                # Number of shards for multi-threading
                self._N_shards = ForwardVariableDef("N_shards", "int")
                self._N_shards_str = ["[", self._N_shards, "]"]

                # Max number of events per shard
                self._J = ForwardVariableDef("J", "int")

            # Detected directions as unit vectors
            self._omega_det = ForwardArrayDef(
                "omega_det", "unit_vector[3]", self._N_str
            )

            # Dected energies
            self._Edet = ForwardVariableDef("Edet", "vector[N]")

            # Angular uncertainty, 0.683 coverage in one coordinate
            self._ang_errs = ForwardVariableDef("ang_err", "vector[N]")
            # Added (in quadrature) angular uncertainty to be fit, done in transformed data

            # Event types as track/cascades
            self._event_type = ForwardVariableDef("event_type", "vector[N]")

            # Uncertainty on the event's angular reconstruction
            self._kappa = ForwardVariableDef("kappa", "vector[N]")

            if self._bg:
                self._bg_llh = ForwardVariableDef("bg_llh", "vector[N]")

            if self._fit_nex:
                self._Nex_src_min = ForwardVariableDef("Nex_src_min", "real")
                self._Nex_src_max = ForwardVariableDef("Nex_src_max", "real")

            # Event tags
            if self._use_event_tag:
                self._event_tag = ForwardArrayDef("event_tag", "int", self._N_str)

            # To store the Ereco-grid index for each event, speeds up the 2d interpolation
            # self._ereco_idx = ForwardArrayDef("ereco_idx", "int", self._N_str)
            grid_size = R2021EnergyResolution._log_tE_grid.size
            self._ereco_grid = ForwardArrayDef(
                "ereco_grid", "real", ["[N, ", str(grid_size), "]"]
            )

            # Energy range at source
            self._Emin_src = ForwardArrayDef("Emin_src", "real", ["[Ns]"])
            self._Emax_src = ForwardArrayDef("Emax_src", "real", ["[Ns]"])

            if self.sources.diffuse:
                self._Emin_diff = ForwardVariableDef("Emin_diff", "real")
                self._Emax_diff = ForwardVariableDef("Emax_diff", "real")
                self._Enorm_diff = ForwardVariableDef("Enorm_diff", "real")

            # Energy range at the detector
            self._Emin = ForwardVariableDef("Emin", "real")
            self._Emax = ForwardVariableDef("Emax", "real")
            if self.sources.point_source:
                if not self._seyfert:
                    self._Lmin = ForwardVariableDef("Lmin", "real")
                    self._Lmax = ForwardVariableDef("Lmax", "real")
                if self._seyfert:
                    self._Pmin = ForwardVariableDef("P_min", "real")
                    self._Pmax = ForwardVariableDef("P_max", "real")
                if self._fit_index:
                    self._src_index_min = ForwardVariableDef("src_index_min", "real")
                    self._src_index_max = ForwardVariableDef("src_index_max", "real")
                elif self._logparabola:
                    self._src_index = ForwardArrayDef("src_index", "real", ["[Ns]"])
                if self._fit_beta:
                    self._beta_index_min = ForwardVariableDef("beta_index_min", "real")
                    self._beta_index_max = ForwardVariableDef("beta_index_max", "real")
                elif self._logparabola:
                    self._beta_index = ForwardArrayDef("beta_index", "real", ["[Ns]"])
                if self._fit_Enorm:
                    self._E0_src_min = ForwardVariableDef("E0_src_min", "real")
                    self._E0_src_max = ForwardVariableDef("E0_src_max", "real")
                elif self._logparabola:
                    self._E0_src = ForwardArrayDef("E0", "real", ["[Ns]"])

            if self.sources.diffuse:
                self._diff_index_min = ForwardVariableDef("diff_index_min", "real")
                self._diff_index_max = ForwardVariableDef("diff_index_max", "real")
                self._diffuse_norm_min = ForwardVariableDef("diffuse_norm_min", "real")
                self._diffuse_norm_max = ForwardVariableDef("diffuse_norm_max", "real")

            if self.sources.atmospheric:
                self._F_atmo_min = ForwardVariableDef("F_atmo_min", "real")
                self._F_atmo_max = ForwardVariableDef("F_atmo_max", "real")

            # True directions and distances of point sources
            self._varpi = ForwardArrayDef("varpi", "unit_vector[3]", self._Ns_str)
            self._D = ForwardVariableDef("D", "vector[Ns]")

            # Density of interpolation grid and energy grid points
            self._Ngrid = ForwardVariableDef("Ngrid", "int")

            # Observation time
            self._T = ForwardArrayDef("T", "real", ["[", self._Net, "]"])

            # Redshift
            if self.sources.diffuse:
                self._z = ForwardVariableDef("z", "vector[Ns+1]")

            else:
                self._z = ForwardVariableDef("z", "vector[Ns]")

            # Interpolation grid points in spectral indices for
            # sources with spectral index as a free param, and
            # the exposure integral evaluated at these points, for
            # different event types (i.e. different Aeff)
            if self.sources.point_source or self.sources.diffuse:
                if self.sources.point_source:
                    if self._fit_index:
                        self._src_index_grid = ForwardVariableDef(
                            "src_index_grid", "vector[Ngrid]"
                        )
                    else:
                        # create dummy attributes to be able to loop over all possible grids
                        # the stan variables are only used if they exist
                        self._src_index_grid = 0
                    if self._fit_beta:
                        self._beta_index_grid = ForwardVariableDef(
                            "beta_index_grid", "vector[Ngrid]"
                        )
                    else:
                        self._beta_index_grid = 0
                    if self._fit_Enorm:
                        self._E0_src_grid = ForwardVariableDef(
                            "E0_src_grid", "vector[Ngrid]"
                        )
                    else:
                        self._E0_src_grid = 0
                    if self._fit_eta:
                        self._eta_min = ForwardVariableDef("eta_min", "real")
                        self._eta_max = ForwardVariableDef("eta_max", "real")
                        self._eta_grid = ForwardVariableDef("eta_grid", "vector[Ngrid]")

                    if self._n_params == 2:
                        self._integral_grid_2d = ForwardArrayDef(
                            "integral_grid_2d",
                            "real",
                            [
                                "[",
                                self._Net,
                                ",",
                                self._Ns,
                                ",",
                                self._Ngrid,
                                ",",
                                self._Ngrid,
                                "]",
                            ],
                        )
                if self.sources.diffuse or self._n_params == 1:
                    self._integral_grid = ForwardArrayDef(
                        "integral_grid",
                        "vector[Ngrid]",
                        ["[", self._Net, ",", self._Ns_string_int_grid, "]"],
                    )

                if self.sources.diffuse:
                    self._diff_index_grid = ForwardVariableDef(
                        "diff_index_grid", "vector[Ngrid]"
                    )

            # Don't need a grid for atmo as spectral shape is fixed, so pass single value.
            if self.sources.atmospheric:
                self._atmo_integ_val = ForwardArrayDef(
                    "atmo_integ_val", "real", ["[", self._Net, "]"]
                )
                self._atmo_integrated_flux = ForwardVariableDef(
                    "atmo_integrated_flux", "real"
                )

            if self._sources.point_source:
                # Define variables for the prior mu/sigma
                if self._priors.src_index.name == "notaprior":
                    pass
                elif (
                    isinstance(self._priors.src_index, MultiSourcePrior)
                    and self._fit_index
                ):
                    index_mu_def = ForwardArrayDef("src_index_mu", "real", self._Ns_str)
                    index_sigma_def = ForwardArrayDef(
                        "src_index_sigma", "real", self._Ns_str
                    )
                elif self._fit_index:
                    index_mu_def = ForwardVariableDef("src_index_mu", "real")
                    index_sigma_def = ForwardVariableDef("src_index_sigma", "real")
                if (
                    isinstance(self._priors.beta_index, MultiSourcePrior)
                    and self._fit_beta
                ):
                    beta_mu_def = ForwardArrayDef("beta_index_mu", "real", self._Ns_str)
                    beta_sigma_def = ForwardArrayDef(
                        "beta_index_sigma", "real", self._Ns_str
                    )
                elif self._fit_beta:
                    beta_mu_def = ForwardVariableDef("beta_index_mu", "real")
                    beta_sigma_def = ForwardVariableDef("beta_index_sigma", "real")
                if (
                    isinstance(self._priors.E0_src, MultiSourcePrior)
                    and self._fit_Enorm
                ):
                    E0_src_mu_def = ForwardArrayDef("E0_src_mu", "real", self._Ns_str)
                    E0_src_sigma_def = ForwardArrayDef(
                        "E0_src_sigma", "real", self._Ns_str
                    )
                elif self._fit_Enorm:
                    E0_src_mu_def = ForwardVariableDef("E0_src_mu", "real")
                    E0_src_sigma_def = ForwardVariableDef("E0_src_sigma", "real")

                if self._priors.eta.name == "notaprior":
                    pass
                elif isinstance(self._priors.eta, MultiSourcePrior) and self._fit_eta:
                    eta_mu_def = ForwardArrayDef("eta_mu", "real", self._Ns_str)
                    eta_sigma_def = ForwardArrayDef("eta_sigma", "real", self._Ns_str)
                elif self._fit_eta:
                    eta_mu_def = ForwardVariableDef("eta_mu", "real")
                    eta_sigma_def = ForwardVariableDef("eta_sigma", "real")

                # Store prior data definitions
                if self._fit_index and self._priors.src_index.name in [
                    "normal",
                    "lognormal",
                ]:
                    self._stan_prior_src_index_mu = index_mu_def
                    self._stan_prior_src_index_sigma = index_sigma_def
                if self._fit_beta:
                    self._stan_prior_beta_index_mu = beta_mu_def
                    self._stan_prior_beta_index_sigma = beta_sigma_def
                if self._fit_Enorm:
                    self._stan_prior_E0_src_mu = E0_src_mu_def
                    self._stan_prior_E0_src_sigma = E0_src_sigma_def
                if self._fit_eta and self._priors.eta.name != "notaprior":
                    self._stan_prior_eta_mu = eta_mu_def
                    self._stan_prior_eta_sigma = eta_sigma_def
                # check for luminosity, if they all have the same prior
                if self._fit_nex or self._seyfert:
                    if self._priors.Nex_src.name != "notaprior":
                        self._stan_prior_nex_mu = ForwardVariableDef("Nex_mu", "real")
                        self._stan_prior_nex_sigma = ForwardVariableDef("Nex_sigma", "real")
                elif self._priors.luminosity.name in ["normal", "lognormal"]:
                    if isinstance(self._priors.luminosity, MultiSourcePrior):
                        mu_def = ForwardArrayDef("lumi_mu", "real", self._Ns_str)
                        sigma_def = ForwardArrayDef("lumi_sigma", "real", self._Ns_str)
                    else:
                        mu_def = ForwardVariableDef("lumi_mu", "real")
                        sigma_def = ForwardVariableDef("lumi_sigma", "real")
                    self._stan_prior_lumi_mu = mu_def
                    self._stan_prior_lumi_sigma = sigma_def
                elif self._priors.luminosity.name == "pareto":
                    if isinstance(self._priors.luminosity, MultiSourcePrior):
                        xmin_def = ForwardArrayDef("lumi_xmin", "real", self._Ns_str)
                        alpha_def = ForwardArrayDef("lumi_alpha", "real", self._Ns_str)
                    else:
                        xmin_def = ForwardVariableDef("lumi_xmin", "real")
                        alpha_def = ForwardVariableDef("lumi_alpha", "real")

                    self._stan_prior_lumi_xmin = xmin_def
                    self._stan_prior_lumi_alpha = alpha_def

                if self._fit_nex:
                    pass
                elif self._seyfert:
                    if self._priors.pressure_ratio.name == "notaprior":
                        pass
                    elif isinstance(self._priors.pressure_ratio, MultiSourcePrior):
                        mu_def = ForwardArrayDef("P_mu", "real", self._Ns_str)
                        sigma_def = ForwardArrayDef("P_sigma", "real", self._Ns_str)
                    else:
                        mu_def = ForwardVariableDef("P_mu", "real")
                        sigma_def = ForwardVariableDef("P_sigma", "real")
                    if self._priors.pressure_ratio.name != "notaprior":
                        self._stan_prior_pressure_ratio_mu = mu_def
                        self._stan_prior_pressure_ratio_sigma = sigma_def

            if self._sources.diffuse:
                self._stan_prior_f_diff_mu = ForwardVariableDef("f_diff_mu", "real")
                self._stan_prior_f_diff_sigma = ForwardVariableDef(
                    "f_diff_sigma", "real"
                )

                self._stan_prior_diff_index_mu = ForwardVariableDef(
                    "diff_index_mu", "real"
                )
                self._stan_prior_diff_index_sigma = ForwardVariableDef(
                    "diff_index_sigma", "real"
                )

            if self._sources.atmospheric:
                self._stan_prior_f_atmo_mu = ForwardVariableDef("f_atmo_mu", "real")
                self._stan_prior_f_atmo_sigma = ForwardVariableDef(
                    "f_atmo_sigma", "real"
                )

            if self._fit_ang_sys:
                self._ang_sys_add_min = ForwardVariableDef("ang_sys_min", "real")
                self._ang_sys_add_max = ForwardVariableDef("ang_sys_max", "real")
                self._ang_sys_mu = ForwardVariableDef("ang_sys_mu", "real")
                self._ang_sys_sigma = ForwardVariableDef("ang_sys_sigma", "real")
                if self._priors.ang_sys.name == "exponnorm":
                    self._ang_sys_lam = ForwardVariableDef("ang_sys_lam", "real")

    def _transformed_data(self):
        """
        To write the transformed data section of the Stan file.
        """

        with TransformedDataContext():

            # Count different event types and create variables to store counts
            self._et_stan = ForwardArrayDef("event_types", "int", ["[", self._Net, "]"])
            self._Net_stan = ForwardVariableDef("Net", "int")
            self._Net_stan << StringExpression(["size(event_types)"])

            if self._sources.point_source:
                # Angular separation between sources and events, in rad
                self._angular_separation = ForwardArrayDef(
                    "angular_separation", "real", ["[Ns, N]"]
                )
                with ForLoopContext(1, self._Ns, "k") as k:
                    with ForLoopContext(1, self._N, "i") as i:
                        self._angular_separation[k, i] << FunctionCall(
                            [self._varpi[k], self._omega_det[i]], "ang_sep"
                        )

                # If we do fit angular systematics, we only need the squared uncertainties
                self._ang_errs_squared = ForwardVariableDef(
                    "ang_errs_squared", "vector[N]"
                )
                self._ang_errs_squared << self._ang_errs**2

            for c, et in enumerate(self._event_types, 1):
                self._et_stan[c] << et.S

            if self.sources.point_source:
                # Vector to hold pre-calculated spatial loglikes
                # This needs to be compatible with multiple point sources!
                if not self._fit_ang_sys:
                    self._spatial_loglike = ForwardArrayDef(
                        "spatial_loglike", "real", ["[Ns, N]"]
                    )
                    with ForLoopContext(1, self._N, "i") as i:
                        with ForLoopContext(1, self._Ns, "k") as k:
                            # Insert loop over event types
                            for c, event_type in enumerate(self._event_types):
                                if c == 0:
                                    context = IfBlockContext
                                else:
                                    context = ElseIfBlockContext
                                with context(
                                    [
                                        StringExpression(
                                            [
                                                self._event_type[i],
                                                " == ",
                                                event_type.S,
                                            ]
                                        )
                                    ]
                                ):
                                    # Determine which arguments are needed for the
                                    # needed angular resolution
                                    if event_type in [NT, CAS]:
                                        self._spatial_loglike[k, i] << FunctionCall(
                                            [
                                                self._varpi[k],
                                                self._omega_det[i],
                                                self._kappa[i],
                                            ],
                                            event_type.F + "AngularResolution",
                                        )
                                    else:
                                        self._spatial_loglike[k, i] << FunctionCall(
                                            [
                                                self._angular_separation[k, i],
                                                self._ang_errs_squared[i],
                                            ],
                                            event_type.F + "AngularResolution",
                                        )

            # Find largest permitted range of energies at the detector
            self._Emin_at_det = ForwardVariableDef("Emin_at_det", "real")
            self._Emax_at_det = ForwardVariableDef("Emax_at_det", "real")
            self._Emin_at_det << self._Emin
            self._Emax_at_det << self._Emax

            # Find the largest energy range over all source components
            with ForLoopContext(1, self._Ns, "k") as k:
                with IfBlockContext(
                    [
                        self._Emin_src[k],
                        " < ",
                        self._Emin_at_det,
                    ]
                ):
                    self._Emin_at_det << self._Emin_src[k]
                with IfBlockContext(
                    [
                        self._Emax_src[k],
                        " > ",
                        self._Emax_at_det,
                    ]
                ):
                    self._Emax_at_det << self._Emax_src[k]
            if self.sources.diffuse:
                with IfBlockContext(
                    [
                        self._Emin_diff,
                        " < ",
                        self._Emin_at_det,
                    ]
                ):
                    self._Emin_at_det << self._Emin_diff
                with IfBlockContext(
                    [
                        self._Emax_diff,
                        " > ",
                        self._Emax_at_det,
                    ]
                ):
                    self._Emax_at_det << self._Emax_diff

            if self._nshards not in [0, 1]:
                grid_size = R2021EnergyResolution._log_tE_grid.size
                self._N_shards_use_this = ForwardVariableDef("N_shards_loop", "int")
                self._N_shards_use_this << self._N_shards
                # Create the rectangular data blocks for use in `map_rect`
                self._N_mod_J = ForwardVariableDef("N_mod_J", "int")
                self._N_mod_J << self._N % self._J
                # Find size for real_data array
                sd_events_J = (
                    4 + grid_size
                )  # reco energy, reco dir (unit vector, counts as 3 entries), eres grid
                if self._bg:
                    sd_events_J += 1  # one bg llh entry per event
                sd_if_diff = 3  # redshift of diffuse component, Emin_diff/max
                sd_Ns = 6  # redshift, Emin_src, Emax_src, x, y, z per point source
                sd_other = 2  # Emin, Emax
                # Need Ns * N for spatial loglike, added extra in sd_string -> J*Ns
                if self._fit_ang_sys:
                    # If we use ang_sys we need to pass ang_errs_squared
                    sd_events_J += 1
                if self.sources.atmospheric:
                    # atmo_integrated_flux, why was this here before? not used as far as I can see
                    sd_other += 1  # no atmo in cascades
                sd_string = f"{sd_events_J}*J + {sd_Ns}*Ns + {sd_other}"

                # If we do not use ang_sys we need to pass the fixed spatial loglike for J events x Ns sources
                # and if we do we need to pass angular separations between each source and event (disregard use_event_tag for now) TODO for me
                sd_string += " + J*Ns"
                if self.sources.diffuse:
                    sd_string += f" + {sd_if_diff}"

                # additional data dependent on which parameters are being fit
                # for simplicity of code use one entry for each source
                # even if the value is the same across sources
                # less case distinctions, thank you very much
                if self._logparabola:
                    if not self._fit_index:
                        sd_string += "+Ns"
                    if not self._fit_beta:
                        sd_string += "+Ns"
                    if not self._fit_Enorm:
                        sd_string += "+Ns"

                # Create data arrays
                self.real_data = ForwardArrayDef(
                    "real_data", "real", ["[N_shards,", sd_string, "]"]
                )

                if self._use_event_tag:
                    size = "2+2*J"
                else:
                    size = "2+J"
                self.int_data = ForwardArrayDef(
                    "int_data", "int", ["[", self._N_shards, ", ", size, "]"]
                )

                # Pack data into shards
                # Format is (obviously) the same as the unpacking done in `lp_reduce`
                # First dimension is number of shard, second dimension is what `lp_reduce` will see
                with ForLoopContext(1, self._N_shards, "i") as i:
                    start = ForwardVariableDef("start", "int")
                    end = ForwardVariableDef("end", "int")
                    insert_start = ForwardVariableDef("insert_start", "int")
                    insert_end = ForwardVariableDef("insert_end", "int")
                    insert_len = ForwardVariableDef("insert_len", "int")
                    start << (i - 1) * self._J + 1
                    insert_start << 1
                    end << i * self._J

                    with IfBlockContext([start, ">", self._N]):
                        self._N_shards_use_this << i - 1
                        self._N_shards_str = ["[", self._N_shards_use_this, "]"]
                        StringExpression(["break"])
                    with IfBlockContext([end, ">", self._N]):
                        end << self._N
                        insert_len << end - start + 1
                        insert_end << insert_len
                    with ElseBlockContext():
                        insert_len << self._J
                        insert_end << insert_len

                    self.real_data[i, insert_start:insert_end] << FunctionCall(
                        [self._Edet[start:end]], "to_array_1d"
                    )
                    insert_start << insert_start + insert_len

                    with ForLoopContext(start, end, "f") as f:
                        insert_end << insert_end + grid_size
                        (
                            self.real_data[i, insert_start:insert_end]
                            << self._ereco_grid[f]
                        )
                        insert_start << insert_start + grid_size

                    with ForLoopContext(start, end, "f") as f:
                        insert_end << insert_end + 3
                        self.real_data[i, insert_start:insert_end] << FunctionCall(
                            [self._omega_det[f]], "to_array_1d"
                        )
                        insert_start << insert_start + 3

                    if self._bg:
                        insert_end << insert_end + insert_len
                        self.real_data[i, insert_start:insert_end] << FunctionCall(
                            [self._bg_llh[start:end]], "to_array_1d"
                        )
                        insert_start << insert_start + insert_len

                    with ForLoopContext(1, self._Ns, "f") as f:
                        insert_end << insert_end + 3
                        self.real_data[i, insert_start:insert_end] << FunctionCall(
                            [self._varpi[f]], "to_array_1d"
                        )
                        insert_start << insert_start + 3

                    if self.sources.diffuse:
                        insert_end << insert_end + self._Ns + 1
                        self.real_data[i, insert_start:insert_end] << FunctionCall(
                            [self._z], "to_array_1d"
                        )
                        insert_start << insert_start + self._Ns + 1

                    else:
                        insert_end << insert_end + self._Ns
                        self.real_data[i, insert_start:insert_end] << FunctionCall(
                            [self._z], "to_array_1d"
                        )
                        insert_start << insert_start + self._Ns

                    if self.sources.atmospheric:
                        insert_end << insert_end + 1
                        (self.real_data[i, insert_start] << self._atmo_integrated_flux)
                        insert_start << insert_start + 1

                    if self.sources.point_source:
                        if not self._fit_ang_sys:
                            with ForLoopContext(1, self._Ns, "k") as k:
                                # Loop over sources
                                insert_end << insert_end + insert_len
                                # The double-index is needed because of a bug with the code generator
                                # if I use [k, start:end], a single line of "k;" is printed after entering
                                # the for loop
                                # TODO: fix this in code generator
                                (
                                    self.real_data[i, insert_start:insert_end]
                                    << self._spatial_loglike[k][start:end]
                                )
                                insert_start << insert_start + insert_len
                        else:
                            # TODO raise error if this is used with NT or CAS
                            insert_end << insert_end + insert_len
                            (
                                self.real_data[i, insert_start:insert_end]
                                << FunctionCall(
                                    [self._ang_errs_squared[start:end]], "to_array_1d"
                                )
                            )
                            insert_start << insert_start + insert_len

                            with ForLoopContext(1, self._Ns, "k") as k:
                                insert_end << insert_end + insert_len
                                (
                                    self.real_data[i, insert_start:insert_end]
                                    << self._angular_separation[k, start:end]
                                )
                                insert_start << insert_start + insert_len

                    insert_end << insert_end + self._Ns
                    self.real_data[i, insert_start:insert_end] << self._Emin_src
                    insert_start << insert_start + self._Ns

                    insert_end << insert_end + self._Ns
                    self.real_data[i, insert_start:insert_end] << self._Emax_src
                    insert_start << insert_start + self._Ns

                    if self.sources.diffuse:
                        insert_end << insert_end + 1
                        self.real_data[i, insert_start] << self._Emin_diff
                        insert_start << insert_start + 1

                        insert_end << insert_end + 1
                        self.real_data[i, insert_start] << self._Emax_diff
                        insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Emin
                    insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Emax
                    insert_start << insert_start + 1

                    # Any of the spectral parameters not used as a fit parameter
                    # have length Ns
                    if self._logparabola and not self._fit_index:
                        insert_end << insert_end + self._Ns
                        self.real_data[i, insert_start:insert_end] << self._src_index
                        insert_start << insert_start + self._Ns
                    if self._logparabola and not self._fit_beta:
                        insert_end << insert_end + self._Ns
                        self.real_data[i, insert_start:insert_end] << self._beta_index
                        insert_start << insert_start + self._Ns
                    if self._logparabola and not self._fit_Enorm:
                        insert_end << insert_end + self._Ns
                        self.real_data[i, insert_start:insert_end] << self._E0_src
                        insert_start << insert_start + self._Ns

                    # Pack integer data so real_data can be sorted into correct blocks in `lp_reduce`
                    self.int_data[i, 1] << insert_len
                    self.int_data[i, 2] << self._Ns
                    insert_start << 3
                    # end index is inclusive, subtract one
                    insert_end << insert_start + insert_len - 1
                    self.int_data[i, insert_start:insert_end] << FunctionCall(
                        [FunctionCall([self._event_type[start:end]], "to_array_1d")],
                        "to_int",
                    )

                    insert_start << insert_start + insert_len
                    if self._use_event_tag:
                        insert_end << insert_end + insert_len
                        (
                            self.int_data[i, insert_start:insert_end]
                            << self._event_tag[start:end]
                        )
                        insert_start << insert_start + insert_len
                    """
                    insert_end << insert_end + insert_len
                    (
                        self.int_data[i, insert_start:insert_end]
                        << self._ereco_idx[start:end]
                    )
                    """

    def _parameters(self):
        """
        To write the parameters section of the Stan file.
        """

        with ParametersContext():
            # For point sources, L and src_index can be shared or
            # independent.
            if self.sources.point_source:
                if self._fit_nex:
                    self._Nex_per_ps = ParameterVectorDef(
                        "Nex_per_ps",
                        "vector",
                        self._Ns_str,
                        self._Nex_src_min,
                        self._Nex_src_max,
                    )

                elif self._seyfert:
                    # TODO make vector option for multi ps with individual parameters,
                    # hijack shared luminosity for this
                    if self._shared_luminosity:
                        self._P_glob = ParameterDef(
                            "pressure_ratio", "real", self._Pmin, self._Pmax
                        )
                    else:
                        self._P = ParameterVectorDef(
                            "pressure_ratio",
                            "vector",
                            self._Ns_str,
                            self._Pmin,
                            self._Pmax,
                        )

                else:
                    if self._shared_luminosity:
                        self._L_glob = ParameterDef("L", "real", self._Lmin, self._Lmax)
                    else:
                        self._L = ParameterVectorDef(
                            "L",
                            "vector",
                            self._Ns_str,
                            self._Lmin,
                            self._Lmax,
                        )

                if self._shared_src_index:
                    if self._fit_index:
                        self._src_index_glob = ParameterDef(
                            "src_index",
                            "real",
                            self._src_index_min,
                            self._src_index_max,
                        )
                    if self._fit_beta:
                        self._beta_index_glob = ParameterDef(
                            "beta_index",
                            "real",
                            self._beta_index_min,
                            self._beta_index_max,
                        )
                    if self._fit_Enorm:
                        self._E0_src_glob = ParameterDef(
                            "E0_src",
                            "real",
                            self._E0_src_min,
                            self._E0_src_max,
                        )
                    if self._fit_eta:
                        self._eta_glob = ParameterDef(
                            "eta",
                            "real",
                            self._eta_min,
                            self._eta_max,
                        )

                else:
                    if self._fit_index:
                        self._src_index = ParameterVectorDef(
                            "src_index",
                            "vector",
                            self._Ns_str,
                            self._src_index_min,
                            self._src_index_max,
                        )
                    if self._fit_beta:
                        self._beta_index = ParameterVectorDef(
                            "beta_index",
                            "vector",
                            self._Ns_str,
                            self._beta_min,
                            self._beta_max,
                        )
                    if self._fit_Enorm:
                        self._E0_src_glob = ParameterVectorDef(
                            "E0_src",
                            "vector",
                            self._Ns_str,
                            self._E0_src_min,
                            self._E0_src_max,
                        )
                    if self._fit_eta:
                        self._eta = ParameterVectorDef(
                            "eta", "vector", self._Ns_str, self._eta_min, self._eta_max
                        )

            # Specify F_diff and diff_index to characterise the diffuse comp
            if self.sources.diffuse:
                self._diffuse_norm = ParameterDef(
                    "diffuse_norm",
                    "real",
                    self._diffuse_norm_min,
                    self._diffuse_norm_max,
                )
                self._diff_index = ParameterDef(
                    "diff_index", "real", self._diff_index_min, self._diff_index_max
                )

            # Atmo spectral shape is fixed, but normalisation can move.
            if self.sources.atmospheric:
                self._F_atmo = ParameterDef(
                    "F_atmo", "real", self._F_atmo_min, self._F_atmo_max
                )

            if self._bg:
                self._Nex_bg = ParameterDef(
                    "Nex_bg",
                    "real",
                    0,
                )

            # Vector of latent true neutrino energy for each event
            self._E = ParameterVectorDef(
                "E", "vector", self._N_str, self._Emin_at_det, self._Emax_at_det
            )

            if self._fit_ang_sys:
                self._ang_sys_add = ParameterDef(
                    "ang_sys_add", "real", self._ang_sys_add_min, self._ang_sys_add_max
                )

    def _transformed_parameters(self):
        """
        To write the transformed parameters section of the Stan file.
        """

        with TransformedParametersContext():
            # Expected number of events for different source components (atmo, diff, src) and detector components (_comp)
            self._Nex = ForwardVariableDef("Nex", "real")
            self._Nex_comp = ForwardArrayDef("Nex_comp", "real", ["[", self._Net, "]"])
            self._flux_conv_val = ForwardArrayDef("flux_conv_val", "real", self._Ns_str)
            if self.sources.atmospheric:
                self._Nex_atmo = ForwardVariableDef("Nex_atmo", "real")
                self._Nex_atmo_comp = ForwardArrayDef(
                    "Nex_atmo_comp", "real", ["[", self._Net, "]"]
                )
            if self.sources.point_source:
                if self._shared_luminosity or self._fit_nex or self._seyfert:
                    self._L = ForwardVariableDef(
                        "L_ind",
                        "vector[Ns]",
                    )
                if self._shared_src_index:
                    if self._fit_index:
                        self._src_index = ForwardVariableDef(
                            "src_index_ind", "vector[Ns]"
                        )
                    if self._fit_beta:
                        self._beta_index = ForwardVariableDef(
                            "beta_index_ind", "vector[Ns]"
                        )
                    if self._fit_eta:
                        self._eta = ForwardVariableDef("eta_ind", "vector[Ns]")
                if self._fit_nex and self._seyfert:
                    self._P = ForwardVariableDef("pressure_ratio", "vector[Ns]")
                elif self._shared_luminosity and self._seyfert:
                    self._P = ForwardVariableDef("pressure_ratio_ind", "vector[Ns]")
                if self._fit_Enorm:
                    self._E0_src = ForwardVariableDef("E0_src_ind", "vector[Ns]")
                if self._shared_luminosity or self._shared_src_index:
                    with ForLoopContext(1, self._Ns, "k") as k:
                        if (
                            self._shared_luminosity
                            and not self._fit_nex
                            and not self._seyfert
                        ):
                            self._L[k] << self._L_glob
                        if (
                            self._shared_luminosity
                            and self._seyfert
                            and not self._fit_nex
                        ):
                            self._P[k] << self._P_glob
                        if self._shared_src_index and self._fit_index:
                            self._src_index[k] << self._src_index_glob
                        if self._shared_src_index and self._fit_beta:
                            self._beta_index[k] << self._beta_index_glob
                        if self._shared_src_index and self._fit_Enorm:
                            # Account for the fact that E0 is only sensibly defined in the source frame
                            # meaning that E0_src_glob is defined in the source frame
                            # and E0_src[k] is redshifted using z[k]
                            self._E0_src[k] << self._E0_src_glob / (1 + self._z[k])
                        if self._shared_src_index and self._fit_eta:
                            self._eta[k] << self._eta_glob
                if self._fit_ang_sys:
                    self._ang_sys_add_squared = ForwardVariableDef(
                        "ang_sys_add_squared", "real"
                    )
                    self._ang_sys_add_squared << self._ang_sys_add**2

                if not self._shared_src_index and self._fit_Enorm:
                    with ForLoopContext(1, self._Ns, "k") as k:
                        # Define E0_src in the detector frame and E0_src_glob in source frame
                        self._E0_src[k] << self._E0_src_glob[k] / (1 + self._z[k])

                self._Nex_src = ForwardVariableDef("Nex_src", "real")
                self._Nex_src_comp = ForwardArrayDef(
                    "Nex_src_comp", "real", ["[", self._Net, "]"]
                )

                with ForLoopContext(1, self._Net_stan, "i") as i:
                    self._Nex_src_comp[i] << 0
                if not self._fit_nex:
                    self._Nex_per_ps = ForwardArrayDef("Nex_per_ps", "real", ["[Ns]"])
                    with ForLoopContext(1, self._Ns, "i") as i:
                        self._Nex_per_ps[i] << 0
            if self.sources.diffuse:
                self._F_diff = ForwardVariableDef("F_diff", "real")
                self._F_diff << self._diffuse_norm * self._diff_spectrum_flux_conv(
                    self._Enorm_diff,
                    self._diff_index,
                    self._Emin_diff,
                    self._Emax_diff,
                )
                self._Nex_diff = ForwardVariableDef("Nex_diff", "real")
                self._Nex_diff_comp = ForwardArrayDef(
                    "Nex_diff_comp", "real", ["[", self._Net, "]"]
                )

            if self.sources.background:
                self._log_N_bg = ForwardVariableDef("log_N_bg", "real")
                self._log_N_bg << FunctionCall([self._Nex_bg], "log")

            # Total flux
            # self._Ftot = ForwardVariableDef("Ftot", "real")

            # Total flux form point sources
            # self._F_src = ForwardVariableDef("Fs", "real")

            # Different definitions of fractional association
            # self._f_arr = ForwardVariableDef("f_arr", "real")
            # self._f_det = ForwardVariableDef("f_det", "real")
            # self._f_arr_astro = ForwardVariableDef("f_arr_astro", "real")
            # self._f_det_astro = ForwardVariableDef("f_det_astro", "real")

            # Decide how many source components we have and calculate
            # `logF` accordingly.
            # These will be used as weights in the mixture model likelihood
            # We store the log probability for each source comp in `lp`.
            if self.sources.diffuse and self.sources.atmospheric:
                self._F = ForwardVariableDef("F", "vector[Ns+2]")
                self._logF = ForwardVariableDef("logF", "vector[Ns+2]")

                if self._nshards in [0, 1]:
                    if self._use_event_tag:
                        self._lp = ForwardArrayDef("lp", "vector[3]", self._N_str)
                    else:
                        self._lp = ForwardArrayDef("lp", "vector[Ns+2]", self._N_str)

                n_comps_max = "Ns+2"

            elif self.sources.diffuse or self.sources.atmospheric:
                self._F = ForwardVariableDef("F", "vector[Ns+1]")
                self._logF = ForwardVariableDef("logF", "vector[Ns+1]")

                if self._nshards in [0, 1]:
                    if self._use_event_tag:
                        self._lp = ForwardArrayDef("lp", "vector[2]", self._N_str)
                    else:
                        self._lp = ForwardArrayDef("lp", "vector[Ns+1]", self._N_str)

                n_comps_max = "Ns+1"

            else:
                # Even if background likelihood is used, only point sources may have a flux
                self._F = ForwardVariableDef("F", "vector[Ns]")
                self._logF = ForwardVariableDef("logF", "vector[Ns]")

                if self._nshards in [0, 1]:
                    if self._use_event_tag:
                        if self._bg:
                            # for PS + background use one entry for all PS and one for background
                            self._lp = ForwardArrayDef("lp", "vector[2]", self._N_str)
                        else:
                            self._lp = ForwardArrayDef("lp", "vector[1]", self._N_str)
                    else:
                        if self._bg:
                            # for PS + background Ns entries for PSs and one for background
                            self._lp = ForwardArrayDef(
                                "lp", "vector[Ns+1]", self._N_str
                            )
                        else:
                            self._lp = ForwardArrayDef("lp", "vector[Ns]", self._N_str)

                # No exposure for background likelihood
                n_comps_max = "Ns"

            self._eps = ForwardArrayDef(
                "eps", "vector[" + n_comps_max + "]", ["[", self._Net, "]"]
            )

            if self._nshards not in [0, 1]:
                # Create vector of parameters
                # Global pars are src_index, optionally beta_index, diff_index, logF
                # Count number of pars:
                # start with logF of point sources
                num_of_pars = "Ns"

                python_counter = 0

                if self._fit_index:
                    python_counter += 1
                if self._fit_beta:
                    python_counter += 1
                if self._fit_Enorm:
                    python_counter += 1
                if self._fit_eta:
                    python_counter += 1
                # always use individual, transformed parameters, even if shared_src_index
                num_of_pars += f" + Ns * {python_counter:d}"

                par_counter = 0
                if self.sources.diffuse:
                    par_counter += 2
                if self.sources.atmospheric:
                    par_counter += 1
                if self.sources.background:
                    par_counter += 1
                if self._fit_ang_sys:
                    par_counter += 1
                num_of_pars += f" + {par_counter:d}"

                self._global_pars = ForwardVariableDef(
                    "global_pars", f"vector[{num_of_pars}]"
                )

                self._local_pars = ForwardArrayDef(
                    "local_pars", "vector[J]", self._N_shards_str
                )

                # Pack source energies into local parameter vector
                with ForLoopContext(1, self._N_shards_use_this, "i") as i:
                    start = ForwardVariableDef("start", "int")
                    end = ForwardVariableDef("end", "int")
                    start << (i - 1) * self._J + 1
                    end << i * self._J

                    # If it's not the last shard or all shards have same length anyway:
                    with IfBlockContext([end, ">", self._N]):
                        length = ForwardVariableDef("length", "int")
                        length << self._N - start + 1
                        self._local_pars[i][1:length] << self._E[start : self._N]

                    # Else, only relevant for last shard if it's shorter
                    with ElseBlockContext():
                        self._local_pars[i] << self._E[start:end]

            else:
                if self._use_event_tag and not (
                    self.sources.diffuse or self.sources.atmospheric
                ):
                    self._irf_return = ForwardVariableDef(
                        "irf_return",
                        "tuple(real, real)",
                    )
                elif self._use_event_tag:
                    self._irf_return = ForwardVariableDef(
                        "irf_return",
                        "tuple(real, real, real, real)",
                    )
                elif self.sources.diffuse or self.sources.atmospheric:
                    self._irf_return = ForwardVariableDef(
                        "irf_return",
                        "tuple(array[Ns] real, array[Ns] real, real, real)",
                    )
                else:
                    self._irf_return = ForwardVariableDef(
                        "irf_return",
                        "tuple(array[Ns] real, array[Ns] real)",
                    )
                if not self._use_event_tag:
                    self._eres_src = ForwardArrayDef("eres_src", "real", ["[Ns]"])
                    self._aeff_src = ForwardArrayDef("aeff_src", "real", ["[Ns]"])
                else:
                    self._eres_src = ForwardVariableDef("eres_src", "real")
                    self._aeff_src = ForwardVariableDef("aeff_src", "real")

                if self.sources.diffuse or self.sources.atmospheric:
                    self._eres_diff = ForwardVariableDef("eres_diff", "real")
                    self._aeff_diff = ForwardVariableDef("aeff_diff", "real")

            # self._F_src << 0.0
            self._Nex_src << 0.0

            # For each source, calculate the number flux and update F, logF
            if self.sources.point_source:
                with ForLoopContext(1, self._Ns, "k") as k:

                    if not self._fit_nex and not self._seyfert:
                        self._F[k] << StringExpression(
                            [
                                self._L[k],
                                "/ (4 * pi() * pow(",
                                self._D[k],
                                " * ",
                                3.086e22,
                                ", 2))",
                            ]
                        )
                    elif not self._fit_nex:
                        # Use pressure_ratio * integrated flux (latter is normalised to the pressure ratio used in sims)
                        if len(self.sources.point_source) == 1:
                            self._F[k] << self._P[k] * self._src_flux_table[0](
                                self._eta[k]
                            )
                        else:
                            for j in range(1, len(self.sources.point_source) + 1):
                                if j == 1:
                                    context = IfBlockContext
                                else:
                                    context == ElseIfBlockContext
                                with context([k, " == ", j]):
                                    self._F[k] << self._P[k] * self._src_flux_table[
                                        j - 1
                                    ](self._eta[k])

                    if self._logparabola or self._pgamma:
                        # create even more references
                        # go through all three params
                        fit = [self._fit_index, self._fit_beta, self._fit_Enorm]
                        refs = [self._src_index, self._beta_index, self._E0_src]
                        grids = [
                            self._src_index_grid,
                            self._beta_index_grid,
                            self._E0_src_grid,
                        ]
                        first_param = True
                        first_data = True
                        theta = ["{"]
                        data = ["{"]
                        for f, r, g in zip(fit, refs, grids):
                            if f:
                                if not first_param:
                                    if self._n_params == 2:
                                        p2 = r[k]
                                        g2 = g
                                    theta.append(",")
                                else:
                                    if self._n_params == 2:
                                        p1 = r[k]
                                        g1 = g
                                theta.append(r[k])
                                first_param = False
                            elif self._logparabola:
                                # put the leftovers in the fridge, please
                                if not first_data:
                                    data.append(",")
                                data.append(r[k])
                                first_data = False

                        theta.append("}")
                        theta = StringExpression(theta)

                        if self._logparabola:
                            data.append(",")
                        data += [
                            self._Emin_src[k],
                            ",",
                            self._Emax_src[k],
                            "}",
                        ]
                        x_r = StringExpression(data)
                        x_i = StringExpression(
                            [
                                "{",
                                0,
                                "}",
                            ]
                        )
                        self._flux_conv_val[k] << self._flux_conv(
                            theta,
                            x_r,
                            x_i,
                        )

                    elif self._seyfert:
                        if len(self.sources.point_source) == 1:
                            self._flux_conv_val[k] << self._flux_conv[0](
                                self._eta[k],
                            )
                        else:
                            for j in range(1, len(self.sources.point_source) + 1):
                                if j == 1:
                                    context = IfBlockContext
                                else:
                                    context = ElseIfBlockContext
                                with context([k, " == ", j]):
                                    self._flux_conv_val[k] << self._flux_conv[j - 1](
                                        self._eta[k]
                                    )
                    else:
                        self._flux_conv_val[k] << self._flux_conv(
                            self._src_index[k],
                            self._Emin_src[k],
                            self._Emax_src[k],
                        )
                    if not self._fit_nex and not self._seyfert:
                        StringExpression([self._F[k], "*=", self._flux_conv_val[k]])

                    with ForLoopContext(1, self._Net_stan, "i") as i:
                        # For each source, calculate the exposure via interpolation
                        # and then the expected number of events
                        if self._n_params == 2:
                            args = [
                                p1,
                                p2,
                                FunctionCall([g1], "to_array_1d"),
                                FunctionCall([g2], "to_array_1d"),
                                self._integral_grid_2d[i, k],
                            ]
                            method = "interp2dlog"

                        elif self._fit_index:
                            args = [
                                self._src_index_grid,
                                self._integral_grid[i, k],
                                self._src_index[k],
                            ]
                            method = "interpolate_log_y"
                        elif self._fit_beta:
                            args = [
                                self._beta_index_grid,
                                self._integral_grid[i, k],
                                self._beta_index[k],
                            ]
                            method = "interpolate_log_y"
                        elif self._fit_Enorm:
                            args = [
                                self._E0_src_grid,
                                self._integral_grid[i, k],
                                self._E0_src[k],
                            ]
                            method = "interpolate_log_y"
                        elif self._fit_eta:
                            args = [
                                self._eta_grid,
                                self._integral_grid[i, k],
                                self._eta[k],
                            ]
                            method = "interpolate_log_y"

                        (
                            self._eps[i, k]
                            << FunctionCall(
                                args,
                                method,
                            )
                            * self._T[i]
                        )

                        if not self._fit_nex:
                            StringExpression(
                                [
                                    self._Nex_src_comp[i],
                                    "+=",
                                    self._F[k] * self._eps[i, k],
                                ]
                            )
                    if self._fit_nex:
                        StringExpression(
                            [
                                self._F[k],
                                " = ",
                                self._Nex_per_ps[k],
                                " / sum(eps[:, k])",
                            ]
                        )
                        # self._Nex_per_ps[k] / FunctionCall([self._eps[:, k]], "sum")
                    if self._fit_nex or self._seyfert:
                        self._L[k] << self._F[k] / self._flux_conv_val[
                            k
                        ] * StringExpression(
                            [
                                "(4 * pi() * pow(",
                                self._D[k],
                                " * ",
                                3.086e22,
                                ", 2))",
                            ]
                        )
                    if self._seyfert and self._fit_nex:
                        # Nex = P * F_tab(eta, divided by accomp. P) * eps, so F = P * F_tab(eta),
                        # Nex = F * eps
                        # F = F(eta) * P = Nex / eps
                        # P = F / F(eta)
                        if len(self.sources.point_source) == 1:
                            self._P[k] << self._F[k] / self._src_flux_table[0](
                                self._eta[k]
                            )
                        else:
                            for j in range(1, len(self.sources.point_source) + 1):
                                if j == 1:
                                    context = IfBlockContext
                                else:
                                    context = ElseIfBlockContext
                                with context([k, " == ", j]):
                                    self._P[k] << self._F[k] / self._src_flux_table[
                                        j - 1
                                    ](self._eta[k])

                    with ForLoopContext(1, self._Net_stan, "i") as i:
                        StringExpression(
                            [
                                self._Nex_src_comp[i],
                                "+=",
                                self._F[k] * self._eps[i, k],
                            ]
                        )
                    if not self._fit_nex:
                        StringExpression(
                            [
                                self._Nex_per_ps[k],
                                "+=",
                                self._F[k],
                                " * ",
                                "sum(eps[:, k])",
                            ]
                        )
                    # StringExpression([self._F_src, " += ", self._F[k]])

            if self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_diff

            if self.sources.atmospheric and not self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_atmo

            if self.sources.atmospheric and self.sources.diffuse:
                StringExpression("F[Ns+2]") << self._F_atmo

            if self.sources.diffuse and self.sources.atmospheric:
                with ForLoopContext(1, self._Net_stan, "i") as i:
                    (
                        self._eps[i, "Ns+1"]
                        << FunctionCall(
                            [
                                self._diff_index_grid,
                                self._integral_grid[i, self._Ns_string_int_grid],
                                self._diff_index,
                            ],
                            "interpolate_log_y",
                        )
                        * self._T[i]
                    )

                    (
                        self._Nex_diff_comp[i]
                        << self._F["Ns + 1"] * self._eps[i, "Ns + 1"]
                    )

                    # no interpolation needed for atmo as spectral shape is fixed
                    self._eps[i, "Ns + 2"] << self._atmo_integ_val[i] * self._T[i]

                    (
                        self._Nex_atmo_comp[i]
                        << self._F["Ns + 2"] * self._eps[i, "Ns + 2"]
                    )

            elif self.sources.diffuse:
                with ForLoopContext(1, self._Net_stan, "i") as i:
                    (
                        self._eps[i, "Ns + 1"]
                        << FunctionCall(
                            [
                                self._diff_index_grid,
                                self._integral_grid[i, self._Ns_string_int_grid],
                                self._diff_index,
                            ],
                            "interpolate_log_y",
                        )
                        * self._T[i]
                    )

                    (
                        self._Nex_diff_comp[i]
                        << self._F["Ns + 1"] * self._eps[i, "Ns + 1"]
                    )

            elif self.sources.atmospheric:
                with ForLoopContext(1, self._Net_stan, "i") as i:
                    self._eps[i, "Ns + 1"] << self._atmo_integ_val[i] * self._T[i]

                    (
                        self._Nex_atmo_comp[i]
                        << self._F["Ns + 1"] * self._eps[i, "Ns + 1"]
                    )

            with ForLoopContext(1, self._Net_stan, "i") as i:
                self._Nex_comp[i] << FunctionCall([self._F, self._eps[i]], "get_Nex")

            if self.sources.point_source:
                self._Nex_src << FunctionCall([self._Nex_per_ps], "sum")
            if self.sources.diffuse:
                self._Nex_diff << FunctionCall([self._Nex_diff_comp], "sum")
            if self.sources.atmospheric:
                self._Nex_atmo << FunctionCall([self._Nex_atmo_comp], "sum")

            if self._bg:
                self._Nex << FunctionCall([self._Nex_comp], "sum") + self._Nex_bg
            else:
                self._Nex << FunctionCall([self._Nex_comp], "sum")

            """
            # Evaluate the different fractional associations as derived parameters
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
            """
            if self.sources.diffuse and self.sources.atmospheric:
                self._k_diff = "Ns + 1"
                self._k_atmo = "Ns + 2"

            elif self.sources.diffuse:
                self._k_diff = "Ns + 1"

            elif self.sources.atmospheric:
                self._k_atmo = "Ns + 1"

            elif self.sources.background:
                self._k_bg = "Ns + 1"

            # Evaluate logF, packup global parameters
            self._logF << StringExpression(["log(", self._F, ")"])

            if self._nshards not in [0, 1]:
                with DummyContext():
                    start = InstantVariableDef("start", "int", [1])
                    end = InstantVariableDef("end", "int", [0])

                    if self._fit_index:
                        end << end + self._Ns
                        self._global_pars[start:end] << self._src_index
                        start << start + self._Ns
                    if self._fit_beta:
                        end << end + self._Ns
                        self._global_pars[start:end] << self._beta_index
                        start << start + self._Ns
                    if self._fit_Enorm:
                        end << end + self._Ns
                        self._global_pars[start:end] << self._E0_src
                        start << start + self._Ns
                    if self._fit_eta:
                        end << end + self._Ns
                        self._global_pars[start:end] << self._eta
                        start << start + self._Ns
                    if self.sources.diffuse:
                        end << end + 1
                        self._global_pars[start] << self._diff_index
                        start << start + self._Ns
                    if self._fit_ang_sys:
                        end << end + 1
                        self._global_pars[start] << self._ang_sys_add_squared
                        start << start + 1
                    end << end + StringExpression(["size(logF)"])
                    self._global_pars[start:end] << self._logF
                    if self._bg:
                        start << start + 1
                        self._global_pars[start] << self._log_N_bg
                    # Likelihood is evaluated in `lp_reduce`

            else:
                self._model_likelihood()

    def _model(self):
        """
        To write the model section of the Stan file.
        """

        with ModelContext():
            # Likelihood: e^(-Nex) \prod_(i=1)^N_events \sum_(k=1)^N_sources lp[i][k]

            if self._nshards not in [0, 1]:
                # Map data to lp_reduce
                StringExpression(
                    [
                        "target += sum(map_rect(lp_reduce, global_pars, local_pars[:N_shards_loop], real_data[:N_shards_loop], int_data[:N_shards_loop]))"
                    ]
                )

            else:
                # Likelihood
                with ForLoopContext(1, self._N, "i") as i:
                    StringExpression(["target += log_sum_exp(", self._lp[i], ")"])

            StringExpression(["target += -", self._Nex])

            # Priors
            if self.sources.point_source:

                if self._fit_nex and not self._priors.Nex_src.name == "notaprior":
                    StringExpression(
                        [
                            self._Nex_src,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._stan_prior_nex_mu,
                                    self._stan_prior_nex_sigma,
                                ],
                                self._priors.Nex_src.name
                            )
                        ]
                    )    
                elif self._priors.pressure_ratio.name == "notaprior":
                    pass
                elif self._shared_luminosity and self._seyfert:
                    # use global prior for pressure ratio
                    StringExpression(
                        [
                            self._P_glob,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._stan_prior_pressure_ratio_mu,
                                    self._stan_prior_pressure_ratio_sigma,
                                ],
                                self._priors.pressure_ratio.name,
                            ),
                        ]
                    )
                elif self._seyfert and isinstance(
                    self._priors.pressure_ratio, MultiSourcePrior
                ):
                    with ForLoopContext(1, self._Ns, "i") as i:
                        StringExpression(
                            [
                                self._P[i],
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_pressure_ratio_mu[i],
                                        self._stan_prior_pressure_ratio_sigma[i],
                                    ],
                                    self._priors.pressure_ratio.name,
                                ),
                            ]
                        )

                else:
                    with ForLoopContext(1, self._Ns, "i") as i:
                        StringExpression(
                            [
                                self._P[i],
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_pressure_ratio_mu,
                                        self._stan_prior_pressure_ratio_sigma,
                                    ],
                                    self._priors.pressure_ratio.name,
                                ),
                            ]
                        )
                if (
                    self._priors.luminosity.name == "notaprior"
                    or self._seyfert
                    or self._fit_nex
                ):
                    pass

                elif self._priors.luminosity.name in ["normal", "lognormal"]:
                    if self._shared_luminosity and not self._fit_nex:
                        StringExpression(
                            [
                                self._L_glob,
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_lumi_mu,
                                        self._stan_prior_lumi_sigma,
                                    ],
                                    self._priors.luminosity.name,
                                ),
                            ]
                        )
                    elif isinstance(self._priors.luminosity, MultiSourcePrior):
                        with ForLoopContext(1, self._Ns, "i") as i:
                            StringExpression(
                                [
                                    self._L[i],
                                    " ~ ",
                                    FunctionCall(
                                        [
                                            self._stan_prior_lumi_mu[i],
                                            self._stan_prior_lumi_sigma[i],
                                        ],
                                        self._priors.luminosity.name,
                                    ),
                                ]
                            )
                    elif not self._fit_nex:
                        with ForLoopContext(1, self._Ns, "i") as i:
                            StringExpression(
                                [
                                    self._L[i],
                                    " ~ ",
                                    FunctionCall(
                                        [
                                            self._stan_prior_lumi_mu,
                                            self._stan_prior_lumi_sigma,
                                        ],
                                        self._priors.luminosity.name,
                                    ),
                                ]
                            )

                elif self._priors.luminosity.name == "pareto":
                    if isinstance(self._priors.luminosity, MultiSourcePrior):
                        raise ValueError("This is not intended")

                    if self._shared_luminosity and not self._fit_nex:
                        StringExpression(
                            [
                                self._L_glob,
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_lumi_xmin,
                                        self._stan_prior_lumi_alpha,
                                    ],
                                    self._priors.luminosity.name,
                                ),
                            ]
                        )
                    elif not self._fit_nex:
                        with ForLoopContext(1, self._Ns, "i") as i:
                            StringExpression(
                                [
                                    self._L[i],
                                    " ~ ",
                                    FunctionCall(
                                        [
                                            self._stan_prior_lumi_xmin,
                                            self._stan_prior_lumi_alpha,
                                        ],
                                        self._priors.luminosity.name,
                                    ),
                                ]
                            )
                else:
                    raise NotImplementedError(
                        "Luminosity prior distribution not recognised."
                    )

                # TODO add pressure ratio prior here

                if self._priors.src_index.name not in [
                    "normal",
                    "lognormal",
                    "notaprior",
                ]:
                    raise ValueError("Prior type for source index not recognised.")
                if self._priors.src_index.name == "notaprior":
                    pass
                elif (
                    isinstance(self._priors.src_index, MultiSourcePrior)
                    and self._fit_index
                ):
                    with ForLoopContext(1, self._Ns, "i") as i:
                        StringExpression(
                            [
                                self._src_index[i],
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_src_index_mu[i],
                                        self._stan_prior_src_index_sigma[i],
                                    ],
                                    self._priors.src_index.name,
                                ),
                            ]
                        )
                elif self._fit_index and self._shared_src_index:
                    StringExpression(
                        [
                            self._src_index_glob,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._stan_prior_src_index_mu,
                                    self._stan_prior_src_index_sigma,
                                ],
                                self._priors.src_index.name,
                            ),
                        ]
                    )
                elif self._fit_index:
                    with ForLoopContext(1, self._Ns, "i") as i:
                        StringExpression(
                            [
                                self._src_index[i],
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_src_index_mu,
                                        self._stan_prior_src_index_sigma,
                                    ],
                                    self._priors.src_index.name,
                                ),
                            ]
                        )

                if (
                    isinstance(self._priors.beta_index, MultiSourcePrior)
                    and self._fit_beta
                ):
                    with ForLoopContext(1, self._Ns, "i") as i:
                        StringExpression(
                            [
                                self._beta_index[i],
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_beta_index_mu[i],
                                        self._stan_prior_beta_index_sigma[i],
                                    ],
                                    self._priors.beta_index.name,
                                ),
                            ]
                        )
                elif self._shared_src_index and self._fit_beta:
                    StringExpression(
                        [
                            self._beta_index_glob,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._stan_prior_beta_index_mu,
                                    self._stan_prior_beta_index_sigma,
                                ],
                                self._priors.beta_index.name,
                            ),
                        ]
                    )
                elif self._fit_beta:
                    with ForLoopContext(1, self._Ns, "i") as i:
                        StringExpression(
                            [
                                self._beta_index[i],
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_beta_index_mu,
                                        self._stan_prior_beta_index_sigma,
                                    ],
                                    self._priors.beta_index.name,
                                ),
                            ]
                        )

                if (
                    isinstance(self._priors.E0_src, MultiSourcePrior)
                    and self._fit_Enorm
                ):
                    with ForLoopContext(1, self._Ns, "i") as i:
                        StringExpression(
                            [
                                self._E0_src_glob[i],
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_E0_src_mu[i],
                                        self._stan_prior_E0_src_sigma[i],
                                    ],
                                    self._priors.E0_src.name,
                                ),
                            ]
                        )
                elif self._fit_Enorm and self._shared_src_index:
                    StringExpression(
                        [
                            self._E0_src_glob,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._stan_prior_E0_src_mu,
                                    self._stan_prior_E0_src_sigma,
                                ],
                                self._priors.E0_src.name,
                            ),
                        ]
                    )
                elif self._fit_Enorm:
                    with ForLoopContext(1, self._Ns, "i") as i:
                        StringExpression(
                            [
                                self._E0_src_glob[i],
                                " ~ ",
                                FunctionCall(
                                    [
                                        self._stan_prior_E0_src_mu,
                                        self._stan_prior_E0_src_sigma,
                                    ],
                                    self._priors.E0_src.name,
                                ),
                            ]
                        )

            if self._priors.eta.name == "notaprior":
                pass
            elif self._priors.eta.name not in ["normal", "lognormal"]:
                raise ValueError("Prior type not recognised for eta")
            elif self._fit_eta and isinstance(self._priors.eta, MultiSourcePrior):
                with ForLoopContext(1, self._Ns, "i") as i:
                    StringExpression(
                        [
                            self._eta[i],
                            " ~ ",
                            FunctionCall(
                                [self._stan_prior_eta_mu, self._stan_prior_eta_sigma],
                                self._priors.eta.name,
                            ),
                        ]
                    )
            elif self._fit_eta:
                StringExpression(
                    [
                        self._eta_glob,
                        " ~ ",
                        FunctionCall(
                            [self._stan_prior_eta_mu, self._stan_prior_eta_sigma],
                            self._priors.eta.name,
                        ),
                    ]
                )

            if self.sources.diffuse:
                if self._priors.diffuse_flux.name not in ["normal", "lognormal"]:
                    raise NotImplementedError(
                        "Prior type for diffuse flux not recognised."
                    )
                StringExpression(
                    [
                        self._diffuse_norm,
                        " ~ ",
                        FunctionCall(
                            [
                                self._stan_prior_f_diff_mu,
                                self._stan_prior_f_diff_sigma,
                            ],
                            self._priors.diffuse_flux.name,
                        ),
                    ]
                )

                if self._priors.diff_index.name not in ["normal", "lognormal"]:
                    raise NotImplementedError(
                        "Prior type for diffuse index not recognised."
                    )
                StringExpression(
                    [
                        self._diff_index,
                        " ~ ",
                        FunctionCall(
                            [
                                self._stan_prior_diff_index_mu,
                                self._stan_prior_diff_index_sigma,
                            ],
                            self._priors.diff_index.name,
                        ),
                    ]
                )
            if self._priors.atmospheric_flux.name not in ["normal", "lognormal"]:
                raise NotImplementedError(
                    "Prior type for atmospheric flux not recognised."
                )
            if self.sources.atmospheric:
                StringExpression(
                    [
                        self._F_atmo,
                        " ~ ",
                        FunctionCall(
                            [
                                self._stan_prior_f_atmo_mu,
                                self._stan_prior_f_atmo_sigma,
                            ],
                            self._priors.atmospheric_flux.name,
                        ),
                    ]
                )
            if self._fit_ang_sys:
                if self._priors.ang_sys.name == "normal":
                    StringExpression(
                        [
                            self._ang_sys_add,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._ang_sys_mu,
                                    self._ang_sys_sigma,
                                ],
                                self._priors.ang_sys.name,
                            ),
                        ]
                    )
                elif self._priors.ang_sys.name == "exponnorm":
                    StringExpression(
                        [
                            self._ang_sys_add,
                            " ~ ",
                            FunctionCall(
                                [
                                    self._ang_sys_mu,
                                    self._ang_sys_sigma,
                                    self._ang_sys_lam,
                                ],
                                "exp_mod_normal",
                            ),
                        ]
                    )
                else:
                    raise NotImplementedError(
                        "Prior type for angular systematics not recognised."
                    )

    def _generated_quantities(self):
        """
        To write the generated quantities section of the Stan file.
        """

        with GeneratedQuantitiesContext():
            # Remove _log_lik? was only used for some niche testing
            # self._log_lik = ForwardArrayDef("log_lik", "real", ["[N]"])
            if self._pgamma:
                self._E_peak = ForwardArrayDef("E_peak", "real", ["[Ns]"])
                self._peak_flux = ForwardArrayDef("peak_energy_flux", "real", ["[Ns]"])
                with ForLoopContext(1, self._Ns, "k") as k:
                    # Define peak energy of E^2 dN/dE
                    # find via derivative of above expression, set to zero, ask wolfram alpha
                    self._E_peak[k] << self._E0_src[k] * FunctionCall(
                        ["exp(1)", f"1 / {self._sources.point_source_spectrum._beta}"],
                        "pow",
                    )
                    # Calculate peak E^2 dN/dE flux
                    self._peak_flux[k] << self._F[k] * FunctionCall(
                        [
                            self._src_spectrum_lpdf(
                                self._E_peak[k],
                                StringExpression(["{", self._E0_src[k], "}"]),
                                StringExpression(
                                    [
                                        "{",
                                        self._Emin_src[k],
                                        ",",
                                        self._Emax_src[k],
                                        "}",
                                    ]
                                ),
                                StringExpression(["{", 0, "}"]),
                            )
                        ],
                        "exp",
                    ) * FunctionCall([self._E_peak[k], 2], "pow")
            # Calculation of individual source-event logprobs
            # Only when parallel mode on
            if self._nshards not in [0, 1]:
                # Define variables to store loglikelihood
                # and latent energies
                if self._use_event_tag:
                    self._lp = ForwardArrayDef("lp", "vector[Ns_tot-Ns+1]", self._N_str)
                else:
                    self._lp = ForwardArrayDef("lp", "vector[Ns_tot]", self._N_str)
                if self._use_event_tag and not (
                    self.sources.diffuse or self.sources.atmospheric
                ):
                    self._irf_return = ForwardVariableDef(
                        "irf_return",
                        "tuple(real, real)",
                    )
                elif self._use_event_tag:
                    self._irf_return = ForwardVariableDef(
                        "irf_return",
                        "tuple(real, real, real, real)",
                    )
                elif self.sources.diffuse or self.sources.atmospheric:
                    self._irf_return = ForwardVariableDef(
                        "irf_return",
                        "tuple(array[Ns] real, array[Ns] real, real, real)",
                    )
                else:
                    self._irf_return = ForwardVariableDef(
                        "irf_return",
                        "tuple(array[Ns] real, array[Ns] real)",
                    )
                if not self._use_event_tag:
                    self._eres_src = ForwardArrayDef("eres_src", "real", ["[Ns]"])
                    self._aeff_src = ForwardArrayDef("aeff_src", "real", ["[Ns]"])
                else:
                    self._eres_src = ForwardVariableDef("eres_src", "real")
                    self._aeff_src = ForwardVariableDef("aeff_src", "real")

                if self.sources.diffuse or self.sources.atmospheric:
                    self._eres_diff = ForwardVariableDef("eres_diff", "real")
                    self._aeff_diff = ForwardVariableDef("aeff_diff", "real")

                if self._fit_ang_sys:
                    self._spatial_loglike = ForwardArrayDef(
                        "spatial_loglike", "real", ["[Ns, N]"]
                    )

                self._model_likelihood()

                if self._fit_ang_sys:
                    self._ang_sys_deg = ForwardVariableDef("ang_sys_deg", "real")
                    self._ang_sys_deg << self._ang_sys_add * StringExpression(
                        ["180 / pi()"]
                    )

                if self._debug:
                    self._lp_gen_q = ForwardVariableDef("lp_gen_q", "vector[N]")
                    self._lp_debug = ForwardVariableDef("lp_debug", "vector[N]")
                    self._lp_debug << StringExpression(
                        [
                            "map_rect(lp_reduce, global_pars, local_pars[:N_shards_loop], real_data[:N_shards_loop], int_data[:N_shards_loop])"
                        ]
                    )
                    with ForLoopContext(1, self._N, "i") as i:
                        self._lp_gen_q[i] << FunctionCall([self._lp[i]], "log_sum_exp")
