import numpy as np
from astropy import units as u
from typing import List
from collections import OrderedDict
from hierarchical_nu.detector.detector_model import DetectorModel
from hierarchical_nu.detector.r2021 import R2021DetectorModel

from hierarchical_nu.priors import Priors
from hierarchical_nu.stan.interface import StanInterface

from hierarchical_nu.backend.stan_generator import (
    FunctionsContext,
    Include,
    DataContext,
    TransformedDataContext,
    ParametersContext,
    TransformedParametersContext,
    GeneratedQuantitiesContext,
    ForLoopContext,
    IfBlockContext,
    ElseBlockContext,
    ModelContext,
    FunctionCall,
    UserDefinedFunction,
)

from hierarchical_nu.backend.expression import (
    ReturnStatement,
    StringExpression,
)

from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ParameterDef,
    ParameterVectorDef,
    InstantVariableDef,
)

from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.events import TRACKS, CASCADES
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources


class StanFitInterface(StanInterface):
    """
    An interface for generating the Stan fit code.
    """

    def __init__(
        self,
        output_file: str,
        sources: Sources,
        detector_model_type: DetectorModel,
        atmo_flux_energy_points: int = 100,
        atmo_flux_theta_points: int = 30,
        includes: List[str] = ["interpolation.stan", "utils.stan", "vMF.stan"],
        priors: Priors = Priors(),
        nshards: int = 1,
    ):
        """
        An interface for generating Stan fit code.

        :param output_file: Name of the file to write to
        :param sources: Sources object containing sources to be fit
        :param detector_model_type: Type of the detector model to be used
        :param atmo_flux_theta_points: Number of points to use for the grid of
        atmospheric flux
        :param includes: List of names of stan files to include into the
        functions block of the generated file
        :param priors: Priors object detailing the priors to use
        :param nshards: Number of shards for multithreading, defaults to zero
        """

        if (
            detector_model_type == R2021DetectorModel
            and "r2021_pdf.stan" not in includes
        ):
            includes.append("r2021_pdf.stan")
            R2021DetectorModel.generate_code(
                DistributionMode.PDF, rewrite=False, gen_type="lognorm"
            )

        super().__init__(
            output_file=output_file,
            sources=sources,
            detector_model_type=detector_model_type,
            includes=includes,
        )

        self._priors = priors

        self._atmo_flux_energy_points = atmo_flux_energy_points

        self._atmo_flux_theta_points = atmo_flux_theta_points

        self._get_par_ranges()

        assert isinstance(nshards, int)
        assert nshards >= 0
        self._nshards = nshards

    def _get_par_ranges(self):
        """
        Extract the parameter ranges to use in Stan from the
        defined parameters.
        """

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
        """
        Write the functions section of the Stan file.
        """

        with FunctionsContext():

            # Include all the specified files
            for include_file in self._includes:
                _ = Include(include_file)

            self._dm = OrderedDict()

            for event_type in self._event_types:

                # Include the PDF mode of the detector model
                self._dm[event_type] = self._detector_model_type(
                    event_type=event_type,
                    mode=DistributionMode.PDF,
                )

            # If we have point sources, include the shape of their PDF
            # and how to convert from energy to number flux
            if self.sources.point_source:

                self._src_spectrum_lpdf = self._ps_spectrum.make_stan_lpdf_func(
                    "src_spectrum_logpdf"
                )

                self._flux_conv = self._ps_spectrum.make_stan_flux_conv_func(
                    "flux_conv"
                )

            # If we have diffuse sources, include the shape of their PDF
            if self.sources.diffuse:

                self._diff_spectrum_lpdf = self._diff_spectrum.make_stan_lpdf_func(
                    "diff_spectrum_logpdf"
                )

            # If we have atmospheric sources, include the atmospheric flux table
            # the density of the grid in theta (ie. declination) is specified here
            if self.sources.atmospheric:

                # Increasing theta points too much makes compilation very slow
                # Could switch to passing array as data if problematic
                self._atmo_flux_func = self._atmo_flux.make_stan_function(
                    energy_points=self._atmo_flux_energy_points,
                    theta_points=self._atmo_flux_theta_points,
                )

                # Include integral for normalisation to PDF
                self._atmo_flux_integral = self._atmo_flux.total_flux_int.to(
                    1 / (u.m**2 * u.s)
                ).value

            if self._nshards not in [0, 1]:
                # Create a function to be used in map_rect in the model block
                # Signature is determined by stan's `map_rect` function
                lp_reduce = UserDefinedFunction(
                    "lp_reduce",
                    ["global", "local", "real_data", "int_data"],
                    ["vector", "vector", "array[] real", "array[] int"],
                    "vector",
                )

                with lp_reduce:
                    #TODO: make obs time available
                    # Unpack integer data, needed to interpret real data
                    # Use InstantVariableDef to save on lines
                    N = InstantVariableDef("N", "int", ["int_data[1]"])
                    Ns = InstantVariableDef("Ns", "int", ["int_data[2]"])
                    diffuse = InstantVariableDef("diffuse", "int", ["int_data[3]"])
                    atmo = InstantVariableDef("atmo", "int", ["int_data[4]"])
                    Ns_tot = InstantVariableDef("Ns_tot", "int", ["Ns+atmo+diffuse"])
                    Ngrid = InstantVariableDef("Ngrid", "int", ["int_data[5]"])
                    start = ForwardVariableDef("start", "int")
                    end = ForwardVariableDef("end", "int")
                    len = ForwardVariableDef("len", "int")
                    src_index_grid = ForwardVariableDef("src_index_grid", "vector[Ngrid]")
                    if self.sources.atmospheric and "tracks" in self._event_types:
                        atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")
                    T = ForwardVariableDef("T", "real")
                    if self._sources.diffuse:
                        diff_index_grid = ForwardVariableDef("diff_index_grid", "vector[Ngrid]")

                    if "tracks" in self._event_types:
                        integral_grid_t = ForwardArrayDef(
                            "integral_grid_t", "vector[Ngrid]", ["[Ns+diffuse]"]
                        )
                        eps_t = ForwardVariableDef("eps_t", "vector[Ns_tot]")
                    if "cascades" in self._event_types:
                        integral_grid_c = ForwardArrayDef(
                            "integral_grid_c", "vector[Ngrid]", ["[Ns+diffuse]"]
                        )
                        eps_c = ForwardVariableDef("eps_c", "vector[Ns_tot]")

                    # Get global parameters
                    # Check for shared index
                    if self._shared_src_index:
                        src_index = ForwardVariableDef("src_index", "real")
                        src_index << StringExpression(["global[1]"])
                        idx = 2
                    else:
                        src_index = ForwardVariableDef("src_index", "vector[Ns]")
                        idx = len(self.sources._point_sources) + 1

                    # Get diffuse index
                    if self.sources.diffuse:
                        diff_index = ForwardVariableDef("diff_index", "real")
                        diff_index << StringExpression(["global[", idx, "]"])
                        idx += 1

                    logF = ForwardVariableDef("logF", "vector[Ns_tot]")
                    logF << StringExpression(["global[", idx, ":]"])

                    # Local pars are only source energies
                    Esrc = ForwardVariableDef("Esrc", "vector[size(local)]")
                    Esrc << StringExpression(["local"])
                    E = ForwardVariableDef("E", "vector[N]")

                    # Define indices for unpacking of real_data
                    start << 1
                    len << N
                    end << N

                    # Define variable to store loglikelihood
                    lp = ForwardArrayDef("lp", "vector[Ns_tot]", ["[N]"])

                    # Unpack event types (track or cascade)
                    event_type = ForwardArrayDef("event_type", "int", ["[N]"])
                    event_type << StringExpression(["int_data[6:5+N]"])

                    Edet = ForwardVariableDef("Edet", "vector[N]")
                    Edet << FunctionCall(["real_data[start:end]"], "to_vector")
                    # Shift indices appropriate amount for next batch of data
                    start << start + len
                    end << end + len
                    kappa = ForwardVariableDef("kappa", "vector[N]")
                    kappa << StringExpression(["to_vector(real_data[start:end])"])

                    omega_det = ForwardArrayDef("omega_det", "vector[3]", ["[N]"])
                    start << start + len
                    varpi = ForwardArrayDef("varpi", "vector[3]", ["[Ns]"])
                    # Loop over events to unpack reconstructed direction
                    with ForLoopContext(1, N, "i") as i:
                        end << end + 3
                        omega_det[i] << StringExpression(
                            ["to_vector(real_data[start:end])"]
                        )
                        start << start + 3
                    # Loop over sources to unpack source direction (for point sources only)
                    with ForLoopContext(1, Ns, "i") as i:
                        end << end + 3
                        varpi[i] << StringExpression(
                            ["to_vector(real_data[start:end])"]
                        )
                        start << start + 3
                    # If diffuse source, z is longer by 1 element
                    if self.sources.diffuse:
                        end << end + Ns + 1
                        z = ForwardVariableDef("z", "vector[Ns+diffuse]")
                        z << StringExpression(["to_vector(real_data[start:end])"])
                        start << start + Ns + 1
                    else:
                        end << end + Ns
                        z = ForwardVariableDef("z", "vector[Ns]")
                        z << StringExpression(["to_vector(real_data[start:end])"])
                        start << start + Ns

                    Esrc_min = ForwardVariableDef("Esrc_min", "real")
                    Esrc_max = ForwardVariableDef("Esrc_max", "real")

                    end << end + 1
                    Esrc_min << StringExpression(["real_data[start]"])
                    start << start + 1

                    end << end + 1
                    Esrc_max << StringExpression(["real_data[start]"])
                    start << start + 1

                    end << end + 1
                    T << StringExpression(["real_data[start]"])
                    start << start + 1

                    if self.sources.atmospheric and "tracks" in self._event_types:
                        end << end + 1
                        atmo_integ_val << StringExpression(["real_data[start]"])
                        start << start + 1

                    end << end + Ngrid
                    src_index_grid << StringExpression(["to_vector(real_data[start:end])"])
                    start << start + Ngrid

                    if self._sources.diffuse:
                        end << end + Ngrid
                        diff_index_grid << StringExpression(["to_vector(real_data[start:end])"])
                        start << start + Ngrid

                    if "tracks" in self._event_types:
                        with ForLoopContext(1, "Ns+diffuse", "k") as k:
                            end << end + Ngrid
                            integral_grid_t[k] << StringExpression(["to_vector(real_data[start:end])"])
                            start << start + Ngrid

                    if "cascades" in self._event_types:
                        with ForLoopContext(1, "Ns+diffuse", "k") as k:
                            end << end + Ngrid
                            integral_grid_c[k] << StringExpression(["to_vector(real_data[start:end])"])
                            start << start + Ngrid

                    # Define tracks and cascades to sort events into correct detector response
                    if "tracks" in self._event_types:
                        track_type = ForwardVariableDef("track_type", "int")
                        track_type << TRACKS

                    if "cascades" in self._event_types:
                        cascade_type = ForwardVariableDef("cascade_type", "int")
                        cascade_type << CASCADES

                    if self.sources.diffuse and self.sources.atmospheric:
                        k_diff = "Ns + 1"
                        k_atmo = "Ns + 2"

                    elif self.sources.diffuse:
                        k_diff = "Ns + 1"

                    elif self.sources.atmospheric:
                        k_atmo = "Ns + 1"


                    
                    # Actual function body goes here
                    # Starting here, everything needs to go to lp_reduce!
                    with ForLoopContext(1, N, "i") as i:

                        # Tracks
                        if "tracks" in self._event_types:
                            with IfBlockContext(
                                [StringExpression([event_type[i], " == ", track_type])]
                            ):
                                # Sum over sources => evaluate and store components
                                with ForLoopContext(1, "Ns+atmo+diffuse", "k") as k:
                                    # Point source components
                                    if self.sources.point_source:
                                        with IfBlockContext(
                                            [StringExpression([k, " < ", Ns + 1])]
                                        ):
                                            if self._shared_src_index:
                                                src_index_ref = src_index
                                            else:
                                                src_index_ref = src_index[k]

                                            eps_t[k] << FunctionCall(
                                                [
                                                    src_index_grid,
                                                    integral_grid_t[k],
                                                    src_index_ref,
                                                ],
                                                "interpolate",
                                            ) * T

                                            # log_prob += log(p(Esrc|src_index))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    self._src_spectrum_lpdf(
                                                        Esrc[i],
                                                        src_index_ref,
                                                        Esrc_min,
                                                        Esrc_max,
                                                    ),
                                                ]
                                            )

                                            # E = Esrc / (1+z)
                                            E[i] << StringExpression(
                                                [Esrc[i], " / (", 1 + z[k], ")"]
                                            )

                                            # log_prob += log(p(omega_det|varpi, kappa))
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

                                    # Diffuse component
                                    if self.sources.diffuse:

                                        with IfBlockContext(
                                            [StringExpression([k, " == ", k_diff])]
                                        ):
                                            
                                            eps_t[k] << FunctionCall(
                                                [
                                                    diff_index_grid,
                                                    integral_grid_t[k],
                                                    diff_index,
                                                ],
                                                "interpolate",
                                            ) * T

                                            # log_prob += log(p(Esrc|diff_index))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    self._diff_spectrum_lpdf(
                                                        Esrc[i],
                                                        diff_index,
                                                        Esrc_min,
                                                        Esrc_max,
                                                    ),
                                                ]
                                            )

                                            # E = Esrc / (1+z)
                                            E[i] << StringExpression(
                                                [Esrc[i], " / (", 1 + z[k], ")"]
                                            )

                                            # log_prob += log(1/4pi)
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    np.log(1 / (4 * np.pi)),
                                                ]
                                            )

                                    # Atmospheric component
                                    if self.sources.atmospheric:

                                        with IfBlockContext(
                                            [StringExpression([k, " == ", k_atmo])]
                                        ):
                                            
                                            eps_t[k] << atmo_integ_val * T

                                            # log_prob += log(p(Esrc, omega | atmospheric source))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    FunctionCall(
                                                        [
                                                            self._atmo_flux_func(
                                                                Esrc[i],
                                                                omega_det[i],
                                                            )
                                                            / self._atmo_flux_integral
                                                        ],
                                                        "log",
                                                    ),
                                                ]
                                            )

                                            # E = Esrc
                                            E[i] << Esrc[i]

                                    # Detection effects
                                    if self.detector_model_type == R2021DetectorModel:

                                        StringExpression(
                                            [
                                                lp[i][k],
                                                " += ",
                                                self._dm["tracks"].energy_resolution(
                                                    FunctionCall([E[i]], "log10"),
                                                    FunctionCall([Edet[i]], "log10"),
                                                    omega_det[i],
                                                ),
                                            ]
                                        )

                                    else:

                                        StringExpression(
                                            [
                                                lp[i][k],
                                                " += ",
                                                self._dm["tracks"].energy_resolution(
                                                    E[i], Edet[i]
                                                ),
                                            ]
                                        )

                                    StringExpression(
                                        [
                                            lp[i][k],
                                            " += log(",
                                            self._dm["tracks"].effective_area(
                                                E[i], omega_det[i]
                                            ),
                                            ")",
                                        ]
                                    )
                                StringExpression(
                                    [
                                        lp[i],
                                        " += ",
                                        FunctionCall(
                                            [eps_t],
                                            "log"
                                        ),
                                        " + ",
                                        logF
                                    ]
                                )

                        # Cascades
                        # See comments for tracks for more details, approach is the same
                        if "cascades" in self._event_types:

                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [event_type[i], " == ", cascade_type]
                                    )
                                ]
                            ):

                                with ForLoopContext(1, "Ns+atmo+diffuse", "k") as k:

                                    # Point source components
                                    if self.sources.point_source:

                                        with IfBlockContext(
                                            [StringExpression([k, " < ", Ns + 1])]
                                        ):

                                            if self._shared_src_index:
                                                src_index_ref = src_index
                                            else:
                                                src_index_ref = src_index[k]

                                            eps_c[k] << FunctionCall(
                                                [
                                                    src_index_grid,
                                                    integral_grid_c[k],
                                                    src_index_ref,
                                                ],
                                                "interpolate",
                                            ) * T

                                            # log_prob += log(p(Esrc | src_index))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    self._src_spectrum_lpdf(
                                                        Esrc[i],
                                                        src_index_ref,
                                                        Esrc_min,
                                                        Esrc_max,
                                                    ),
                                                ]
                                            )

                                            # E = Esrc / (1+z)
                                            E[i] << StringExpression(
                                                [Esrc[i], " / (", 1 + z[k], ")"]
                                            )

                                            # log_prob += log(p(omega_det | varpi, kappa))
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

                                    # Diffuse component
                                    if self.sources.diffuse:

                                        with IfBlockContext(
                                            [StringExpression([k, " == ", k_diff])]
                                        ):
                                            
                                            eps_c[k] << FunctionCall(
                                                [
                                                    diff_index_grid,
                                                    integral_grid_c[k],
                                                    diff_index,
                                                ],
                                                "interpolate",
                                            ) * T

                                            # log_prob += log(p(Esrc | diff_index))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    self._diff_spectrum_lpdf(
                                                        Esrc[i],
                                                        diff_index,
                                                        Esrc_min,
                                                        Esrc_max,
                                                    ),
                                                ]
                                            )

                                            # E = Esrc / (1+z)
                                            E[i] << StringExpression(
                                                [Esrc[i], " / (", 1 + z[k], ")"]
                                            )

                                            # log_prob += log(1/4pi)
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    np.log(1 / (4 * np.pi)),
                                                ]
                                            )

                                    # Atmospheric component
                                    if self.sources.atmospheric:

                                        with IfBlockContext(
                                            [StringExpression([k, " == ", k_atmo])]
                                        ):

                                            # log_prob += -inf (no atmo comp for cascades!)
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += negative_infinity()",
                                                ]
                                            )

                                            # E = Esrc
                                            E[i] << Esrc[i]

                                            eps_c[k] << atmo_integ_val * T

                                    # Detection effects
                                    # log_prob += log(p(Edet | E))
                                    StringExpression(
                                        [
                                            lp[i][k],
                                            " += ",
                                            self._dm["cascades"].energy_resolution(
                                                E[i], Edet[i]
                                            ),
                                        ]
                                    )
                                StringExpression(
                                    [
                                        lp[i],
                                        " += ",
                                        FunctionCall(
                                            [eps_c],
                                            "log"
                                        ),
                                        " + ",
                                        logF
                                    ]
                                )

                    results = ForwardArrayDef("results", "real", ["[N]"])
                    with ForLoopContext(1, N, "i") as i:
                        results[i] << FunctionCall([lp[i]], "log_sum_exp")
                    ReturnStatement(["[sum(results)]'"])

    def _data(self):

        with DataContext():

            # Total number of detected events
            self._N = ForwardVariableDef("N", "int")
            self._N_str = ["[", self._N, "]"]

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

            # Event types as track/cascades
            self._event_type = ForwardVariableDef("event_type", "vector[N]")

            # Uncertainty on the event's angular reconstruction
            self._kappa = ForwardVariableDef("kappa", "vector[N]")

            # Energy range at source
            self._Esrc_min = ForwardVariableDef("Esrc_min", "real")
            self._Esrc_max = ForwardVariableDef("Esrc_max", "real")
            self._Esrc_norm = ForwardVariableDef("Esrc_norm", "real")

            # Number of point sources
            self._Ns = ForwardVariableDef("Ns", "int")
            self._Ns_str = ["[", self._Ns, "]"]
            self._Ns_1p_str = ["[", self._Ns, "+1]"]
            self._Ns_2p_str = ["[", self._Ns, "+2]"]

            # Total number of sources
            self._Ns_tot = ForwardVariableDef("Ns_tot", "int")

            # True directions and distances of point sources
            self._varpi = ForwardArrayDef("varpi", "unit_vector[3]", self._Ns_str)
            self._D = ForwardVariableDef("D", "vector[Ns]")

            # Density of interpolation grid and energy grid points
            self._Ngrid = ForwardVariableDef("Ngrid", "int")

            # Observation time
            self._T = ForwardVariableDef("T", "real")

            # Redshift
            if self.sources.diffuse:

                N_int_str = self._Ns_1p_str
                self._z = ForwardVariableDef("z", "vector[Ns+1]")

            else:

                N_int_str = self._Ns_str
                self._z = ForwardVariableDef("z", "vector[Ns]")

            # Interpolation grid points in spectral indices for
            # sources with spectral index as a free param, and
            # the exposure integral evaluated at these points, for
            # different event types (i.e. different Aeff)
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

            # Don't need a grid for atmo as spectral shape is fixed, so pass single value.
            if self.sources.atmospheric:

                self._atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")

    def _transformed_data(self):
        """
        To write the transformed data section of the Stan file.
        """

        with TransformedDataContext():

            if "tracks" in self._event_types:

                self._track_type = ForwardVariableDef("track_type", "int")
                self._track_type << TRACKS

            if "cascades" in self._event_types:

                self._cascade_type = ForwardVariableDef("cascade_type", "int")
                self._cascade_type << CASCADES

            # Find out how many cascade and how many track events in data
            if "cascades" in self._event_types and "tracks" in self._event_types:

                self._N_c = ForwardVariableDef("N_c", "int")
                self._N_c << 0
                self._N_t = ForwardVariableDef("N_t", "int")
                self._N_t << 0

                with ForLoopContext(1, self._N, "k") as k:

                    with IfBlockContext(
                        [
                            StringExpression(
                                [self._event_type[k], " == ", self._cascade_type]
                            )
                        ]
                    ):
                        self._N_c << self._N_c + 1

                    with ElseBlockContext():
                        self._N_t << self._N_t + 1

            if self._nshards not in [0, 1]:
                self._N_shards_use_this = ForwardVariableDef("N_shards_loop", "int")
                self._N_shards_use_this << self._N_shards
                # Create the rectangular data blocks for use in `map_rect`
                self._N_mod_J = ForwardVariableDef("N_mod_J", "int")
                self._N_mod_J << self._N % self._J
                self._N_ev_distributed = ForwardVariableDef("N_ev_distributed", "int")
                self._N_ev_distributed << 0
                # Find size for real_data array
                sd_events_J = 5    # reco energy, kappa, reco dir (unit vector)
                sd_varpi_Ns = 3    # coords of events in the sky (unit vector)
                sd_if_atmo_z = 1   # redshift of diffuse component
                sd_z_Ns = 1        # redshift of PS
                sd_other = 3       # Esrc_min, max, obs time
                if self.sources.atmospheric and "tracks" in self._event_types:
                    sd_other += 1    # no atmo in cascades
                sd_string = (
                    f"{sd_events_J}*J + {sd_varpi_Ns}*Ns + {sd_z_Ns}*Ns + {sd_other}"
                )
                if self.sources.diffuse:
                    sd_string += f" + {sd_if_atmo_z}"
                sd_Ngrid = 0      # arrays of length Ngrid
                #how many are needed?
                #grid itself for PS and diff
                #exposure for tracks and cascades for PS and diff
                #              exposure integrals                      index grids
                # (--------------------------------------------)   (-----------------)
                # ((if cascades) + (if tracks))*(Ns + (if diff)) + (if PS) + (if diff)
                if "tracks" in self._event_types:
                    sd_Ngrid += 1
                if "cascades" in self._event_types:
                    sd_Ngrid += 1
                sd_Ngrid_diff = 1 if self._sources.diffuse else 0
                sd_Ngrid_ps = 1 if self._sources._point_source else 0
                sd_string += f" + ({sd_Ngrid}*(Ns + {sd_Ngrid_diff}) + {sd_Ngrid_diff} + {sd_Ngrid_ps})*Ngrid"

                #order:
                # ps index grid, diff index grid, tracks: ps grids, diff grid
                # cascades: ps grids, diff grid

                # Create data arrays
                self.real_data = ForwardArrayDef(
                    "real_data", "real", ["[N_shards,", sd_string, "]"]
                )

                self.int_data = ForwardArrayDef(
                    "int_data", "int", ["[", self._N_shards, ", ", "J+5", "]"]
                )
                #TODO: pack atmo_integ_val into real_data
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
                    insert_end << insert_end + insert_len
                    self.real_data[i, insert_start:insert_end] << FunctionCall(
                        [self._kappa[start:end]], "to_array_1d"
                    )
                    insert_start << insert_start + insert_len

                    with ForLoopContext(start, end, "f") as f:
                        insert_end << insert_end + 3
                        self.real_data[i, insert_start:insert_end] << FunctionCall(
                            [self._omega_det[f]], "to_array_1d"
                        )
                        insert_start << insert_start + 3

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

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Esrc_min
                    insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Esrc_max
                    insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._T
                    insert_start << insert_start + 1

                    if self.sources.atmospheric and "tracks" in self._event_types:
                        insert_end << insert_end + 1
                        self.real_data[i, insert_start] << self._atmo_integ_val
                        insert_start << insert_start + 1
                    
                    """
                    insert_end << insert_end + self._Ngrid
                    self.real_data[i, insert_start:insert_end] << FunctionCall(
                        [self._Eg], "to_array_1d"
                    )
                    insert_start << insert_start + self._Ngrid
                    """
                    # Spectral index grids
                    # used to interpolate over exposure factor
                    # For order of entries see above
                    insert_end << insert_end + self._Ngrid
                    self.real_data[i, insert_start:insert_end] << FunctionCall(
                        [self._src_index_grid], "to_array_1d"
                    )
                    insert_start << insert_start + self._Ngrid

                    if self._sources.diffuse:
                        insert_end << insert_end + self._Ngrid
                        self.real_data[i, insert_start:insert_end] << FunctionCall(
                            [self._diff_index_grid], "to_array_1d"
                        )
                        insert_start << insert_start + self._Ngrid

                    if "tracks" in self._event_types:
                        with ForLoopContext(1, self._Ns, "k") as k:
                            insert_end << insert_end + self._Ngrid
                            self.real_data[i, insert_start:insert_end] << FunctionCall(
                                [self._integral_grid_t[k]], "to_array_1d"
                            )
                            insert_start << insert_start + self._Ngrid

                        if self._sources.diffuse:
                            insert_end << insert_end + self._Ngrid
                            self.real_data[i, insert_start:insert_end] << FunctionCall(
                                [self._integral_grid_t[self._Ns+1]], "to_array_1d"
                            )
                            insert_start << insert_start + self._Ngrid
                    
                    if "cascades" in self._event_types:
                        with ForLoopContext(1, self._Ns, "k") as k:
                            insert_end << insert_end + self._Ngrid
                            self.real_data[i, insert_start:insert_end] << FunctionCall(
                                [self._integral_grid_c[k]], "to_array_1d"
                            )
                            insert_start << insert_start + self._Ngrid

                        if self._sources.diffuse:
                            insert_end << insert_end + self._Ngrid
                            self.real_data[i, insert_start:insert_end] << FunctionCall(
                                [self._integral_grid_c[self._Ns+1]], "to_array_1d"
                            )
                            insert_start << insert_start + self._Ngrid

                    """
                    if (
                        "tracks" in self._event_types
                        and "cascades" in self._event_types
                    ):
                        with ForLoopContext(1, self._Ns_tot, "f") as f:
                            insert_end << insert_end + self._Ngrid
                            self.real_data[i, insert_start:insert_end] << FunctionCall(
                                [self._Pg_t[f]], "to_array_1d"
                            )
                            insert_start << insert_start + self._Ngrid

                        with ForLoopContext(1, self._Ns_tot, "f") as f:
                            insert_end << insert_end + self._Ngrid
                            self.real_data[i, insert_start:insert_end] << FunctionCall(
                                [self._Pg_c[f]], "to_array_1d"
                            )
                            insert_start << insert_start + self._Ngrid

                    elif "tracks" in self._event_types:
                        with ForLoopContext(1, self._Ns_tot, "f") as f:
                            insert_end << insert_end + self._Ngrid
                            self.real_data[i, insert_start:insert_end] << FunctionCall(
                                [self._Pg_t[f]], "to_array_1d"
                            )
                            insert_start << insert_start + self._Ngrid
                    else:
                        with ForLoopContext(1, self._Ns_tot, "f") as f:
                            insert_end << insert_end + self._Ngrid
                            self.real_data[i, insert_start:insert_end] << FunctionCall(
                                [self._Pg_c[f]], "to_array_1d"
                            )
                            insert_start << insert_start + self._Ngrid
                    """
                    # Pack integer data so real_data can be sorted into correct blocks in `lp_reduce`
                    self.int_data[i, 1] << insert_len
                    self.int_data[i, 2] << self._Ns
                    if self.sources.diffuse:
                        self.int_data[i, 3] << 1
                    else:
                        self.int_data[i, 3] << 0
                    if self.sources.atmospheric:
                        self.int_data[i, 4] << 1
                    else:
                        self.int_data[i, 4] << 0
                    self.int_data[i, 5] << self._Ngrid

                    self.int_data[i, 6:"5+insert_len"] << FunctionCall(
                        [
                            FunctionCall(
                                [self._event_type[start:end]], "to_array_1d"
                            )
                        ],
                        "to_int",
                    )

    def _parameters(self):
        """
        To write the parameters section of the Stan file.
        """

        with ParametersContext():

            # For point sources, L and src_index can be shared or
            # independent.
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

            # Specify F_diff and diff_index to characterise the diffuse comp
            if self.sources.diffuse:

                diff_index_min, diff_index_max = self._diff_index_par_range

                self._F_diff = ParameterDef("F_diff", "real", 0, None)
                self._diff_index = ParameterDef(
                    "diff_index", "real", diff_index_min, diff_index_max
                )

            # Atmo spectral shape is fixed, but normalisation can move.
            if self.sources.atmospheric:

                self._F_atmo = ParameterDef("F_atmo", "real", 0.0, None)

            # Vector of latent true source energies for each event
            self._Esrc = ParameterVectorDef(
                "Esrc", "vector", self._N_str, self._Esrc_min, self._Esrc_max
            )

    def _transformed_parameters(self):
        """
        To write the transformed parameters section of the Stan file.
        """

        # The likelihood is defined here, simplifying the code in the
        # model section for readability
        with TransformedParametersContext():

            # Expected number of events for different components
            self._Nex = ForwardVariableDef("Nex", "real")
            self._Nex_atmo = ForwardVariableDef("Nex_atmo", "real")
            self._Nex_src = ForwardVariableDef("Nex_src", "real")
            self._Nex_diff = ForwardVariableDef("Nex_diff", "real")

            # Total flux
            self._Ftot = ForwardVariableDef("Ftot", "real")

            # Total flux form point sources
            self._F_src = ForwardVariableDef("Fs", "real")

            # Different definitions of fractional association
            self._f_arr = ForwardVariableDef("f_arr", "real")
            self._f_det = ForwardVariableDef("f_det", "real")
            self._f_arr_astro = ForwardVariableDef("f_arr_astro", "real")
            self._f_det_astro = ForwardVariableDef("f_det_astro", "real")

            # Decide how many source components we have and calculate
            # `logF` accordingly.
            # These will be used as weights in the mixture model likelihood
            # We store the log probability for each source comp in `lp`.
            if self.sources.diffuse and self.sources.atmospheric:

                self._F = ForwardVariableDef("F", "vector[Ns+2]")
                self._logF = ForwardVariableDef("logF", "vector[Ns+2]")

                if self._nshards in [0, 1]:
                    self._lp = ForwardArrayDef("lp", "vector[Ns+2]", self._N_str)

                n_comps_max = "Ns+2"
                N_tot_t = "[Ns+2]"
                N_tot_c = "[Ns+1]"

            elif self.sources.diffuse or self.sources.atmospheric:

                self._F = ForwardVariableDef("F", "vector[Ns+1]")
                self._logF = ForwardVariableDef("logF", "vector[Ns+1]")

                if self._nshards in [0, 1]:
                    self._lp = ForwardArrayDef("lp", "vector[Ns+1]", self._N_str)

                n_comps_max = "Ns+1"
                N_tot_t = N_tot_c = "[Ns+1]"

            else:

                self._F = ForwardVariableDef("F", "vector[Ns]")
                self._logF = ForwardVariableDef("logF", "vector[Ns]")

                if self._nshards in [0, 1]:
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

            """
            if "cascades" in self._event_types and "tracks" in self._event_types:

                self._logp_c = ForwardVariableDef("logp_c", "real")
                self._logp_t = ForwardVariableDef("logp_t", "real")
            """

            if self._nshards not in [0, 1]:
                # Create vector of parameters
                # Global pars are src_index, diff_index, logF
                # Count number of pars:
                num_of_pars = 0
                if self._shared_luminosity:
                    num_of_pars += 1
                else:
                    num_of_pars += self._Ns

                if self._shared_src_index:
                    num_of_pars += 1
                else:
                    num_of_pars += self._Ns

                if self.sources.diffuse:
                    num_of_pars += 2
                if self.sources.atmospheric:
                    num_of_pars += 1

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
                        self._local_pars[i][1 : length] << self._Esrc[start : self._N]
                        
                    # Else, only relevant for last shard if it's shorter
                    with ElseBlockContext():
                        self._local_pars[i] << self._Esrc[start:end]


            else:

                # Latent arrival energies for each event
                self._E = ForwardVariableDef("E", "vector[N]")

            self._F_src << 0.0
            self._Nex_src << 0.0
            if self.sources.atmospheric:
                self._Nex_atmo << 0.0

            # For each source, calculate the number flux and update F, logF
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
                                src_index_ref, self._Esrc_min/(1+self._z[k]), self._Esrc_max/(1+self._z[k]),
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

            # For each source, calculate the exposure via interpolation
            # and then the expected number of events
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

            if "cascades" in self._event_types:

                self._Nex_c << FunctionCall([self._F, self._eps_c], "get_Nex")

            if "tracks" in self._event_types and "cascades" in self._event_types:

                self._Nex_src << self._Nex_src_t + self._Nex_src_c
                self._Nex_diff << self._Nex_diff_t + self._Nex_diff_c
                self._Nex << self._Nex_t + self._Nex_c

                # Relative probability of event types
                # How does this relate to the likelihood?
                # see l. ~1840 something
                self._logp_c << self._Nex_c / self._Nex
                self._logp_c << StringExpression(["log(", self._logp_c, ")"])
                self._logp_t << self._Nex_t / self._Nex
                self._logp_t << StringExpression(["log(", self._logp_t, ")"])

            elif "tracks" in self._event_types:

                self._Nex_src << self._Nex_src_t
                self._Nex_diff << self._Nex_diff_t
                self._Nex << self._Nex_t

            elif "cascades" in self._event_types:

                self._Nex_src << self._Nex_src_c
                self._Nex_diff << self._Nex_diff_c
                self._Nex << self._Nex_c

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

            if self.sources.diffuse and self.sources.atmospheric:

                self._k_diff = "Ns + 1"
                self._k_atmo = "Ns + 2"

            elif self.sources.diffuse:

                self._k_diff = "Ns + 1"

            elif self.sources.atmospheric:

                self._k_atmo = "Ns + 1"

            # Evaluate logF, packup global parameters
            self._logF << StringExpression(["log(", self._F, ")"])

            if self._nshards not in [0, 1]:
                if self._shared_src_index:
                    self._global_pars[1] << self._src_index
                    idx = 2
                else:
                    self._global_pars[1 : self._Ns] << self._src_index
                    idx = len(self.sources._point_sources) + 1
                if self.sources.diffuse:
                    self._global_pars[idx] << self._diff_index
                    idx += 1

                self._global_pars[idx : idx + self.sources.N - 1] << self._logF
                # Likelihood is evaluated in `lp_reduce`

            else:
                # Product over events => add log likelihoods
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

                            # Sum over sources => evaluate and store components
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

                                        # log_prob += log(p(Esrc|src_index))
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

                                        # E = Esrc / (1+z)
                                        self._E[i] << StringExpression(
                                            [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(p(omega_det|varpi, kappa))
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
                                        [StringExpression([k, " == ", self._k_diff])]
                                    ):

                                        # log_prob += log(p(Esrc|diff_index))
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

                                        # E = Esrc / (1+z)
                                        self._E[i] << StringExpression(
                                            [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(1/4pi)
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
                                        [StringExpression([k, " == ", self._k_atmo])]
                                    ):

                                        # log_prob += log(p(Esrc, omega | atmospheric source))
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

                                        # E = Esrc
                                        self._E[i] << self._Esrc[i]

                                # Detection effects
                                if self.detector_model_type == R2021DetectorModel:

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            self._dm["tracks"].energy_resolution(
                                                FunctionCall([self._E[i]], "log10"),
                                                FunctionCall([self._Edet[i]], "log10"),
                                                self._omega_det[i],
                                            ),
                                        ]
                                    )

                                else:

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            self._dm["tracks"].energy_resolution(
                                                self._E[i], self._Edet[i]
                                            ),
                                        ]
                                    )
                                """
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
                                """
                                StringExpression(
                                    [
                                        self._lp[i][k],
                                        " += log(",
                                        self._dm["tracks"].effective_area(
                                            self._E[i], self._omega_det[i]
                                        ),
                                        ")",
                                    ]
                                )

                    # Cascades
                    # See comments for tracks for more details, approach is the same
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

                                        # log_prob += log(p(Esrc | src_index))
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

                                        # E = Esrc / (1+z)
                                        self._E[i] << StringExpression(
                                            [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(p(omega_det | varpi, kappa))
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
                                        [StringExpression([k, " == ", self._k_diff])]
                                    ):

                                        # log_prob += log(p(Esrc | diff_index))
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

                                        # E = Esrc / (1+z)
                                        self._E[i] << StringExpression(
                                            [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(1/4pi)
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
                                        [StringExpression([k, " == ", self._k_atmo])]
                                    ):

                                        # log_prob += -inf (no atmo comp for cascades!)
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += negative_infinity()",
                                            ]
                                        )

                                        # E = Esrc
                                        self._E[i] << self._Esrc[i]

                                # Detection effects
                                # log_prob += log(p(Edet | E))
                                StringExpression(
                                    [
                                        self._lp[i][k],
                                        " += ",
                                        self._dm["cascades"].energy_resolution(
                                            self._E[i], self._Edet[i]
                                        ),
                                    ]
                                )

                                # log_prob += log(p(Edet > Edet_min | E))
                                StringExpression(
                                    [
                                        self._lp[i][k],
                                        " += log(",
                                        self._dm["cascades"].effective_area(
                                            self._E[i], self._omega_det[i]
                                        ),
                                        ")",
                                    ]
                                )

    def _model(self):
        """
        To write the model section of the Stan file.
        """

        with ModelContext():

            # Likelihood: e^(-Nex) \prod_(i=1)^N_events \sum_(k=1)^N_sources lp[i][k]
            # with ForLoopContext(1, self._N, "i") as i:

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

            # Add factor for relative probability of event types
            if "tracks" in self._event_types and "cascades" in self._event_types:

                StringExpression(["target += ", self._N_c, " * ", self._logp_c])
                StringExpression(["target += ", self._N_t, " * ", self._logp_t])

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

    def _generated_quantities(self):
        """
        To write the generated quantities section of the Stan file.
        """

        with GeneratedQuantitiesContext():

            # Calculation of individual source-event logprobs
            # Only when parallel mode on
            if self._nshards not in [0, 1]:

                # Define variables to store loglikelihood
                # and latent energies
                self._lp = ForwardArrayDef("lp", "vector[Ns_tot]", ["[N]"])
                self._E = ForwardVariableDef("E", "vector[N]")

                with ForLoopContext(1, self._N, "i") as i:
                    # TODO: change to use both tracks and cascades
                    if "tracks" in self._event_types and not "cascades" in self._event_types:
                        self._lp[i] << StringExpression(["log(F .* eps_t)"])
                    elif "cascades" in self._event_types and not "cascades" in self._event_types:
                        self._lp[i] << StringExpression(["log(F .* eps_c)"])
                    else:
                        self._lp[i] << StringExpression(["log(F .* eps_t) + log(F .* eps_c)"])

                    # Tracks
                    if "tracks" in self._event_types:
                        with IfBlockContext(
                            [
                                StringExpression(
                                    [self._event_type[i], " == ", self._track_type]
                                )
                            ]
                        ):
                            # Sum over sources => evaluate and store components
                            with ForLoopContext(1, self._Ns_tot, "k") as k:
                                # Point source components
                                if self.sources.point_source:
                                    with IfBlockContext(
                                        [StringExpression([k, " < ", self._Ns + 1])]
                                    ):
                                        if self._shared_src_index:
                                            src_index_ref = self._src_index
                                        else:
                                            src_index_ref = self._src_index[k]
                                        # log_prob += log(p(Esrc|src_index))
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

                                        # E = Esrc / (1+z)
                                        self._E[i] << StringExpression(
                                            [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(p(omega_det|varpi, kappa))
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
                                        [StringExpression([k, " == ", self._k_diff])]
                                    ):

                                        # log_prob += log(p(Esrc|diff_index))
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

                                        # E = Esrc / (1+z)
                                        self._E[i] << StringExpression(
                                            [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(1/4pi)
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
                                        [StringExpression([k, " == ", self._k_atmo])]
                                    ):

                                        # log_prob += log(p(Esrc, omega | atmospheric source))
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

                                        # E = Esrc
                                        self._E[i] << self._Esrc[i]

                                # Detection effects
                                if self.detector_model_type == R2021DetectorModel:

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            self._dm["tracks"].energy_resolution(
                                                FunctionCall([self._E[i]], "log10"),
                                                FunctionCall([self._Edet[i]], "log10"),
                                                self._omega_det[i],
                                            ),
                                        ]
                                    )

                                else:

                                    StringExpression(
                                        [
                                            self._lp[i][k],
                                            " += ",
                                            self._dm["tracks"].energy_resolution(
                                                self._E[i], self._Edet[i]
                                            ),
                                        ]
                                    )
                                """
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
                                """
                                StringExpression(
                                    [
                                        self._lp[i][k],
                                        " += log(",
                                        self._dm["tracks"].effective_area(
                                            self._E[i], self._omega_det[i]
                                        ),
                                        ")",
                                    ]
                                )

                    # Cascades
                    # See comments for tracks for more details, approach is the same
                    if "cascades" in self._event_types:

                        with IfBlockContext(
                            [
                                StringExpression(
                                    [self._event_type[i], " == ", self._cascade_type]
                                )
                            ]
                        ):

                            with ForLoopContext(1, self._Ns_tot, "k") as k:

                                # Point source components
                                if self.sources.point_source:

                                    with IfBlockContext(
                                        [StringExpression([k, " < ", self._Ns + 1])]
                                    ):

                                        if self._shared_src_index:
                                            src_index_ref = self._src_index
                                        else:
                                            src_index_ref = self._src_index[k]

                                        # log_prob += log(p(Esrc | src_index))
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

                                        # E = Esrc / (1+z)
                                        self._E[i] << StringExpression(
                                            [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(p(omega_det | varpi, kappa))
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
                                        [StringExpression([k, " == ", self._k_diff])]
                                    ):

                                        # log_prob += log(p(Esrc | diff_index))
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

                                        # E = Esrc / (1+z)
                                        self._E[i] << StringExpression(
                                            [self._Esrc[i], " / (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(1/4pi)
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
                                        [StringExpression([k, " == ", self._k_atmo])]
                                    ):

                                        # log_prob += -inf (no atmo comp for cascades!)
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += negative_infinity()",
                                            ]
                                        )

                                        # E = Esrc
                                        self._E[i] << self._Esrc[i]

                                # Detection effects
                                # log_prob += log(p(Edet | E))
                                StringExpression(
                                    [
                                        self._lp[i][k],
                                        " += ",
                                        self._dm["cascades"].energy_resolution(
                                            self._E[i], self._Edet[i]
                                        ),
                                    ]
                                )

                                # log_prob += log(p(detected | E))
                                StringExpression(
                                    [
                                        self._lp[i][k],
                                        " += log(",
                                        self._dm["cascades"].effective_area(
                                            self._E[i], self._omega_det[i]
                                        ),
                                        ")",
                                    ]
                                )
