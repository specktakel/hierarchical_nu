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
        includes: List[str] = ["interpolation.stan", "utils.stan", "vMF.stan", "power_law.stan"],
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
                    # Unpack integer data, needed to interpret real data
                    # Use InstantVariableDef to save on lines
                    N = InstantVariableDef("N", "int", ["int_data[1]"])
                    Ns = InstantVariableDef("Ns", "int", ["int_data[2]"])
                    diffuse = InstantVariableDef("diffuse", "int", ["int_data[3]"])
                    atmo = InstantVariableDef("atmo", "int", ["int_data[4]"])
                    Ns_tot = InstantVariableDef("Ns_tot", "int", ["Ns+atmo+diffuse"])

                    aeff_len_t = InstantVariableDef("aeff_len_t", "int", ["int_data[5]"])
                    aeff_len_c = InstantVariableDef("aeff_len_c", "int", ["int_data[6]"])
                    #with IfBlockContext([aeff_len_t, " > ", 0]):
                    aeff_egrid_t = ForwardVariableDef("aeff_egrid_t", "vector[aeff_len_t]")
                    aeff_slice_t = ForwardArrayDef("aeff_slice_t", "vector[aeff_len_t]", ["[Ns]"])
                    #with IfBlockContext([aeff_len_c, " > ", 0]):
                    aeff_egrid_c = ForwardVariableDef("aeff_egrid_c", "vector[aeff_len_c]")
                    aeff_slice_c = ForwardArrayDef("aeff_slice_c", "vector[aeff_len_c]", ["[Ns]"])
                                        
                    start = ForwardVariableDef("start", "int")
                    end = ForwardVariableDef("end", "int")
                    length = ForwardVariableDef("length", "int")
                    
                    # Get global parameters
                    # Check for shared index
                    if self._shared_src_index:
                        src_index = ForwardVariableDef("src_index", "real")
                        src_index << StringExpression(["global[1]"])
                        idx = 2
                    else:
                        src_index = ForwardVariableDef("src_index", "vector[Ns]")
                        src_index << StringExpression(["global[1:Ns]"])
                        idx = len(self.sources._point_source) + 1

                    # Get diffuse index
                    if self.sources.diffuse:
                        diff_index = ForwardVariableDef("diff_index", "real")
                        diff_index << StringExpression(["global[", idx, "]"])
                        idx += 1

                    logF = ForwardVariableDef("logF", "vector[Ns_tot]")
                    logF << StringExpression(["global[", idx, ":]"])

                    # Local pars are only source energies
                    E = ForwardVariableDef("E", "vector[N]")
                    E << StringExpression(["local[:N]"])
                    Esrc = ForwardVariableDef("Esrc", "vector[N]")

                    # Define indices for unpacking of real_data
                    start << 1
                    length << N
                    end << N

                    # Define variable to store loglikelihood
                    lp = ForwardArrayDef("lp", "vector[Ns_tot]", ["[N]"])

                    # Unpack event types (track or cascade)
                    event_type = ForwardArrayDef("event_type", "int", ["[N]"])
                    event_type << StringExpression(["int_data[7:6+N]"])

                    Edet = ForwardVariableDef("Edet", "vector[N]")
                    Edet << FunctionCall(["real_data[start:end]"], "to_vector")
                    # Shift indices appropriate amount for next batch of data
                    start << start + length

                    #end << end + length
                    #kappa = ForwardVariableDef("kappa", "vector[N]")
                    #kappa << StringExpression(["to_vector(real_data[start:end])"])
                    #start << start + length

                    omega_det = ForwardArrayDef("omega_det", "vector[3]", ["[N]"])
                    # Loop over events to unpack reconstructed direction
                    with ForLoopContext(1, N, "i") as i:
                        end << end + 3
                        omega_det[i] << StringExpression(
                            ["to_vector(real_data[start:end])"]
                        )
                        start << start + 3

                    varpi = ForwardArrayDef("varpi", "vector[3]", ["[Ns]"])
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


                    if self.sources.point_source:
                        spatial_loglike = ForwardArrayDef("spatial_loglike", "real", ["[Ns, N]"])
                        with ForLoopContext(1, Ns, "k") as k:
                            end << end + length
                            spatial_loglike[k] << StringExpression(["real_data[start:end]"])
                            start << start + length

                        if "tracks" in self._event_types:
                            end << end + aeff_len_t
                            aeff_egrid_t << StringExpression(["to_vector(real_data[start:end])"])
                            start << start + aeff_len_t

                            with ForLoopContext(1, Ns, "k") as k:
                                end << end + aeff_len_t
                                aeff_slice_t[k] << StringExpression(["to_vector(real_data[start:end])"])
                                start << start + aeff_len_t

                        if "cascades" in self._event_types:
                            end << end + aeff_len_c
                            aeff_egrid_c << StringExpression(["to_vector(real_data[start:end])"])
                            start << start + aeff_len_c

                            with ForLoopContext(1, Ns, "k") as k:
                                end << end + aeff_len_c
                                aeff_slice_c[k] << StringExpression(["to_vector(real_data[start:end])"])
                                start << start + aeff_len_c

                    Esrc_min = ForwardVariableDef("Esrc_min", "real")
                    Esrc_max = ForwardVariableDef("Esrc_max", "real")
                    Emin = ForwardVariableDef("Emin", "real")
                    Emax = ForwardVariableDef("Emax", "real")
                    if self.sources.diffuse:
                        Ediff_min = ForwardVariableDef("Ediff_min", "real")
                        Ediff_max = ForwardVariableDef("Ediff_max", "real")
                    Emin_at_det = ForwardVariableDef("Emin_at_det", "real")
                    Emax_at_det = ForwardVariableDef("Emax_at_det", "real")

                    end << end + 1
                    Esrc_min << StringExpression(["real_data[start]"])
                    start << start + 1

                    end << end + 1
                    Esrc_max << StringExpression(["real_data[start]"])
                    start << start + 1

                    if self.sources.diffuse:
                        end << end + 1
                        Ediff_min << StringExpression(["real_data[start]"])
                        start << start + 1

                        end << end + 1
                        Ediff_max << StringExpression(["real_data[start]"])
                        start << start + 1

                    end << end + 1
                    Emin << StringExpression(["real_data[start]"])
                    start << start + 1

                    end << end + 1
                    Emax << StringExpression(["real_data[start]"])
                    start << start + 1

                    end << end + 1
                    Emin_at_det << StringExpression(["real_data[start]"])
                    start << start + 1

                    end << end + 1
                    Emax_at_det << StringExpression(["real_data[start]"])
                    start << start + 1

                    # Define tracks and cascades to sort events into correct detector response
                    if "tracks" in self._event_types:
                        track_type = ForwardVariableDef("track_type", "int")
                        track_type << TRACKS
                        eres_tracks = ForwardVariableDef("eres_tracks", "real")
                        aeff_tracks = ForwardVariableDef("aeff_tracks", "real")

                    if "cascades" in self._event_types:
                        cascade_type = ForwardVariableDef("cascade_type", "int")
                        cascade_type << CASCADES
                        eres_cascades = ForwardVariableDef("eres_cascades", "real")

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

                        lp[i] << logF

                        # Tracks
                        if "tracks" in self._event_types:
                            with IfBlockContext(
                                [StringExpression([event_type[i], " == ", track_type])]
                            ):
                                # Selection effects
                                if self.detector_model_type == R2021DetectorModel:
                                    eres_tracks << self._dm["tracks"].energy_resolution(
                                        FunctionCall([E[i]], "log10"),
                                        FunctionCall([Edet[i]], "log10"),
                                        omega_det[i],
                                    )

                                else:
                                    eres_tracks << self._dm["tracks"].energy_resolution(
                                        E[i], Edet[i]
                                    )

                                aeff_tracks << FunctionCall(
                                    [self._dm["tracks"].effective_area(E[i], omega_det[i])],
                                    "log"
                                )
                                # Sum over sources => evaluate and store components
                                with ForLoopContext(1, "Ns+atmo+diffuse", "k") as k:

                                    if self.sources.point_source:
                                        # Point source components
                                        with IfBlockContext(
                                            [StringExpression([k, " < ", Ns + 1])]
                                        ):

                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    "log(",
                                                    FunctionCall(
                                                        [
                                                            aeff_egrid_t,
                                                            aeff_slice_t[k],
                                                            E[i],
                                                        ],
                                                        "interpolate"
                                                    ),
                                                    " + 1e-10)",
                                                ]
                                            )
                                            
                                            if self._shared_src_index:
                                                src_index_ref = src_index
                                            else:
                                                src_index_ref = src_index[k]

                                            # E = Esrc / (1+z)
                                            Esrc[i] << StringExpression(
                                                [E[i], " * (", 1 + z[k], ")"]
                                            )
                                            # log_prob += log(p(Esrc|src_index))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    self._src_spectrum_lpdf(
                                                        E[i],
                                                        src_index_ref,
                                                        Esrc_min / (1 + z[k]),
                                                        Esrc_max / (1 + z[k]),
                                                        #Emin,
                                                        #Emax,
                                                    ),
                                                    #"dbbpl_logpdf(",
                                                    #E[i], 
                                                    #", ",
                                                    #src_index_ref,
                                                    #", ",
                                                    #Emin_at_det, 
                                                    #", ",
                                                    #Esrc_min/(1+z[k]),
                                                    #", ",
                                                    #Esrc_max/(1+z[k]),
                                                    #", ",
                                                    #Emax_at_det,
                                                    #")"
                                                ]
                                            )

                                            if self.detector_model_type == R2021DetectorModel:
                                                
                                                StringExpression(
                                                    [
                                                        lp[i][k],
                                                        " += ",
                                                        self._dm["tracks"].energy_resolution(
                                                            FunctionCall([E[i]], "log10"),
                                                            FunctionCall([Edet[i]], "log10"),
                                                            varpi[k],
                                                        ),
                                                    ]
                                                )

                                            else:

                                                StringExpression(
                                                    [
                                                        lp[i][k],
                                                        " += ",
                                                        eres_tracks,
                                                    ]
                                                )

                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    spatial_loglike[k, i],
                                                ]
                                            )

                                    # Diffuse component
                                    if self.sources.diffuse:

                                        with IfBlockContext(
                                            [StringExpression([k, " == ", k_diff])]
                                        ):

                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    aeff_tracks,
                                                ]
                                            )
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    eres_tracks,
                                                ]
                                            )

                                            # E = Esrc / (1+z)
                                            #E[i] << StringExpression(
                                            #    [Esrc[i], " / (", 1 + z[k], ")"]
                                            #)
                                            Esrc[i] << E[i] * (1 + z[k])

                                            # log_prob += log(p(Esrc|diff_index))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    self._diff_spectrum_lpdf(
                                                        E[i],
                                                        diff_index,
                                                        Ediff_min / (1. + z[k]),
                                                        Ediff_max / (1. + z[k]),
                                                    ),
                                                ]
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
                                            
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    aeff_tracks,
                                                ]
                                            )
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    eres_tracks,
                                                ]
                                            )

                                            # E = Esrc
                                            Esrc[i] << E[i]

                                            # log_prob += log(p(Esrc, omega | atmospheric source))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    FunctionCall(
                                                        [
                                                            self._atmo_flux_func(
                                                                E[i],
                                                                omega_det[i],
                                                            )
                                                            / self._atmo_flux_integral
                                                        ],
                                                        "log",
                                                    ),
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

                                eres_cascades << self._dm["cascades"].energy_resolution(
                                    E[i], Edet[i]
                                )

                                with ForLoopContext(1, "Ns+atmo+diffuse", "k") as k:

                                    # Point source components
                                    if self.sources.point_source:

                                        with IfBlockContext(
                                            [StringExpression([k, " < ", Ns + 1])]
                                        ):
                                            
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    "log(",
                                                    FunctionCall(
                                                        [
                                                            aeff_egrid_c,
                                                            aeff_slice_c[k],
                                                            E[i],
                                                        ],
                                                        "interpolate"
                                                    ),
                                                    " + 1e-10)",
                                                ]
                                            )

                                            if self._shared_src_index:
                                                src_index_ref = src_index
                                            else:
                                                src_index_ref = src_index[k]

                                            # log_prob += log(p(Esrc | src_index))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    self._src_spectrum_lpdf(
                                                        E[i],
                                                        src_index_ref,
                                                        Esrc_min / (1 + z[k]),
                                                        Esrc_max / (1 + z[k]),
                                                    ),
                                                ]
                                            )

                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    eres_cascades,
                                                ]
                                            )

                                            # E = Esrc / (1+z)
                                            Esrc[i] << StringExpression(
                                                [Esrc[i], " * (", 1 + z[k], ")"]
                                            )

                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    spatial_loglike[k, i],
                                                ]
                                            )

                                    # Diffuse component
                                    if self.sources.diffuse:

                                        with IfBlockContext(
                                            [StringExpression([k, " == ", k_diff])]
                                        ):

                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += log(",
                                                    self._dm["cascades"].effective_area(
                                                        E[i], omega_det[i]
                                                    ),
                                                    " + 1e-10)",
                                                ]
                                            )

                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    eres_cascades,
                                                ]
                                            )

                                            # log_prob += log(p(Esrc | diff_index))
                                            StringExpression(
                                                [
                                                    lp[i][k],
                                                    " += ",
                                                    self._diff_spectrum_lpdf(
                                                        E[i],
                                                        diff_index,
                                                        Ediff_min / (1. + z[k]),
                                                        Ediff_max / (1. + z[k]),
                                                    ),
                                                ]
                                            )

                                            # E = Esrc / (1+z)
                                            #E[i] << StringExpression(
                                            #    [Esrc[i], " / (", 1 + z[k], ")"]
                                            #)
                                            Esrc[i] << E[i] * (1. + z[k])

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
                                            Esrc[i] << E[i]

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

            #Energy range at the diffuse component at redshift z
            self._Ediff_min = ForwardVariableDef("Ediff_min", "real")
            self._Ediff_max = ForwardVariableDef("Ediff_max", "real")
            
            # Energy range at the detector
            self._Emin = ForwardVariableDef("Emin", "real")
            self._Emax = ForwardVariableDef("Emax", "real")

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

            # Aeff interpolation points for point sources
            if self.sources.point_source:
                if "tracks" in self._event_types:
                    self._aeff_len_t = ForwardVariableDef("aeff_len_t", "int")
                    self._aeff_slice_t = ForwardArrayDef(
                        "aeff_slice_t", f"vector[aeff_len_t]", self._Ns_str
                    )
                    self._aeff_egrid_t = ForwardVariableDef("aeff_egrid_t", "vector[aeff_len_t]")

                if "cascades" in self._event_types:
                    self._aeff_len_c = ForwardVariableDef("aeff_len_c", "int")
                    self._aeff_slice_c = ForwardArrayDef(
                        "aeff_slice_c", f"vector[aeff_len_c]", self._Ns_str
                    )
                    self._aeff_egrid_c = ForwardVariableDef("aeff_egrid_c", "vector[aeff_len_c]")

            if self.sources.diffuse and self.sources.atmospheric:

                N_pdet_str = self._Ns_2p_str

            elif self.sources.diffuse or self.sources.atmospheric:

                N_pdet_str = self._Ns_1p_str

            else:

                N_pdet_str = self._Ns_str

            # Don't need a grid for atmo as spectral shape is fixed, so pass single value.
            if self.sources.atmospheric:

                self._atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")

            if self._sources.point_source:
                
                self._stan_prior_src_index_mu = ForwardVariableDef("src_index_mu", "real")
                self._stan_prior_src_index_sigma = ForwardVariableDef("src_index_sigma", "real")
                # check for luminosity, if they all have the same prior
                if self._priors.luminosity.name in ["normal", "lognormal"]:
                    self._stan_prior_lumi_mu = ForwardVariableDef("lumi_mu", "real")
                    self._stan_prior_lumi_sigma = ForwardVariableDef("lumi_sigma", "real")
                elif self._priors.luminosity.name == "pareto":
                    self._stan_prior_lumi_xmin = ForwardVariableDef("lumi_xmin", "real")
                    self._stan_prior_lumi_alpha = ForwardVariableDef("lumi_alpha", "real")
            
            if self._sources.diffuse:
                
                self._stan_prior_f_diff_mu = ForwardVariableDef("f_diff_mu", "real")
                self._stan_prior_f_diff_sigma = ForwardVariableDef("f_diff_sigma", "real")

                self._stan_prior_diff_index_mu = ForwardVariableDef("diff_index_mu", "real")
                self._stan_prior_diff_index_sigma = ForwardVariableDef("diff_index_sigma", "real")

            if self._sources.atmospheric:
                
                self._stan_prior_f_atmo_mu = ForwardVariableDef("f_atmo_mu", "real")
                self._stan_prior_f_atmo_sigma = ForwardVariableDef("f_atmo_sigma", "real")

    def _transformed_data(self):
        """
        To write the transformed data section of the Stan file.
        """

        with TransformedDataContext():

            if self.sources.point_source:
                #Vector to hold pre-calculated spatial loglikes
                #This needs to be compatible with multiple point sources!
                self._spatial_loglike = ForwardArrayDef("spatial_loglike", "real", ["[Ns, N]"])
                with ForLoopContext(1, self._N, "i") as i:
                    with ForLoopContext(1, self._Ns, "k") as k:
                        StringExpression(
                            [
                                "spatial_loglike[k, i]",
                                " = vMF_lpdf(",
                                self._omega_det[i],
                                " | ",
                                self._varpi[k],
                                ", ",
                                self._kappa[i],
                                ")",
                            ]
                        )


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

            # Find largest permitted range of energies at the detector
            # TODO: not sure about this construct...
            self._Emin_at_det = ForwardVariableDef("Emin_at_det", "real")
            self._Emax_at_det = ForwardVariableDef("Emax_at_det", "real")
            self._Emin_at_det << self._Emin
            self._Emax_at_det << self._Emax

            with ForLoopContext(1, self._Ns, "k") as k:
                with IfBlockContext([self._Esrc_min / (1 + self._z[k]), " < ", self._Emin_at_det]):
                    self._Emin_at_det << self._Esrc_min / (1 + self._z[k])
                with IfBlockContext([self._Esrc_max / (1 + self._z[k]), " > ", self._Emax_at_det]):
                    self._Emax_at_det << self._Esrc_max / (1 + self._z[k])
            if self.sources.diffuse:
                with IfBlockContext([self._Ediff_min / (1 + self._z[self._Ns+1]), " < ", self._Emin_at_det]):
                    self._Emin_at_det << self._Ediff_min / (1. + self._z[self._Ns + 1])
                with IfBlockContext([self._Ediff_max / (1 + self._z[self._Ns+1]), " > ", self._Emax_at_det]):
                    self._Emax_at_det << self._Ediff_max / (1. + self._z[self._Ns + 1])

            if self._nshards not in [0, 1]:
                self._N_shards_use_this = ForwardVariableDef("N_shards_loop", "int")
                self._N_shards_use_this << self._N_shards
                # Create the rectangular data blocks for use in `map_rect`
                self._N_mod_J = ForwardVariableDef("N_mod_J", "int")
                self._N_mod_J << self._N % self._J
                # Find size for real_data array
                sd_events_J = 4    # reco energy, reco dir (unit vector)
                sd_varpi_Ns = 3    # coords of events in the sky (unit vector)
                sd_if_diff = 3     # redshift of diffuse component, Ediff_min/max
                sd_z_Ns = 1        # redshift of PS
                sd_other = 6       # Esrc_min, Esrc_max, Emin, Emax
                #Need Ns * N for spatial loglike, added extra in sd_string
                if self.sources.atmospheric and "tracks" in self._event_types:
                    sd_other += 1    # no atmo in cascades
                sd_string = (
                    f"{sd_events_J}*J + {sd_varpi_Ns}*Ns + {sd_z_Ns}*Ns + {sd_other} + J*Ns"
                )
                if self.sources.diffuse:
                    sd_string += f" + {sd_if_diff}"
                if self.sources.point_source:
                    if "tracks" in self._event_types:
                        sd_string += " + (Ns+1)*aeff_len_t"
                    if "cascades" in self._event_types:
                        sd_string += " + (Ns+1)*aeff_len_c"
                # Create data arrays
                self.real_data = ForwardArrayDef(
                    "real_data", "real", ["[N_shards,", sd_string, "]"]
                )

                self.int_data = ForwardArrayDef(
                    "int_data", "int", ["[", self._N_shards, ", ", "J+6", "]"]
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

                    if self.sources.point_source:
                        with ForLoopContext(1, self._Ns, "k") as k:
                            # Loop over sources
                            insert_end << insert_end + insert_len
                            #The double-index is needed because of a bug with the code generator
                            # if I use [k, start:end], a single line of "k;" is printed after entering
                            # the for loop
                            self.real_data[i, insert_start:insert_end] << self._spatial_loglike[k][start:end]
                            insert_start << insert_start + insert_len

                        # Pack aeff slices and the egrid of the slice
                        if "tracks" in self._event_types:

                            insert_end << insert_end + self._aeff_len_t
                            self.real_data[i, insert_start:insert_end] << FunctionCall([self._aeff_egrid_t], "to_array_1d")
                            insert_start << insert_start + self._aeff_len_t

                            with ForLoopContext(1, self._Ns, "k") as k:
                                insert_end << insert_end + self._aeff_len_t
                                self.real_data[i, insert_start:insert_end] << FunctionCall([self._aeff_slice_t[k]], "to_array_1d")
                                insert_start << insert_start + self._aeff_len_t

                        if "cascades" in self._event_types:

                            insert_end << insert_end + self._aeff_len_c
                            self.real_data[i, insert_start:insert_end] << FunctionCall([self._aeff_egrid_c], "to_array_1d")
                            insert_start << insert_start + self._aeff_len_c

                            with ForLoopContext(1, self._Ns, "k") as k:
                                insert_end << insert_end + self._aeff_len_c
                                self.real_data[i, insert_start:insert_end] << FunctionCall([self._aeff_slice_c[k]], "to_array_1d")
                                insert_start << insert_start + self._aeff_len_c

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Esrc_min
                    insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Esrc_max
                    insert_start << insert_start + 1

                    if self.sources.diffuse:
                        insert_end << insert_end + 1
                        self.real_data[i, insert_start] << self._Ediff_min
                        insert_start << insert_start + 1

                        insert_end << insert_end + 1
                        self.real_data[i, insert_start] << self._Ediff_max
                        insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Emin
                    insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Emax
                    insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Emin_at_det
                    insert_start << insert_start + 1

                    insert_end << insert_end + 1
                    self.real_data[i, insert_start] << self._Emax_at_det
                    insert_start << insert_start + 1

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
                    
                    if "tracks" in self._event_types:
                        self.int_data[i, 5] << self._aeff_len_t
                    else:
                        self.int_data[i, 5] << 0
                    if "cascades" in self._event_types:
                        self.int_data[i, 6] << self._aeff_len_c
                    else:
                        self.int_data[i, 6] << 0


                    self.int_data[i, 7:"6+insert_len"] << FunctionCall(
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
            #TODO change to energies at the detector
            #find largest allowed range in loop over sources

            #self._Esrc = ParameterVectorDef(
            #    "Esrc", "vector", self._N_str, self._Esrc_min, self._Esrc_max
            #)
            self._E = ParameterVectorDef(
                "E", "vector", self._N_str, self._Emin_at_det, self._Emax_at_det
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
                self._eres_tracks = ForwardVariableDef("eres_tracks", "real")      # use only for diffuse spectrum
                self._aeff_tracks = ForwardVariableDef("aeff_tracks", "real")   # for source, use instead IRF/Aeff at source dec


            if "cascades" in self._event_types:

                self._eps_c = ForwardVariableDef("eps_c", "vector" + N_tot_c)
                self._Nex_src_c = ForwardVariableDef("Nex_src_c", "real")
                self._Nex_diff_c = ForwardVariableDef("Nex_diff_c", "real")
                self._Nex_c = ForwardVariableDef("Nex_c", "real")
                self._Nex_src_c << 0.0
                self._eres_cascades = ForwardVariableDef("eres_cascades", "real") # same as above
                self._aeff_cascades = ForwardVariableDef("aeff_cascades", "real")

            if "cascades" in self._event_types and "tracks" in self._event_types:

                self._logp_c = ForwardVariableDef("logp_c", "real")
                self._logp_t = ForwardVariableDef("logp_t", "real")

            if self._nshards not in [0, 1]:
                # Create vector of parameters
                # Global pars are src_index, diff_index, logF
                # Count number of pars:
                if self._shared_luminosity:
                    num_of_pars = "1"
                else:
                    num_of_pars = " Ns"

                if self._shared_src_index:
                    num_of_pars += " + 1"
                else:
                    num_of_pars += " + Ns"

                if self.sources.diffuse:
                    num_of_pars += " + 2"
                if self.sources.atmospheric:
                    num_of_pars += " + 1"

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
                        self._local_pars[i][1 : length] << self._E[start : self._N]
                        
                    # Else, only relevant for last shard if it's shorter
                    with ElseBlockContext():
                        self._local_pars[i] << self._E[start:end]


            else:

                # Latent arrival energies for each event
                #self._E = ForwardVariableDef("E", "vector[N]")
                self._Esrc = ForwardVariableDef("Esrc", "vector[N]")

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
                    idx = len(self.sources._point_source) + 1
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
                            
                            # Detection effects
                            # Detection probability for diffuse sources
                            self._aeff_tracks << FunctionCall(
                                [
                                    self._dm["tracks"].effective_area(
                                            self._E[i], self._omega_det[i]
                                        )
                                ],
                                "log"
                            )

                            if self.detector_model_type == R2021DetectorModel:
                                # IRF is declination dependent, use for diffuse components
                                self._eres_tracks << \
                                    self._dm["tracks"].energy_resolution(
                                        FunctionCall([self._E[i]], "log10"),
                                        FunctionCall([self._Edet[i]], "log10"),
                                        self._omega_det[i],
                                    )

                            else:
                                # This can be reused for all source components
                                self._eres_tracks << \
                                    self._dm["tracks"].energy_resolution(
                                        self._E[i], self._Edet[i]
                                    )

                            # Sum over sources => evaluate and store components
                            with ForLoopContext(1, n_comps_max, "k") as k:

                                # Point source components
                                if self.sources.point_source:

                                    with IfBlockContext(
                                        [StringExpression([k, " < ", self._Ns + 1])]
                                    ):
                                        """
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                FunctionCall(
                                                    [
                                                        self._dm["tracks"].effective_area(
                                                                self._E[i], self._varpi[k]
                                                            )
                                                    ],
                                                    "log"
                                                )
                                            ]
                                        )
                                        """
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                "log(",
                                                FunctionCall(
                                                    [
                                                        self._aeff_egrid_t,
                                                        self._aeff_slice_t[k],
                                                        self._E[i]
                                                    ],
                                                    "interpolate"
                                                ),
                                                " + 1e-10)",
                                            ]
                                        )

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
                                                    self._E[i],
                                                    src_index_ref,
                                                    self._Esrc_min / (1 + self._z[k]),
                                                    self._Esrc_max / (1 + self._z[k]),
                                                    #self._Emin,
                                                    #self._Emax,
                                                ),
                                                #"dbbpl_logpdf(",
                                                #self._E[i], 
                                                #", ",
                                                #src_index_ref,
                                                #", ",
                                                #self._Emin_at_det, 
                                                #", ",
                                                #self._Esrc_min/(1+self._z[k]),
                                                #", ",
                                                #self._Esrc_max/(1+self._z[k]),
                                                #", ",
                                                #self._Emax_at_det,
                                                #")"
                                            ]
                                        )

                                        if self.detector_model_type == R2021DetectorModel:

                                            StringExpression(
                                                [
                                                    self._lp[i][k],
                                                    " += ",
                                                    self._dm["tracks"].energy_resolution(
                                                        FunctionCall([self._E[i]], "log10"),
                                                        FunctionCall([self._Edet[i]], "log10"),
                                                        self._varpi[k],
                                                    ),
                                                ]
                                            )

                                        else:

                                            StringExpression(
                                                [
                                                    self._lp[i][k],
                                                    " += ",
                                                    self._eres_tracks,
                                                ]
                                            )

                                        # E = Esrc / (1+z)
                                        self._Esrc[i] << StringExpression(
                                            [self._E[i], " * (", 1 + self._z[k], ")"]
                                        )

                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._spatial_loglike[k, i],
                                            ]
                                        )

                                # Diffuse component
                                if self.sources.diffuse:

                                    with IfBlockContext(
                                        [StringExpression([k, " == ", self._k_diff])]
                                    ):
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._eres_tracks,
                                            ]
                                        )

                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._aeff_tracks,
                                            ]
                                        )

                                        # E = Esrc / (1+z)
                                        self._Esrc[i] << self._E[i] * (1. + self._z[k])

                                        # log_prob += log(p(Esrc|diff_index))
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._diff_spectrum_lpdf(
                                                    self._E[i],
                                                    self._diff_index,
                                                    self._Ediff_min / (1. + self._z[k]),
                                                    self._Ediff_max / (1. + self._z[k]),
                                                ),
                                            ]
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
                                        
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._eres_tracks,
                                            ]
                                        )

                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._aeff_tracks,
                                            ]
                                        )
                                        
                                        # E = Esrc
                                        self._Esrc[i] << self._E[i]

                                        # log_prob += log(p(Esrc, omega | atmospheric source))
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                FunctionCall(
                                                    [
                                                        self._atmo_flux_func(
                                                            self._E[i],
                                                            self._omega_det[i],
                                                        )
                                                        / self._atmo_flux_integral
                                                    ],
                                                    "log",
                                                ),
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
                            # Detection effects
                            # log(p(Edet|E))
                            # Can be reused because it is not declination dependent
                            self._eres_cascades << self._dm["cascades"].energy_resolution(
                                self._E[i], self._Edet[i]
                            )

                            with ForLoopContext(1, n_comps_max, "k") as k:


                                StringExpression(
                                    [
                                        self._lp[i][k],
                                        " += ",
                                        self._eres_cascades,
                                    ]
                                )

                                # Point source components
                                if self.sources.point_source:

                                    with IfBlockContext(
                                        [StringExpression([k, " < ", self._Ns + 1])]
                                    ):

                                        StringExpression(
                                                [
                                                    self._lp[i][k],
                                                    " += log(",
                                                    self._dm["cascades"].effective_area(
                                                        self._E[i], self._varpi[k]
                                                    ),
                                                    " + 1e-10)",
                                                ]
                                            )

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
                                                    self._E[i],
                                                    src_index_ref,
                                                    self._Esrc_min / (1 + self._z[k]),
                                                    self._Esrc_max / (1 + self._z[k]),
                                                ),
                                            ]
                                        )

                                        # E = Esrc / (1+z)
                                        self._Esrc[i] << StringExpression(
                                            [self._E[i], " * (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(p(omega_det | varpi, kappa))
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._spatial_loglike[k, i],
                                            ]
                                        )

                                # Diffuse component
                                if self.sources.diffuse:

                                    with IfBlockContext(
                                        [StringExpression([k, " == ", self._k_diff])]
                                    ):

                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                "log(",
                                                FunctionCall(
                                                    [
                                                        self._aeff_egrid_c,
                                                        self._aeff_slice_c[k],
                                                        self._E[i]
                                                    ],
                                                    "interpolate"
                                                ),
                                                " + 1e-10)",
                                            ]
                                        )

                                        # log_prob += log(p(Esrc | diff_index))
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._diff_spectrum_lpdf(
                                                    self._E[i],
                                                    self._diff_index,
                                                    self._Ediff_min / (1. + self._z[k]),
                                                    self._Ediff_max / (1. + self._z[k]),
                                                ),
                                            ]
                                        )

                                        # E = Esrc / (1+z)
                                        self._Esrc[i] << self._E[i] * (1. + self._z[k])
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
                                        self._Esrc[i] << self._E[i]

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
                                    self._stan_prior_lumi_mu,
                                    self._stan_prior_lumi_sigma,
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

                StringExpression(
                    [
                        self._src_index,
                        " ~ ",
                        FunctionCall(
                            [self._stan_prior_src_index_mu, self._stan_prior_src_index_sigma],
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
                                self._stan_prior_f_diff_mu,
                                self._stan_prior_f_diff_sigma,
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
                            [self._stan_prior_diff_index_mu, self._stan_prior_diff_index_sigma],
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
                                self._stan_prior_f_atmo_mu,
                                self._stan_prior_f_atmo_sigma,
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
                self._Esrc = ForwardVariableDef("Esrc", "vector[N]")

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
                            with ForLoopContext(1, self._Ns_tot, "k") as k:
                                # Point source components
                                if self.sources.point_source:
                                    with IfBlockContext(
                                        [StringExpression([k, " < ", self._Ns + 1])]
                                    ):
                                        # E = Esrc / (1+z)
                                        self._Esrc[i] << StringExpression(
                                            [self._E[i], " * (", 1 + self._z[k], ")"]
                                        )

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
                                                    self._E[i],
                                                    src_index_ref,
                                                    self._Esrc_min / (1 + self._z[k]),
                                                    self._Esrc_max / (1 + self._z[k]),
                                                ),
                                                #"dbbpl_logpdf(",
                                                #self._E[i], 
                                                #", ",
                                                #src_index_ref,
                                                #", ",
                                                #self._Emin_at_det, 
                                                #", ",
                                                #self._Esrc_min/(1+self._z[k]),
                                                #", ",
                                                #self._Esrc_max/(1+self._z[k]),
                                                #", ",
                                                #self._Emax_at_det,
                                                #")"
                                            ]
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

                                        # Detection effects
                                        if self.detector_model_type == R2021DetectorModel:

                                            StringExpression(
                                                [
                                                    self._lp[i][k],
                                                    " += ",
                                                    self._dm["tracks"].energy_resolution(
                                                        FunctionCall([self._E[i]], "log10"),
                                                        FunctionCall([self._Edet[i]], "log10"),
                                                        self._varpi[k],
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

                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += log(",
                                                self._dm["tracks"].effective_area(
                                                    self._E[i], self._varpi[k]
                                                ),
                                                ")",
                                            ]
                                        )

                                # Diffuse component
                                if self.sources.diffuse:

                                    with IfBlockContext(
                                        [StringExpression([k, " == ", self._k_diff])]
                                    ):
                                        # E = Esrc / (1+z)
                                        self._Esrc[i] << StringExpression(
                                            [self._E[i], " * (", 1 + self._z[k], ")"]
                                        )

                                        # log_prob += log(p(Esrc|diff_index))
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                self._diff_spectrum_lpdf(
                                                    self._E[i],
                                                    self._diff_index,
                                                    self._Ediff_min / (1. + self._z[k]),
                                                    self._Ediff_max / (1. + self._z[k]),
                                                ),
                                            ]
                                        )

                                        # log_prob += log(1/4pi)
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                np.log(1 / (4 * np.pi)),
                                            ]
                                        )

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

                                # Atmospheric component
                                if self.sources.atmospheric:

                                    with IfBlockContext(
                                        [StringExpression([k, " == ", self._k_atmo])]
                                    ):
                                        
                                        # E = Esrc
                                        self._Esrc[i] << self._E[i]

                                        # log_prob += log(p(Esrc, omega | atmospheric source))
                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += ",
                                                FunctionCall(
                                                    [
                                                        self._atmo_flux_func(
                                                            self._E[i],
                                                            self._omega_det[i],
                                                        )
                                                        / self._atmo_flux_integral
                                                    ],
                                                    "log",
                                                ),
                                            ]
                                        )

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
                                                    self._E[i],
                                                    src_index_ref,
                                                    self._Esrc_min / (1 + self._z[k]),
                                                    self._Esrc_max / (1 + self._z[k]),
                                                ),
                                            ]
                                        )

                                        # E = Esrc / (1+z)
                                        self._Esrc[i] << StringExpression(
                                            [self._E[i], " * (", 1 + self._z[k], ")"]
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

                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += log(",
                                                self._dm["cascades"].effective_area(
                                                    self._E[i], self._varpi[k]
                                                ),
                                                " + 1e-10)",
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
                                                    self._E[i],
                                                    self._diff_index,
                                                    self._Ediff_min / (1. + self._z[k]),
                                                    self._Ediff_max / (1. + self._z[k]),
                                                ),
                                            ]
                                        )

                                        # E = Esrc / (1+z)
                                        self._Esrc[i] << StringExpression(
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

                                        StringExpression(
                                            [
                                                self._lp[i][k],
                                                " += log(",
                                                self._dm["cascades"].effective_area(
                                                    self._E[i], self._omega_det[i]
                                                ),
                                                " + 1e-10)",
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
                                        self._Esrc[i] << self._E[i]

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
