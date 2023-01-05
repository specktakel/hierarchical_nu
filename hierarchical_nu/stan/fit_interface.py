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
    ForLoopContext,
    IfBlockContext,
    ElseBlockContext,
    ElseIfBlockContext,
    ModelContext,
    FunctionCall,
    UserDefinedFunction
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
        """
        
        if detector_model_type == R2021DetectorModel:
            includes.append("r2021_pdf.stan")
            R2021DetectorModel.generate_code(DistributionMode.PDF, rewrite=False, gen_type="lognorm")

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
            
            # Create a function to be used in map_rect in the model block
            # Signature is determined by stan's `map_rect` function
            lp_reduce = UserDefinedFunction(
                "lp_reduce",
                ["global", "local", "real_data", "int_data"],
                ["vector", "vector", "array[,] real", "array[] int"],
                "vector"
            )

            with lp_reduce:
                # Define function block
                # First, unpack arguments
                if self._shared_src_index:
                    src_index = ForwardVariableDef("src_index", "real")
                    src_index << StringExpression(["global[1]"])
                    idx = 2
                else:
                    src_index = ForwardVariableDef(
                        "src_index",
                        "vector["+str(len(self.sources._point_sources))+"]"
                    )
                    idx = len(self.sources._point_sources) + 1

                if self.sources.diffuse:
                    diff_index = ForwardVariableDef("diff_index", "real")
                    diff_index << StringExpression(["global[", idx, "]"])
                    idx += 1

                logF = ForwardVariableDef("logF", "vector["+str(self.sources.N)+"]")
                logF << StringExpression(["global[", idx, ":]"])

                # Local pars
                Esrc = ForwardVariableDef("Esrc", "vector[int_data[1]]")
                Esrc << StringExpression(["local"])
                E = ForwardVariableDef("E", "vector[int_data[1]]")

                #Unpack integer data, needed to interpret real data
                N = ForwardVariableDef("N", "int")
                N << StringExpression(["int_data[1]"])
                Ns = ForwardVariableDef("Ns", "int")
                Ns << StringExpression(["int_data[2]"])
                diffuse = ForwardVariableDef("diffuse", "int")
                diffuse << StringExpression(["int_data[3]"])
                atmo = ForwardVariableDef("atmo", "int")
                atmo << StringExpression(["int_data[4]"])
                Ngrid = ForwardVariableDef("Ngrid", "int")
                Ngrid << StringExpression(["int_data[5]"])

                if self.sources.diffuse and self.sources.atmospheric:
                    lp = ForwardArrayDef("lp", "vector[int_data[1]]", ["[int_data[2]+2]"])
                elif self.sources.diffuse or self.sources.atmospheric:
                    lp = ForwardArrayDef("lp", "vector[int_data[1]]", ["[int_data[2]+1]"])
                else:
                    lp = ForwardArrayDef("lp", "vector[int_data[1]]", ["[int_data[2]]"])

                event_type = ForwardArrayDef("event_type", "int", ["[N]"])
                event_type << StringExpression(["int_data[6:5+N]"])
                Edet = ForwardVariableDef("Edet", "vector[N]")
                Edet << StringExpression(["to_vector(real_data[1, 1:N])"])
                kappa = ForwardVariableDef("kappa", "vector[N]")
                kappa << StringExpression(["to_vector(real_data[2, 1:N])"])
                omega_det = ForwardArrayDef("omega_det", "vector[3]", ["[N]"])

                varpi = ForwardArrayDef("varpi", "vector[3]", ["[int_data[2]]"])
                with ForLoopContext(1, N, "i") as i:
                    omega_det[i] << StringExpression(["to_vector(real_data[3:5, i])"])
                with ForLoopContext(1, Ns, "i") as i:
                    varpi[i] << StringExpression(["to_vector(real_data[6:8, i])"])
                if self.sources.diffuse:
                    z = ForwardVariableDef("z", "vector[int_data[2]+int_data[3]]")
                    z << StringExpression(["to_vector(real_data[9, 1:int_data[2]+int_data[3]])"])
                else:
                    z = ForwardVariableDef("z", "vector[int_data[2]]")
                    z << StringExpression(["to_vector(real_data[9, 1:int_data[2])"])
                Esrc_min = ForwardVariableDef("Esrc_min", "real")
                Esrc_max = ForwardVariableDef("Esrc_max", "real")
                Esrc_min << StringExpression(["real_data[10, 1]"])
                Esrc_max << StringExpression(["real_data[10, 2]"])
                Eg = ForwardVariableDef("Egrid", "vector[int_data[5]]")
                Eg << StringExpression(["to_vector(real_data[11, 1:int_data[5]])"])

        
                if "tracks" in self._event_types and "cascades" in self._event_types:
                    Pdet_grid_t = ForwardArrayDef("Pdet_grid_t", "vector[int_data[5]]", ["[sum(int_data[2:4])]"])
                    Pdet_grid_c = ForwardArrayDef("Pdet_grid_c", "vector[int_data[5]]", ["[sum(int_data[2:4])]"])
                    with ForLoopContext(1, "Ns+atmo+diffuse", "f") as f:
                        Pdet_grid_t[f] << StringExpression(["to_vector(real_data[11+f, 1:int_data[5]])"])
                        Pdet_grid_c[f] << StringExpression(["to_vector(real_data[11+f+sum(int_data[2:4]), 1:int_data[5]])"])
                elif "tracks" in self._event_types:
                    Pdet_grid_t = ForwardArrayDef("Pdet_grid_t", "vector[int_data[5]]", ["[sum(int_data[2:4])]"])
                    with ForLoopContext(1, "sum(int_data[2:4])", "f") as f:
                        Pdet_grid_t[f] << StringExpression(["to_vector(real_data[11+f, 1:int_data[5]])"])
                        
                else:
                    Pdet_grid_c = ForwardArrayDef("Pdet_grid_c", "vector[int_data[5]]", ["[sum(int_data[2:4])]"])
                    with ForLoopContext(1, "sum(int_data[2:4])", "f") as f:
                        Pdet_grid_c[f] << StringExpression(["to_vector(real_data[11+f, 1:int_data[5]])"])
                        
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
                    lp[i] << logF
                    # Tracks
                    if "tracks" in self._event_types:
                        with IfBlockContext(
                            [
                                StringExpression(
                                    [event_type[i], " == ", track_type]
                                )
                            ]
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
                                                omega_det[i]
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
                                        " += log(interpolate(",
                                        Eg,
                                        ", ",
                                        Pdet_grid_t[k],
                                        ", ",
                                        E[i],
                                        "))",
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

                                # log_prob += log(p(Edet > Edet_min | E))
                                StringExpression(
                                    [
                                        lp[i][k],
                                        " += log(interpolate(",
                                        Eg,
                                        ", ",
                                        Pdet_grid_c[k],
                                        ", ",
                                        E[i],
                                        "))",
                                    ]
                                )
                results = ForwardArrayDef("results", "real", "[int_data[1]]")
                with ForLoopContext(1, N, "i") as i:
                    results[i] << FunctionCall([lp[i]], "log_sum_exp")
                ReturnStatement(["[sum(result)]'"])


    def _data(self):

        with DataContext():

            # Total number of detected events
            self._N = ForwardVariableDef("N", "int")
            self._N_str = ["[", self._N, "]"]

            

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
            self._Eg = ForwardVariableDef("E_grid", "vector[Ngrid]")

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

            # Interpolation grid used for the probability of detecting
            # an event above the minimum detected energy threshold
            if "tracks" in self._event_types:

                self._Pg_t = ForwardArrayDef("Pdet_grid_t", "vector[Ngrid]", N_pdet_str)

            if "cascades" in self._event_types:

                self._Pg_c = ForwardArrayDef("Pdet_grid_c", "vector[Ngrid]", N_pdet_str)

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

            # Create the rectangular data blocks for use in `map_rect`
            # Number of events per shard, integer division rounds down, so increase by one and pad last shard with dummy-entries
            #self._J = InstantVariableDef("J", "int", "")
            self._N_mod_J = ForwardVariableDef("N_mod_J", "int")
            self._N_mod_J << self._N % self._N_shards
            # StringExpression(["print(N)"])
            # StringExpression(["print(N_shards)"])
            # StringExpression(["print(N_mod_J)"])
            #with IfBlockContext([self._N_mod_J == 0]):
            #    self._J = InstantVariableDef("J", "int", "(N%/%N_shards)")
            #with ElseBlockContext():
            #    self._J = InstantVariableDef("J", "int", "(N%/%N_shards) + 1")
            #self._J_str = ["[", self._J, "]"]
            
            # StringExpression(["print(J)"])

            # Create data structures for integer and real data
            # neded integer data
            # N, event_type

            # needed real data
            # in length of events:
            # N: kappa, Edet, omega_det (x3)
            # Ns: varpi (x3), z, P_det_grid (Ns+atmo+diff, Ngrid)
            # Ngrid: Egrid, P_det_grid(Ns+atmo+diff, Ngrid)
            # scalar: Esrc_min, Esrc_max
            # second_dim = ForwardVariableDef("second_dim", "int")
            # P_det_grid should be stored transposed, I guess
            """
            with IfBlockContext([StringExpression([self._J, ">=", self._Ngrid])]):
                # if J greater than Ngrid, use J as second dim, else Ngrid
                # TODO re-work this in the end when I know how large the array is
                self.real_data = ForwardArrayDef(
                    "real_data",
                    "real",
                    ["[", self._N_shards, ", ", "11+Ns_tot", ", ", self._J, "]"]
                )
            with ElseBlockContext():
                self.real_data = ForwardArrayDef(
                    "real_data",
                    "real",
                    ["[", self._N_shards, ", ", "11+Ns_tot", ", ", self._Ngrid, "]"]
                )
            """
            self.real_data = ForwardArrayDef(
                "real_data",
                "real",
                ["[N_shards, 11+Ns_tot, J>=Ngrid ? J : Ngrid]"]
            )
            
            

            self.int_data = ForwardArrayDef(
                "int_data",
                "int",
                ["[", self._N_shards, ", ", "J+5", "]"]
            )
       
            with ForLoopContext(1, self._N_shards, "i") as i:
                start = ForwardVariableDef("start", "int")
                end = ForwardVariableDef("end", "int")
                insert_end = ForwardVariableDef("insert_end", "int")
                start << (i - 1) * self._J + 1
                with IfBlockContext([i != self._N_shards, "||", self._N_mod_J == 0]):
                    end << i * self._J
                    insert_end << self._J
                with ElseBlockContext():
                    end << start - 1 + self._N_mod_J
                    insert_end << self._N_mod_J
                self.real_data[i, 1, 1:insert_end] << FunctionCall([self._Edet[start:end]], "to_array_1d")
                self.real_data[i, 2, 1:insert_end] << FunctionCall([self._kappa[start:end]], "to_array_1d")

                with ForLoopContext(1, insert_end, "f") as f:
                    self.real_data[i, 3:5, f] << FunctionCall([self._omega_det[f]], "to_array_1d")
                with ForLoopContext(1, self._Ns, "f") as f:
                    self.real_data[i, 6:8, f] << FunctionCall([self._varpi[f]], "to_array_1d")
                if self.sources.diffuse:
                    self.real_data[i, 9, 1:"Ns+1"] << FunctionCall([self._z], "to_array_1d")
                else:
                    self.real_data[i, 9, 1:self._Ns] << FunctionCall([self._z], "to_array_1d")
                self.real_data[i, 10, 1] << self._Esrc_min
                self.real_data[i, 10, 2] << self._Esrc_max
                self.real_data[i, 11, 1:self._Ngrid] << FunctionCall([self._Eg], "to_array_1d")
                if "tracks" in self._event_types and "cascades" in self._event_types:
                    with ForLoopContext(1, self._Ns_tot, "f") as f:
                        self.real_data[i, "11+f", 1:self._Ngrid] << FunctionCall([self._Pg_t[f]], "to_array_1d")
                        self.real_data[i, "11+f+Ns_tot", 1:self._Ngrid] << FunctionCall([self._Pg_c[f]], "to_array_1d")
                elif "tracks" in self._event_types:
                    with ForLoopContext(1, self._Ns_tot, "f") as f:
                        self.real_data[i, "11+f", 1:self._Ngrid] << FunctionCall([self._Pg_t[f]], "to_array_1d")
                else:
                    with ForLoopContext(1, self._Ns_tot, "f") as f:
                        self.real_data[i, "11+f", 1:self._Ngrid] << FunctionCall([self._Pg_c[f]], "to_array_1d")
                    
                
                """
                # z has one entry more if there is a diffuse source
                if self.sources.diffuse:
                    self.real_data[i][3][1:self._Ns+1] << self._z
                else:
                    self.real_data[i][3][1:self._Ns] << self._z
                self.real_data[i][3][self._Ns+1] << self._Esrc_min
                self.real_data[i][3][self._Ns+2] << self._Esrc_min
                self.real_data[i][4][1:self._Ngrid] << self._Eg   # maybe cast to array
                
                #loop over all source components, including backgrounds if they exist
                """
                #with ForLoopContext(1:self._Ns, "l") as l:
                self.int_data[i, 1] << self._N     
                self.int_data[i, 2] << self._Ns 
                self.int_data[i, 3] << 1 if self.sources.diffuse else 0
                self.int_data[i, 4] << 1 if self.sources.atmospheric else 0
                self.int_data[i, 5] << self._Ngrid
                with IfBlockContext([i != self._N_shards, "||", self._N_mod_J == 0]):
                    self.int_data[i, 6:"5+J"] << FunctionCall([FunctionCall([self._event_type[start:end]], "to_array_1d")], "to_int")
                with ElseBlockContext():
                    self.int_data[i, 6:"5+N_mod_J"] << FunctionCall([FunctionCall([self._event_type[start:self._N]], "to_array_1d")], "to_int")

                
                #ForwardArrayDef("Pdet_grid_t", "vector[Ngrid]", N_pdet_str)
                # Find out how many sources Ns+x
                """if "tracks" in self._event_types and "cascades" in self._event_types:
                    with ForLoopContext(1, self._Ns_tot, "f") as f:
                        self.real_data[i][10+f][1:self._Ngrid]

                elif "tracks" in self._event_types:
                    pass
                else: 
                    pass"""
                # StringExpression(["print(int_data[i])"])
            

                


            


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

            # Latent arrival energies for each event
            self._E = ForwardVariableDef("E", "vector[N]")

            # Decide how many source components we have and calculate
            # `logF` accordingly.
            # These will be used as weights in the mixture model likelihood
            # We store the log probability for each source comp in `lp`.
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

            if "cascades" in self._event_types and "tracks" in self._event_types:

                self._logp_c = ForwardVariableDef("logp_c", "real")
                self._logp_t = ForwardVariableDef("logp_t", "real")

            # Create vector of parameters
            # Global pars are src_index, diff_index, L, F_atmo
            # Count number of pars:
            num_of_pars = 0
            global_pars_string = ""
            if self._shared_luminosity:
                num_of_pars += 1
                global_pars_string += "L"
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

            self._global_pars = ForwardVariableDef("global_pars", f"vector[{num_of_pars}]")
            

            self._local_pars = ForwardArrayDef("local_pars", "vector[J]", self._N_shards_str)

            with ForLoopContext(1, self._N_shards, "i") as i:
                start = ForwardVariableDef("start", "int")
                end = ForwardVariableDef("end", "int")
                start << (i - 1) * self._J + 1
                end << i * self._J
                with IfBlockContext([i != self._N_shards, "||", self._N_mod_J == 0]):
                    self._local_pars[i] << self._Esrc[start:end]
                #with ElseIfBlockContext([self._N_mod_J == 0]):
                #    self._local_pars[i] << self._Esrc[start:self._N]
                with ElseBlockContext():
                    self._local_pars[i][1:self._N_mod_J] << self._Esrc[start:self._N]


            self._F_src << 0.0
            self._Nex_src << 0.0
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

                k_diff = "Ns + 1"
                k_atmo = "Ns + 2"

            elif self.sources.diffuse:

                k_diff = "Ns + 1"

            elif self.sources.atmospheric:

                k_atmo = "Ns + 1"

            # Main model loop where likelihood is evaluated
            self._logF << StringExpression(["log(", self._F, ")"])
            #idx = ForwardVariableDef("idx", "int")
            if self._shared_src_index:
                self._global_pars[1] << self._src_index
                idx = 2
            else:
                self._global_pars[1:self._Ns] << self._src_index
                idx = len(self.sources._point_sources) + 1
            if self.sources.diffuse:
                self._global_pars[idx] << self._diff_index
                idx += 1
            
            self._global_pars[idx:idx+self.sources.N-1] << self._logF

            # Product over events => add log likelihoods
            # Starting here, everything needs to go to lp_reduce!
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
                                    [StringExpression([k, " == ", k_diff])]
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
                                    [StringExpression([k, " == ", k_atmo])]
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
                                            self._omega_det[i]
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
                                    [StringExpression([k, " == ", k_diff])]
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
                                    [StringExpression([k, " == ", k_atmo])]
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
        """
        To write the model section of the Stan file.
        """

        with ModelContext():

            # Likelihood: e^(-Nex) \prod_(i=1)^N_events \sum_(k=1)^N_sources lp[i][k]
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
