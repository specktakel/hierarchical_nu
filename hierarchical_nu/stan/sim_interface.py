from typing import List
from collections import OrderedDict
from astropy import units as u

from hierarchical_nu.detector.detector_model import DetectorModel
from hierarchical_nu.detector.r2021 import R2021DetectorModel

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
    ForwardVectorDef,
)

from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.events import TRACKS, CASCADES
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.detector.r2021 import R2021DetectorModel

from hierarchical_nu.source.source import Sources


class StanSimInterface(StanInterface):
    """
    An interface for generating the Stan simulation code.
    """

    def __init__(
        self,
        output_file: str,
        sources: Sources,
        detector_model_type: DetectorModel,
        atmo_flux_theta_points: int = 30,
        includes: List[str] = [
            "interpolation.stan",
            "utils.stan",
            "vMF.stan",
            "rejection_sampling.stan",
        ],
    ):
        """
        An interface for generating the Stan simulation code.

        :param output_file: Name of the file to write to
        :param sources: Sources object containing sources to be simulated
        :param detector_model_type: Type of the detector model to be used
        :param atmo_flux_theta_points: Number of points to use for the grid of
        atmospheric flux
        :param includes: List of names of stan files to include into the
        functions block of the generated file
        """

        if detector_model_type == R2021DetectorModel:
            if "r2021_rng.stan" not in includes:
                includes.append("r2021_rng.stan")
            R2021DetectorModel.generate_code(
                DistributionMode.RNG, rewrite=False, gen_type="histogram"
            )

        super().__init__(
            output_file=output_file,
            sources=sources,
            detector_model_type=detector_model_type,
            includes=includes,
        )

        self._atmo_flux_theta_points = atmo_flux_theta_points

    def _functions(self):
        """
        Write the functions section of the Stan file.
        """

        with FunctionsContext():

            # Include all the specified files
            for include_file in self._includes:
                _ = Include(include_file)

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

                atmo_flux_model = self.sources.atmospheric.flux_model

                # Increasing theta points too much makes compilation very slow
                # Could switch to passing array as data if problematic
                self._atmo_flux = atmo_flux_model.make_stan_function(
                    theta_points=self._atmo_flux_theta_points
                )

    def _data(self):
        """
        Write the data section of the Stan file.
        """

        with DataContext():

            self._N_poisson_t = ForwardVariableDef("N_poisson_t", "int")
            self._N_poisson_c = ForwardVariableDef("N_poisson_c", "int")

            # Useful strings depending on the total number of sources
            # Ns is the number of point sources
            self._Ns = ForwardVariableDef("Ns", "int")
            self._Ns_str = ["[", self._Ns, "]"]
            self._Ns_1p_str = ["[", self._Ns, "+1]"]
            self._Ns_2p_str = ["[", self._Ns, "+2]"]

            if self.sources.diffuse:

                N_int_str = self._Ns_1p_str

            else:

                N_int_str = self._Ns_str

            # True directions of point sources as a unit vector
            self._varpi = ForwardArrayDef("varpi", "unit_vector[3]", self._Ns_str)

            # Distance of sources in Mpc
            self._D = ForwardVariableDef("D", "vector[Ns]")

            # For diffuse and point sources, we have an interpolation grid
            # for the integral of expected number of events to pass
            if self.sources.diffuse or self.sources.point_source:

                self._Ngrid = ForwardVariableDef("Ngrid", "int")

            # Diffuse sources have a redshift (default z=0) a spectral index,
            # and a grid over this index for the interpolation mentioned above
            if self.sources.diffuse:

                self._z = ForwardVariableDef("z", "vector[Ns+1]")
                self._diff_index = ForwardVariableDef("diff_index", "real")
                self._diff_index_grid = ForwardVariableDef(
                    "diff_index_grid", "vector[Ngrid]"
                )

            else:

                self._z = ForwardVariableDef("z", "vector[Ns]")

            # Point sources can have shared/individual spectral indices, and
            # a grid over the spectral index is also passed, as for diffuse sources.
            if self.sources.point_source:

                if self._shared_src_index:
                    self._src_index = ForwardVariableDef("src_index", "real")
                else:
                    self._src_index = ForwardVariableDef("src_index", "vector[Ns]")

                self._src_index_grid = ForwardVariableDef(
                    "src_index_grid", "vector[Ngrid]"
                )

            # The energy range is specified at the source
            self._Esrc_min = ForwardVariableDef("Esrc_min", "real")
            self._Esrc_max = ForwardVariableDef("Esrc_max", "real")

            # Energy range that the detector should consider
            # Is influenced by parameterisation of energy resolution
            self._Emin = ForwardVariableDef("Emin", "real")
            self._Emax = ForwardVariableDef("Emax", "real")

            # Energy range considered for diffuse astrophysical sources, defined at redshift z of shell
            self._Ediff_min = ForwardVariableDef("Ediff_min", "real")
            self._Ediff_max = ForwardVariableDef("Ediff_max", "real")

            # For tracks, we specify Emin_det, and several parameters for the
            # rejection sampling, denoted by rs_...
            # Separate interpolation grids are also provided for tracks and cascades
            if "tracks" in self._event_types:

                self._Emin_det_t = ForwardVariableDef("Emin_det_t", "real")
                self._rs_bbpl_Eth_t = ForwardVariableDef("rs_bbpl_Eth_t", "real")
                self._rs_bbpl_gamma1_t = ForwardVariableDef("rs_bbpl_gamma1_t", "real")
                self._rs_bbpl_gamma2_scale_t = ForwardVariableDef(
                    "rs_bbpl_gamma2_scale_t", "real"
                )
                # self._rs_N_cosz_bins_t = ForwardVariableDef("rs_N_cosz_bins_t", "int")
                self._rs_cvals_t = ForwardArrayDef(
                    "rs_cvals_t", "real", [f"[{self.sources.N}]"]
                )
                #self._rs_cosz_bin_edges_t = ForwardArrayDef(
                #    "rs_cosz_bin_edges_t", "real", ["[rs_N_cosz_bins_t + 1]"]
                #)

                if self.sources.diffuse or self.sources.point_source:
                    self._integral_grid_t = ForwardArrayDef(
                        "integral_grid_t", "vector[Ngrid]", N_int_str
                    )

            # Similarly, we do the same for cascades. Different rejection sampling
            # parameters and interpolation grids are needed due to the different
            # shape of the effective area.
            if "cascades" in self._event_types:

                self._Emin_det_c = ForwardVariableDef("Emin_det_c", "real")
                self._rs_bbpl_Eth_c = ForwardVariableDef("rs_bbpl_Eth_c", "real")
                self._rs_bbpl_gamma1_c = ForwardVariableDef("rs_bbpl_gamma1_c", "real")
                self._rs_bbpl_gamma2_scale_c = ForwardVariableDef(
                    "rs_bbpl_gamma2_scale_c", "real"
                )
                #self._rs_N_cosz_bins_c = ForwardVariableDef("rs_N_cosz_bins_c", "int")
                self._rs_cvals_c = ForwardArrayDef(
                    "rs_cvals_c", "real", [f"[{self.sources.N}]"]
                )
                #self._rs_cosz_bin_edges_c = ForwardArrayDef(
                #    "rs_cosz_bin_edges_c", "real", ["[rs_N_cosz_bins_c + 1]"]
                #)

                if self.sources.diffuse or self.sources.point_source:
                    self._integral_grid_c = ForwardArrayDef(
                        "integral_grid_c", "vector[Ngrid]", N_int_str
                    )

            # We define the necessary source input parameters depending on
            # what kind of sources we have
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

            # v_lim sets the edge of the uniform sampling on a sphere, for example,
            # for Northern skies only. See sphere_lim_rng.
            self._v_lim = ForwardVariableDef("v_lim", "real")

            # The observation time
            self._T = ForwardVariableDef("T", "real")

    def _transformed_data(self):
        """
        Write the transformed data section of the Stan file.
        """

        with TransformedDataContext():

            # Decide how many flux entries, F, we have
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

                # Define track type as in package
                self._track_type = ForwardVariableDef("track_type", "int")
                self._track_type << TRACKS

                # Relative exposure weights of sources for tracks
                self._w_exposure_t = ForwardVariableDef(
                    "w_exposure_t", "simplex" + N_tot_t
                )

                # Exposure of sources for tracks
                self._eps_t = ForwardVariableDef("eps_t", "vector" + N_tot_t)

                # Expected number of events for tracks
                self._Nex_t = ForwardVariableDef("Nex_t", "real")

                # Sampled number of events for tracks
                self._N_t = ForwardVariableDef("N_t", "int")

            if "cascades" in self._event_types:

                # Define cascade type as in package
                self._cascade_type = ForwardVariableDef("cascade_type", "int")
                self._cascade_type << CASCADES

                # Relative exposure weights of sources for cascades
                self._w_exposure_c = ForwardVariableDef(
                    "w_exposure_c", "simplex" + N_tot_c
                )

                # Exposure of sources for cascades
                self._eps_c = ForwardVariableDef("eps_c", "vector" + N_tot_c)

                # Expected number of events for cascades
                self._Nex_c = ForwardVariableDef("Nex_c", "real")

                # Sampled number of events for cascades
                self._N_c = ForwardVariableDef("N_c", "int")

            # Total flux
            self._Ftot = ForwardVariableDef("Ftot", "real")

            # Total flux from point sources
            self._F_src = ForwardVariableDef("Fs", "real")

            # Different assumptions for fractional flux from point sources
            # vs. other components
            self._f_arr_ = ForwardVariableDef("f_arr_", "real")
            self._f_arr_astro_ = ForwardVariableDef("f_arr_astro_", "real")
            self._f_det_ = ForwardVariableDef("f_det_", "real")
            self._f_det_astro_ = ForwardVariableDef("f_det_astro_", "real")

            # Expected number of events from different source components
            self._Nex_src_t = ForwardVariableDef("Nex_src_t", "real")
            self._Nex_src_c = ForwardVariableDef("Nex_src_c", "real")
            self._Nex_src = ForwardVariableDef("Nex_src", "real")
            self._Nex_diff_t = ForwardVariableDef("Nex_diff_t", "real")
            self._Nex_diff_c = ForwardVariableDef("Nex_diff_c", "real")
            self._Nex_diff = ForwardVariableDef("Nex_diff", "real")
            self._Nex_atmo = ForwardVariableDef("Nex_atmo", "real")

            # Sampled total number of events
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

                    # Calculate energy flux from L and D
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

                    # Convert energy flux to number flux based on spectral
                    # shape and energy bounds
                    StringExpression(
                        [
                            self._F[k],
                            "*=",
                            self._flux_conv(
                                src_index_ref,
                                self._Esrc_min / (1 + self._z[k]),
                                self._Esrc_max / (1 + self._z[k])
                            ),
                        ]
                    )

                    # Sum point source flux
                    StringExpression([self._F_src, " += ", self._F[k]])

            if self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_diff

            if self.sources.atmospheric and not self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_atmo

            if self.sources.atmospheric and self.sources.diffuse:
                StringExpression("F[Ns+2]") << self._F_atmo

            # For each source, calculate the exposure for different
            # event types
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

            # Calculate the exposure for diffuse/atmospheric sources
            # For cascades, we assume no atmo component
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

                    # no interpolation needed for atmo as spectral shape is fixed
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

            if self.sources.atmospheric:

                self._atmo_flux_integ_val = ForwardVariableDef(
                    "atmo_flux_integ_val", "real"
                )
                (
                    self._atmo_flux_integ_val
                    << self._sources.atmospheric_flux.total_flux_int.to(
                        1 / (u.m**2 * u.s)
                    ).value
                )

            # Get the relative exposure weights of all sources
            # This will be used to sample the labels
            # Also sample the number of events
            if "tracks" in self._event_types:

                self._Nex_t << FunctionCall([self._F, self._eps_t], "get_Nex")
                self._w_exposure_t << FunctionCall(
                    [self._F, self._eps_t], "get_exposure_weights"
                )
                # WORKAROUND, weird bug in Stan poisson_rng
                # self._N_t << StringExpression(["poisson_rng(", self._Nex_t, ")"])
                self._N_t << self._N_poisson_t

            if "cascades" in self._event_types:

                self._Nex_c << FunctionCall([self._F, self._eps_c], "get_Nex")
                self._w_exposure_c << FunctionCall(
                    [self._F, self._eps_c], "get_exposure_weights"
                )
                # WORKAROUND, weird bug in Stan poisson_rng
                # self._N_c << StringExpression(["poisson_rng(", self._Nex_c, ")"])
                self._N_c << self._N_poisson_c

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

            # Calculate the fractional association for different assumptions
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

    def _generated_quantities(self):
        """
        Write the generated quantities section of the Stan file.
        """

        with GeneratedQuantitiesContext():

            self._dm_rng = OrderedDict()

            # For different event types, define the detector model in both RNG and PDF
            # mode to have all functions included.
            # This will add to the functions section of the Stan file automatically.
            for event_type in self._event_types:
                if (
                    self.detector_model_type == R2021DetectorModel
                    and event_type == TRACKS
                ):
                    self._dm_rng[event_type] = self.detector_model_type(
                        mode=DistributionMode.RNG,
                        event_type=event_type,
                        gen_type="histogram",
                        rewrite=False,
                    )

                else:
                    self._dm_rng[event_type] = self.detector_model_type(
                        mode=DistributionMode.RNG, event_type=event_type
                    )

            # We redefine a bunch of variables from transformed data here, as we would
            # like to access them as outputs from the Stan simulation.
            self._f_arr = ForwardVariableDef("f_arr", "real")
            self._f_arr_astro = ForwardVariableDef("f_arr_astro", "real")
            self._f_det = ForwardVariableDef("f_det", "real")
            self._f_det_astro = ForwardVariableDef("f_det_astro", "real")
            self._f_arr << self._f_arr_
            self._f_arr_astro << self._f_arr_astro_
            self._f_det << self._f_det_
            self._f_det_astro << self._f_det_astro_

            self._N_str = ["[", self._N, "]"]

            # Lambda is the label of each event, matching to its true source
            self._lam = ForwardArrayDef("Lambda", "int", self._N_str)

            # omega is the true direction
            self._omega = ForwardVariableDef("omega", "unit_vector[3]")

            # Energies at the source, Earth and reconstructed in the detector
            self._Esrc = ForwardVariableDef("Esrc", "vector[N]")
            self._E = ForwardVariableDef("E", "vector[N]")
            self._Edet = ForwardVariableDef("Edet", "vector[N]")
            self._Esrc_min_arr = ForwardVariableDef("Esrc_min_arr", "real")
            self._Esrc_max_arr = ForwardVariableDef("Esrc_max_arr", "real")
            self._rs_bbpl_Eth_tmp = ForwardVariableDef("rs_bbpl_Eth_tmp", "real")

            # The cos(zenith) corresponding to each omega and assuming South Pole detector
            self._cosz = ForwardArrayDef("cosz", "real", self._N_str)

            # Detected directions as unit vectors for each event
            self._event = ForwardArrayDef("event", "unit_vector[3]", self._N_str)
            self._pre_event = ForwardVectorDef("pre_event", [4])

            # Variables for rejection sampling
            # Rejection sampling is based on a broken power law envelope,
            # and then the true source spectrum x effective area is sampled.
            # The tail of the envelope must not be steeper than that of the true
            # distribution. See e.g. https://bookdown.org/rdpeng/advstatcomp/rejection-sampling.html
            self._accept = ForwardVariableDef("accept", "int")
            self._detected = ForwardVariableDef("detected", "int")
            self._ntrials = ForwardVariableDef("ntrials", "int")
            self._u_samp = ForwardVariableDef("u_samp", "real")
            self._aeff_factor = ForwardVariableDef("aeff_factor", "real")
            self._src_factor = ForwardVariableDef("src_factor", "real")
            self._f_value = ForwardVariableDef("f_value", "real")
            self._g_value = ForwardVariableDef("g_value", "real")
            self._c_value = ForwardVariableDef("c_value", "real")
            self._idx_cosz = ForwardVariableDef("idx_cosz", "int")
            self._gamma2 = ForwardVariableDef("gamma2", "real")

            if "tracks" in self._event_types:
                Nex_t_sim = ForwardVariableDef("Nex_t_sim", "real")
                Nex_t_sim << self._Nex_t

            if "cascades" in self._event_types:
                Nex_c_sim = ForwardVariableDef("Nex_c_sim", "real")
                Nex_c_sim << self._Nex_c

            self._event_type = ForwardVariableDef("event_type", "vector[N]")

            # Kappa is the shape param of the vMF used in sampling the detected direction
            self._kappa = ForwardVariableDef("kappa", "vector[N]")

            # Here we start the sampling, first fo tracks and then cascades.
            if "tracks" in self._event_types:

                # For each event, we rejection sample the true energy and direction
                # and then directly sample the detected properties
                with ForLoopContext(1, self._N_t, "i") as i:

                    self._event_type[i] << self._track_type

                    # Sample source label
                    self._lam[i] << FunctionCall(
                        [self._w_exposure_t], "categorical_rng"
                    )

                    # Reset rejection
                    self._accept << 0
                    self._detected << 0
                    self._ntrials << 0

                    # While not accepted
                    with WhileLoopContext([StringExpression([self._accept != 1])]):

                        # Used for rejection sampling
                        self._u_samp << FunctionCall([0.0, 1.0], "uniform_rng")

                        # For point sources, the true direction is specified
                        with IfBlockContext(
                            [StringExpression([self._lam[i], " <= ", self._Ns])]
                        ):

                            self._omega << self._varpi[self._lam[i]]

                        # Otherwise, sample uniformly over sphere, considering v_lim
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

                        # Calculate the envelope for rejection sampling and the shape of
                        # the source spectrum for the various source components
                        if self.sources.point_source:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " <= ", self._Ns])]
                            ):

                                if self._shared_src_index:
                                    src_index_ref = self._src_index
                                else:
                                    src_index_ref = self._src_index[self._lam[i]]

                                # The shape of the envelope to use depends on the
                                # source spectrum. This is to make things more efficient.
                                (
                                    self._gamma2
                                    << self._rs_bbpl_gamma2_scale_t - src_index_ref
                                )

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl
                                self._Esrc_min_arr << self._Esrc_min / (
                                    1 + self._z[self._lam[i]]
                                )
                                self._Esrc_max_arr << self._Esrc_max / (
                                    1 + self._z[self._lam[i]]
                                )

                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_t,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_t,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                "<=",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._rs_bbpl_gamma1_t,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._rs_bbpl_gamma1_t,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                ">=",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                # Store the value of the source PDF at this energy
                                self._src_factor << self._src_spectrum_lpdf(
                                    self._E[i],
                                    src_index_ref,
                                    self._Esrc_min / (1 + self._z[self._lam[i]]),
                                    self._Esrc_max / (1 + self._z[self._lam[i]]),
                                )

                                # It is log, to take the exp()
                                self._src_factor << FunctionCall(
                                    [self._src_factor], "exp"
                                )

                                # Account for energy redshift losses
                                self._Esrc[i] << self._E[i] * (
                                    1 + self._z[self._lam[i]]
                                )

                        # Treat the atmospheric and diffuse components similarly
                        if self.sources.atmospheric and not self.sources.diffuse:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                # Assume fixed index of ~3.6 for atmo to get reasonable
                                # envelope function
                                self._gamma2 << self._rs_bbpl_gamma2_scale_t - 3.6

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl
                                self._Esrc_min_arr << self._Emin
                                self._Esrc_max_arr << self._Emax
                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_t,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_t,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                "<=",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._rs_bbpl_gamma1_t,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._rs_bbpl_gamma1_t,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                ">=",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                (
                                    self._src_factor
                                    << self._atmo_flux(self._E[i], self._omega)
                                    / self._atmo_flux_integ_val
                                )  # Normalise
                                self._Esrc[i] << self._E[i]

                        elif self.sources.diffuse:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):

                                (
                                    self._gamma2
                                    << self._rs_bbpl_gamma2_scale_t - self._diff_index
                                )

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl

                                self._Esrc_min_arr << self._Ediff_min / (
                                    1 + self._z[self._lam[i]]
                                )
                                self._Esrc_max_arr << self._Ediff_max / (
                                    1 + self._z[self._lam[i]]
                                )

                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_t,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_t,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                "<=",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._rs_bbpl_gamma1_t,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._rs_bbpl_gamma1_t,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                ">=",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                self._src_factor << self._diff_spectrum_lpdf(
                                    self._E[i],
                                    self._diff_index,
                                    self._Ediff_min / (1 + self._z[self._lam[i]]),
                                    self._Ediff_max / (1 + self._z[self._lam[i]]),
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

                                self._gamma2 << self._rs_bbpl_gamma2_scale_t - 3.6

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl
                                self._Esrc_min_arr << self._Emin
                                self._Esrc_max_arr << self._Emax

                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_t,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_t,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                "<=",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._rs_bbpl_gamma1_t,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_t,
                                            self._rs_bbpl_gamma1_t,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                ">=",
                                                self._rs_bbpl_Eth_t,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_t,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                (
                                    self._src_factor
                                    << self._atmo_flux(self._E[i], self._omega)
                                    / self._atmo_flux_integ_val
                                )  # Normalise
                                self._Esrc[i] << self._E[i]

                        self._aeff_factor << self._dm_rng["tracks"].effective_area(
                            self._E[i], self._omega
                        )

                        if (
                            self.detector_model_type == NorthernTracksDetectorModel
                            or self.detector_model_type == IceCubeDetectorModel
                        ):

                            with IfBlockContext(
                                [StringExpression([self._cosz[i], ">= 0.1"])]
                            ):

                                self._aeff_factor << 0

                        # Calculate quantities for rejection sampling
                        # Value of the distribution that we want to sample from
                        self._f_value << self._src_factor * self._aeff_factor

                        # Find the precomputed c_value for this source and cosz
                        #self._idx_cosz << FunctionCall(
                        #    [self._cosz[i], self._rs_cosz_bin_edges_t], "binary_search"
                        #)
                        self._c_value << self._rs_cvals_t[self._lam[i]]

                        # Debugging when sampling gets stuck
                        StringExpression([self._ntrials, " += ", 1])

                        with IfBlockContext(
                            [StringExpression([self._ntrials, "< 1000000"])]
                        ):

                            # Here is the rejection sampling criterion
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

                                if self.detector_model_type == R2021DetectorModel:
                                    # return of energy resolution is log_10(E/GeV)

                                    self._Edet[i] << 10 ** self._dm_rng[
                                        "tracks"
                                    ].energy_resolution(
                                        FunctionCall([self._E[i]], "log10"), self._omega
                                    )

                                else:
                                    self._Edet[i] << 10 ** self._dm_rng[
                                        "tracks"
                                    ].energy_resolution(self._E[i])

                                # Detection effects
                                if self.detector_model_type == R2021DetectorModel:
                                    # both energies are E/GeV, angular_resolution does log internally

                                    self._pre_event << self._dm_rng[
                                        "tracks"
                                    ].angular_resolution(
                                        FunctionCall([self._E[i]], "log10"),
                                        FunctionCall([self._Edet[i]], "log10"),
                                        self._omega,
                                    )
                                    self._event[i] << StringExpression(
                                        ["pre_event[1:3]"]
                                    )
                                    self._kappa[i] << StringExpression(["pre_event[4]"])

                                else:
                                    self._event[i] << self._dm_rng[
                                        "tracks"
                                    ].angular_resolution(self._E[i], self._omega)
                                    (
                                        self._kappa[i]
                                        << self._dm_rng[
                                            "tracks"
                                        ].angular_resolution.kappa()
                                    )

                            with ElseBlockContext():

                                self._detected << 0

                            # Also apply the threshold on possible detected energies
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

                                # Accept this sample!
                                self._accept << 1

                        # Debugging
                        with ElseBlockContext():

                            # If sampler gets stuck, print a warning message and move on.
                            self._accept << 1

                            StringExpression(
                                ['print("problem component: ", ', self._lam[i], ");\n"]
                            )

            # Repeat as above for cascades! For detailed comments see tracks, approach is identical.
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

                        # Energy spectrum
                        if self.sources.point_source:

                            with IfBlockContext(
                                [StringExpression([self._lam[i], " <= ", self._Ns])]
                            ):

                                if self._shared_src_index:
                                    src_index_ref = self._src_index
                                else:
                                    src_index_ref = self._src_index[self._lam[i]]

                                (
                                    self._gamma2
                                    << self._rs_bbpl_gamma2_scale_c - src_index_ref
                                )

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl
                                self._Esrc_min_arr << self._Esrc_min / (
                                    1 + self._z[self._lam[i]]
                                )
                                self._Esrc_max_arr << self._Esrc_max / (
                                    1 + self._z[self._lam[i]]
                                )

                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_c,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_c,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_c,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_c,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_c,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_c,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_c,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                "<=",
                                                self._rs_bbpl_Eth_c,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_c,
                                            self._rs_bbpl_gamma1_c,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_c,
                                            self._rs_bbpl_gamma1_c,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                ">=",
                                                self._rs_bbpl_Eth_c,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_c,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

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

                                (
                                    self._gamma2
                                    << self._rs_bbpl_gamma2_scale_c - self._diff_index
                                )

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl
                                """
                                self._Esrc_min_arr << self._Esrc_min / (
                                    1 + self._z[self._lam[i]]
                                )
                                self._Esrc_max_arr << self._Esrc_max / (
                                    1 + self._z[self._lam[i]]
                                )
                                """
                                self._Esrc_min_arr << self._Emin
                                self._Esrc_max_arr << self._Emax

                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_c,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_c,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_c,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_c,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_c,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_c,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                "<",
                                                self._rs_bbpl_Eth_c,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                "<=",
                                                self._rs_bbpl_Eth_c,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_c,
                                            self._rs_bbpl_gamma1_c,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._rs_bbpl_gamma1_c,
                                            self._rs_bbpl_gamma1_c,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Esrc_min_arr,
                                                ">=",
                                                self._rs_bbpl_Eth_c,
                                                ") && (",
                                                self._Esrc_max_arr,
                                                ">",
                                                self._rs_bbpl_Eth_c,
                                                ")",
                                            ]
                                        )
                                    ]
                                ):

                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Esrc_min_arr
                                        + (self._Esrc_max_arr - self._Esrc_min_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Esrc_min_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Esrc_max_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                self._src_factor << self._diff_spectrum_lpdf(
                                    self._E[i],
                                    self._diff_index,
                                    #self._Esrc_min / (1 + self._z[self._lam[i]]),
                                    #self._Esrc_max / (1 + self._z[self._lam[i]]),
                                    self._Esrc_min_arr,
                                    self._Esrc_max_arr,
                                )
                                self._src_factor << FunctionCall(
                                    [self._src_factor], "exp"
                                )

                                self._Esrc[i] << self._E[i] * (
                                    1 + self._z[self._lam[i]]
                                )

                        self._aeff_factor << self._dm_rng["cascades"].effective_area(
                            self._E[i], self._omega
                        )

                        self._f_value = self._src_factor * self._aeff_factor

                        #self._idx_cosz << FunctionCall(
                        #    [self._cosz[i], self._rs_cosz_bin_edges_c], "binary_search"
                        #)
                        self._c_value << self._rs_cvals_c[self._lam[i]]

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
                                # effective area veto for possibly empty true_energy bins overcome,
                                # Detection effects
                                # Calculate quantities for rejection sampling
                                self._Edet[i] << 10 ** self._dm_rng[
                                    "cascades"
                                ].energy_resolution(self._E[i])

                                self._event[i] << self._dm_rng[
                                    "cascades"
                                ].angular_resolution(self._E[i], self._omega)
                                (
                                    self._kappa[i]
                                    << self._dm_rng[
                                        "cascades"
                                    ].angular_resolution.kappa()
                                )

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
