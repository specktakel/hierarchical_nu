from typing import Sequence, Tuple, Iterable, List
import os
from itertools import product

import numpy as np
from scipy import stats
from astropy import units as u
import matplotlib.pyplot as plt

from abc import ABC

from hierarchical_nu.utils.roi import ROI, ROIList, CircularROI
from hierarchical_nu.stan.interface import STAN_GEN_PATH
from hierarchical_nu.backend.stan_generator import (
    ElseBlockContext,
    IfBlockContext,
    StanGenerator,
)
from hierarchical_nu.stan.interface import STAN_PATH
from ..utils.cache import Cache
from ..utils.fitting_tools import Residuals
from ..backend import (
    VMFParameterization,
    TruncatedParameterization,
    SimpleHistogram,
    ReturnStatement,
    FunctionCall,
    DistributionMode,
    LognormalMixture,
    ForLoopContext,
    WhileLoopContext,
    ForwardVariableDef,
    ForwardArrayDef,
    ForwardVectorDef,
    StanArray,
    StringExpression,
    UserDefinedFunction,
    TwoDimHistInterpolation,
)
from .detector_model import (
    EffectiveArea,
    EnergyResolution,
    AngularResolution,
    DetectorModel,
)

from ..source.source import Sources

from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.point_source_likelihood.energy_likelihood import (
    MarginalisedIntegratedEnergyLikelihood,
)

import logging

"""
Implements the 10 year muon track point source data set of IceCube.
Makes use of existing `icecube_tools` package.
Classes implement organisation of data and stan code generation.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
Cache.set_cache_dir(".cache")


class HistogramSampler:
    """
    Class to create histograms in stan-readable format.
    """

    def __init__(self, rewrite: bool) -> None:
        pass

    def _generate_ragged_ereco_data(self, irf: R2021IRF) -> None:
        """
        Generates ragged arrays for energy resolution.
        :param irf: Instance of R2021IRF from `icecube_tools`
        """

        logger.debug("Creating ragged arrays for reco energy.")
        # Create empty lists
        num_of_bins = []
        num_of_values = []
        cum_num_of_values = []
        cum_num_of_bins = []
        bins = []
        values = []
        # Iterate over etrue and declination bins of IRF
        for c_e, etrue in enumerate(irf.true_energy_values):
            for c_d, dec in enumerate(irf.declination_bins[:-1]):
                if not (c_e, c_d) in irf.faulty:
                    # Get bins and values of ereco distribution
                    b = irf.reco_energy_bins[c_e, c_d]
                    n = irf.reco_energy[c_e, c_d].pdf(
                        irf.reco_energy_bins[c_e, c_d][:-1] + 0.01
                    )
                else:
                    logger.warning(f"Empty true energy bin: {c_e, c_d}")
                    b = np.array([])
                    n = np.array([])
                # Append to lists
                bins.append(b)
                values.append(n)
                num_of_values.append(n.size)
                num_of_bins.append(b.size)
                # Cumulative number needs previous number
                try:
                    cum_num_of_values.append(cum_num_of_values[-1] + n.size)
                    cum_num_of_bins.append(cum_num_of_bins[-1] + b.size)
                # On first iteration no previous number exists
                except IndexError:
                    cum_num_of_values.append(n.size)
                    cum_num_of_bins.append(b.size)

        # Make attributes
        self._ereco_cum_num_vals = cum_num_of_values
        self._ereco_cum_num_edges = cum_num_of_bins
        self._ereco_num_vals = num_of_values
        self._ereco_num_edges = num_of_bins
        self._ereco_hist = np.concatenate(values)
        self._ereco_edges = np.concatenate(bins)
        self._tE_bin_edges = np.power(10, irf.true_energy_bins)

    def _generate_ragged_psf_data(self, irf: R2021IRF):
        """
        Generates ragged arrays for angular resolution.
        :param irf: Instance of R2021IRF from `icecube_tools`
        """

        logger.debug("Creating ragged arrays for angular parts.")

        # Create empty lists
        psf_vals = []
        psf_edges = []
        psf_num_vals = []
        psf_num_edges = []
        psf_cum_num_vals = []
        psf_cum_num_edges = []
        ang_num_vals = []
        ang_num_edges = []
        ang_cum_num_vals = []
        ang_cum_num_edges = []
        ang_vals = []
        ang_edges = []

        # Iterate over true energy and declination bins of the IRF
        for etrue, _ in enumerate(irf.true_energy_values):
            for dec, _ in enumerate(irf.declination_bins[:-1]):
                # Get ereco bins
                if (etrue, dec) in irf.faulty:
                    logger.warning(f"Empty true energy bin: {etrue, dec}")
                    n_reco = np.zeros(20)
                else:
                    n_reco, bins_reco = irf._marginalisation(etrue, dec)
                for c, v in enumerate(n_reco):
                    # If counts in bin is nonzero, do further stuff
                    if v != 0.0:
                        # get psf distribution
                        n_psf, bins_psf = irf._marginalize_over_angerr(etrue, dec, c)
                        n = n_psf.copy()
                        bins = bins_psf.copy()
                        # Append bins, values, etc. to lists
                        psf_vals.append(n)
                        psf_edges.append(bins)
                        psf_num_vals.append(n.size)
                        psf_num_edges.append(bins.size)
                        # Cumulative numbers, try if previous number exists
                        try:
                            psf_cum_num_vals.append(psf_cum_num_vals[-1] + n.size)
                        # If not (i.e. first iteration of loop):
                        except IndexError:
                            psf_cum_num_vals.append(n.size)
                        try:
                            psf_cum_num_edges.append(psf_cum_num_edges[-1] + bins.size)
                        except IndexError:
                            psf_cum_num_edges.append(bins.size)

                        # do it again for ang_err
                        for c_psf, v_psf in enumerate(n_psf):
                            if v_psf != 0.0:
                                n_ang, bins_ang = irf._get_angerr_dist(
                                    etrue, dec, c, c_psf
                                )
                                n = n_ang.copy()
                                bins = bins_ang.copy()
                                ang_vals.append(n)
                                ang_edges.append(bins)
                                ang_num_vals.append(n.size)
                                ang_num_edges.append(bins.size)
                            else:
                                ang_num_vals.append(0)
                                ang_num_edges.append(0)
                    # If counts in ereco bin is zero:
                    else:
                        psf_num_vals.append(0)
                        psf_num_edges.append(0)
                        # There are 20 PSF bins for each ereco bin,
                        # for each would exist one ang_err distribution
                        for _ in range(20):
                            ang_num_vals.append(0)
                            ang_num_edges.append(0)
                        try:
                            psf_cum_num_vals.append(psf_cum_num_vals[-1])
                        except IndexError:
                            psf_cum_num_vals.append(0)
                        try:
                            psf_cum_num_edges.append(psf_cum_num_edges[-1])
                        except IndexError:
                            psf_cum_num_edges.append(0)

        # Create cumulative numbers for ang_err outside of main loop
        # because these might have different number of bins
        for v in ang_num_vals:
            try:
                ang_cum_num_vals.append(ang_cum_num_vals[-1] + v)
            except IndexError:
                ang_cum_num_vals.append(v)
        for v in ang_num_edges:
            try:
                ang_cum_num_edges.append(ang_cum_num_edges[-1] + v)
            except IndexError:
                ang_cum_num_edges.append(v)

        # Make attributes
        self._psf_cum_num_edges = psf_cum_num_edges
        self._psf_cum_num_vals = psf_cum_num_vals
        self._psf_num_vals = psf_num_vals
        self._psf_num_edges = psf_num_edges
        self._psf_hist = np.concatenate(psf_vals)
        self._psf_edges = np.concatenate(psf_edges)

        self._ang_edges = np.concatenate(ang_edges)
        self._ang_hist = np.concatenate(ang_vals)
        self._ang_num_vals = ang_num_vals
        self._ang_num_edges = ang_num_edges
        self._ang_cum_num_vals = ang_cum_num_vals
        self._ang_cum_num_edges = ang_cum_num_edges

    def _make_hist_lookup_functions(self, season: str) -> None:
        """
        Creates stan code for lookup functions of true energy and declination.
        True energy should be in log10(E/GeV), declination in rad.
        """

        logger.debug("Making etrue/dec lookup functions.")
        self._etrue_lookup = UserDefinedFunction(
            f"{season}_etrue_lookup", ["true_energy"], ["real"], "int"
        )
        with self._etrue_lookup:
            etrue_bins = StanArray(
                "log_etrue_bins", "real", np.log10(self._tE_bin_edges)
            )
            ReturnStatement(["binary_search(true_energy, ", etrue_bins, ")"])

        self._dec_lookup = UserDefinedFunction(
            f"{season}_dec_lookup", ["declination"], ["real"], "int"
        )
        with self._dec_lookup:
            # do binary search for bin of declination
            declination_bins = StanArray("dec_bins", "real", self._declination_bins)
            ReturnStatement(["binary_search(declination, ", declination_bins, ")"])

    def _make_histogram(
        self,
        data_type: str,
        hist_values: Iterable[float],
        hist_bins: Iterable[float],
        season: str,
    ) -> None:
        """
        Creates stan code for ragged arrays used in histograms.
        :param hist_values: Array of all histogram values, can be made up of multiple histograms
        :param hist_bins: Array of all histogram bin edges, can be made up of multiple histograms
        """

        logger.debug("Making histograms.")
        self._ragged_hist = UserDefinedFunction(
            f"{season}_{data_type}_get_ragged_hist", ["idx"], ["int"], "array[] real"
        )
        with self._ragged_hist:
            arr = StanArray("arr", "real", hist_values)
            self._make_ragged_start_stop(data_type, "vals", season)
            ReturnStatement(["arr[start:stop]"])

        self._ragged_edges = UserDefinedFunction(
            f"{season}_{data_type}_get_ragged_edges", ["idx"], ["int"], "array[] real"
        )
        with self._ragged_edges:
            arr = StanArray("arr", "real", hist_bins)
            self._make_ragged_start_stop(data_type, "edges", season)
            ReturnStatement(["arr[start:stop]"])

    def _make_ereco_hist_index(self, season: str) -> None:
        """
        Creates stan code for lookup function for ereco hist index.
        Index is used to lookup which part of ragged array in histogram function is needed.
        There are 14 true energy bins and 3 declination bins.
        """

        logger.debug("Making ereco histogram indexing function.")
        get_ragged_index = UserDefinedFunction(
            f"{season}_ereco_get_ragged_index", ["etrue", "dec"], ["int", "int"], "int"
        )
        # Takes indices of etrue and dec (to be determined elsewhere!)
        with get_ragged_index:
            ReturnStatement(["dec + (etrue - 1) * 3"])

    def _make_psf_hist_index(self, season: str) -> None:
        """
        Creates stan code for lookup function for ereco hist index.
        Index is used to lookup which part of ragged array in histogram function is needed.
        There are 14 true energy bins, 3 declination bins and 20 reco energy bins.
        """

        logger.debug("Making psf histogram indexing function.")
        get_ragged_index = UserDefinedFunction(
            f"{season}_psf_get_ragged_index",
            ["etrue", "dec", "ereco"],
            ["int", "int", "int"],
            "int",
        )
        with get_ragged_index:
            ReturnStatement(["ereco + (dec - 1) * 20 + (etrue - 1) * 3  * 20"])

    def _make_ang_hist_index(self, season: str) -> None:
        """
        Creates stan code for lookup function for ereco hist index.
        Index is used to lookup which part of ragged array in histogram function is needed.
        There are 14 true energy bins, 3 declination bins, 20 reco energy bins and 20 PSF bins.
        """

        logger.debug("Making ang histogram indexing function.")
        get_ragged_index = UserDefinedFunction(
            f"{season}_ang_get_ragged_index",
            ["etrue", "dec", "ereco", "psf"],
            ["int", "int", "int", "int"],
            "int",
        )
        with get_ragged_index:
            ReturnStatement(
                [
                    "psf + (ereco - 1) * 20 + (dec - 1) * 20 * 20 + (etrue - 1) * 3 * 20 * 20"
                ]
            )

    def _make_lookup_functions(self, name: str, array: Iterable, season: str) -> None:
        """
        Creates stan code for lookup functions, i.e. wraps function around array indexing.
        :param name: Name of function
        :param array: Array containing data
        """

        logger.debug("Making generic lookup functions.")
        f = UserDefinedFunction(f"{season}_{name}", ["idx"], ["int"], "int")
        with f:
            arr = StanArray("arr", "int", array)
            with IfBlockContext(["idx > ", len(array), "|| idx < 0"]):
                FunctionCall(['"idx outside range, "', "idx"], "reject")
            ReturnStatement(["arr[idx]"])

    def _make_ragged_start_stop(self, data: str, hist: str, season: str) -> None:
        """
        Creates stan code to find start and end of a histogram in a ragged array structure.
        :param data: str, "ereco", "psf", "ang"
        :param hist: Type of hist, "vals" or "edges"
        """

        logger.debug("Making ragged array indexing.")
        start = ForwardVariableDef("start", "int")
        stop = ForwardVariableDef("stop", "int")
        if hist == "edges" or hist == "vals":
            start << StringExpression(
                [
                    "{}_{}_get_cum_num_{}(idx)-{}_{}_get_num_{}(idx)+1".format(
                        season, data, hist, season, data, hist
                    )
                ]
            )
            stop << StringExpression(
                ["{}_{}_get_cum_num_{}(idx)".format(season, data, hist)]
            )
        else:
            raise ValueError("No other type available.")


class R2021EffectiveArea(EffectiveArea):
    """
    Effective area for the ten-year All Sky Point Source release:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    More or less copied from NorthernTracks implementation.
    """

    def __init__(
        self, mode: DistributionMode = DistributionMode.PDF, season: str = "IC86_II"
    ) -> None:
        self._season = season
        self._func_name = f"{season}EffectiveArea"

        self.CACHE_FNAME = f"aeff_{season}.npz"

        self.mode = mode

        self.setup()

        self._make_spline()

    def generate_code(self):
        super().__init__(
            self._func_name,
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real",
        )

        # Define Stan interface.
        if self.mode == DistributionMode.PDF:
            type_ = TwoDimHistInterpolation
        else:
            type_ = SimpleHistogram

        # Check if ROI should be applied to the effective area
        # This will speed up the fit but requires recompilation for different ROIs
        if ROIList.STACK:
            apply_roi = np.all(_.apply_roi for _ in ROIList.STACK)
        else:
            apply_roi = False

        if apply_roi:
            cosz_min = -np.sin(ROIList.DEC_max())
            cosz_max = -np.sin(ROIList.DEC_min())
            idx_min = np.digitize(cosz_min, self._cosz_bin_edges) - 1
            idx_max = np.digitize(cosz_max, self._cosz_bin_edges, right=True) - 1
            eff_area = self._eff_area[:, idx_min : idx_max + 1]
            cosz_bin_edges = self._cosz_bin_edges[idx_min : idx_max + 2]
        else:
            cosz_bin_edges = self._cosz_bin_edges
            eff_area = self._eff_area
        # Define Stan interface.
        if self.mode == DistributionMode.PDF:
            type_ = TwoDimHistInterpolation
        else:
            type_ = SimpleHistogram

        with self:
            hist = type_(
                eff_area,
                [self._tE_bin_edges, cosz_bin_edges],
                f"{self._season}EffAreaHist",
            )
            # Uses cos(z), so calculate z = pi - theta
            cos_dir = "cos(pi() - acos(true_dir[3]))"

            _ = ReturnStatement([hist("true_energy", cos_dir)])

    def setup(self) -> None:
        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                data = np.load(fr, allow_pickle=True)
                eff_area = data["eff_area"]
                tE_bin_edges = data["tE_bin_edges"]
                cosz_bin_edges = data["cosz_bin_edges"]
        else:
            from icecube_tools.detector.effective_area import EffectiveArea

            # cut the arrays short because of numerical issues in precomputation.py
            aeff = EffectiveArea.from_dataset("20210126", self._season)
            # 1st axis: energy, 2nd axis: cosz
            eff_area = aeff.values
            # Deleting zero-entries above 1e9GeV is done in icecube_tools
            tE_bin_edges = aeff.true_energy_bins
            cosz_bin_edges = aeff.cos_zenith_bins

            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                np.savez(
                    fr,
                    eff_area=eff_area,
                    tE_bin_edges=tE_bin_edges,
                    cosz_bin_edges=cosz_bin_edges,
                )

        self._eff_area = eff_area
        self._tE_bin_edges = tE_bin_edges
        self._cosz_bin_edges = cosz_bin_edges

        self._rs_bbpl_params = {}
        self._rs_bbpl_params["threshold_energy"] = 5e4  # GeV
        self._rs_bbpl_params["gamma1"] = -0.8
        self._rs_bbpl_params["gamma2_scale"] = 0.6


class R2021EnergyResolution(EnergyResolution, HistogramSampler):

    """
    Energy resolution for the ten-year All Sky Point Source release:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    """

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
        season: str = "IC86_II",
    ) -> None:
        """
        Instantiate class.
        :param mode: DistributionMode.PDF or .RNG (fitting or simulating)
        :parm rewrite: bool, True if cached files should be overwritten,
                       if there are no cached files they will be generated either way
        :param make_plots: bool, true if plots of parameterisation in case of lognorm should be made
        :param n_components: int, specifies how many components the lognormal mixture should have
        :param ereco_cuts: bool, if True simulated events below Ereco of the data in the sampled Aeff dec bin are discarded
        :param season: String indicating the detector season
        """

        self._season = season
        self.CACHE_FNAME_LOGNORM = f"energy_reso_lognorm_{season}.npz"
        self.CACHE_FNAME_HISTOGRAM = f"energy_reso_histogram_{season}.npz"
        self.irf = R2021IRF.from_period(season)
        self._icecube_tools_eres = MarginalisedIntegratedEnergyLikelihood(
            season, np.linspace(1, 9, 25)
        )
        self._make_ereco_cuts = ereco_cuts
        self._ereco_cuts = self._icecube_tools_eres._ereco_limits
        self._aeff_dec_bins = self._icecube_tools_eres.declination_bins_aeff
        self.mode = mode
        if self.mode == DistributionMode.PDF:
            self._func_name = f"{season}EnergyResolution"
            self.gen_type = "lognorm"
        elif self.mode == DistributionMode.RNG:
            self._func_name = f"{season}EnergyResolution_rng"
            self.gen_type = "histogram"
        self.mode = mode
        self._rewrite = rewrite
        logger.info("Forced energy rewriting: {}".format(rewrite))
        self.make_plots = make_plots
        self._poly_params_mu: Sequence = []
        self._poly_params_sd: Sequence = []
        self._poly_limits: Sequence = []
        self._poly_limits_battery: Sequence = []
        self._declination_bins = self.irf.declination_bins
        # Find faulty bins (usually at low energy and in the Southern sky)
        # and find pdet_limits according to this
        # For prob_Edet_above_threshold
        self._pdet_limits = (1e2, 1e8)

        self._n_components = n_components
        self.setup()

    def generate_code(self) -> None:
        """
        Generates stan code by instanciating parent class and the other things.
        """

        # initialise parent classes with proper signature for stan functions
        if self.mode == DistributionMode.PDF:
            self.mixture_name = f"{self._season}_energy_res_mix"
            super().__init__(
                self._func_name,
                ["log_true_energy", "log_reco_energy", "omega"],
                ["real", "real", "vector"],
                "real",
            )
        elif self.mode == DistributionMode.RNG:
            self.mixture_name = f"{self._season}_energy_res_mix_rng"
            super().__init__(
                self._func_name,
                ["log_true_energy", "omega"],
                ["real", "vector"],
                "real",
            )

        # Actual code generation
        # Differ between lognorm and histogram
        if self.gen_type == "lognorm":
            logger.info("Generating stan code using lognorm")
            logger.warning(
                "Further sampling of PSF and ang_err will probably fail, you have been warned. Yes, that means you!"
            )
            with self:
                # self._poly_params_mu should have shape (3, n_components, poly_deg+1)
                # 3 from declination

                mu_poly_coeffs = StanArray(
                    "EnergyResolutionMuPolyCoeffs",
                    "real",
                    self._poly_params_mu,
                )
                # same as above

                sd_poly_coeffs = StanArray(
                    "EnergyResolutionSdPolyCoeffs",
                    "real",
                    self._poly_params_sd,
                )

                poly_limits = StanArray("poly_limits", "real", self._poly_limits)

                mu = ForwardArrayDef("mu_e_res", "real", ["[", self._n_components, "]"])
                sigma = ForwardArrayDef(
                    "sigma_e_res", "real", ["[", self._n_components, "]"]
                )

                weights = ForwardVariableDef(
                    "weights", "vector[" + str(self._n_components) + "]"
                )

                # Argument `omega` is cartesian vector, cos(z) (z is direction) is theta in spherical coords
                declination = ForwardVariableDef("declination", "real")
                declination << FunctionCall(["omega"], "omega_to_dec")

                declination_bins = StanArray("dec_bins", "real", self._declination_bins)
                declination_index = ForwardVariableDef("dec_ind", "int")
                declination_index << FunctionCall(
                    [declination, declination_bins], "binary_search"
                )

                # All stan-side energies are in log10!
                lognorm = LognormalMixture(
                    self.mixture_name, self.n_components, self.mode
                )
                log_trunc_e = TruncatedParameterization(
                    "log_true_energy",
                    "log10(poly_limits[dec_ind, 1])",
                    "log10(poly_limits[dec_ind, 2])",
                )

                with ForLoopContext(1, self._n_components, "i") as i:
                    weights[i] << StringExpression(["1.0/", self._n_components])

                log_mu = FunctionCall([mu], "log")

                with ForLoopContext(1, self._n_components, "i") as i:
                    mu[i] << [
                        "eval_poly1d(",
                        log_trunc_e,
                        ", ",
                        "to_vector(",
                        mu_poly_coeffs[declination_index][i],
                        "))",
                    ]

                    sigma[i] << [
                        "eval_poly1d(",
                        log_trunc_e,
                        ", ",
                        "to_vector(",
                        sd_poly_coeffs[declination_index][i],
                        "))",
                    ]

                log_mu_vec = FunctionCall([log_mu], "to_vector")
                sigma_vec = FunctionCall([sigma], "to_vector")

                if self.mode == DistributionMode.PDF:
                    ReturnStatement(
                        [lognorm("log_reco_energy", log_mu_vec, sigma_vec, weights)]
                    )
                else:
                    ReturnStatement([lognorm(log_mu_vec, sigma_vec, weights)])

        elif self.gen_type == "histogram":
            logger.info("Generating stan code using histograms")
            with self:
                # Create necessary lists/attributes, inherited from HistogramSampler
                self._make_hist_lookup_functions(self._season)
                self._make_histogram(
                    "ereco", self._ereco_hist, self._ereco_edges, self._season
                )
                self._make_ereco_hist_index(self._season)

                for name, array in zip(
                    [
                        "ereco_get_cum_num_vals",
                        "ereco_get_cum_num_edges",
                        "ereco_get_num_vals",
                        "ereco_get_num_edges",
                    ],
                    [
                        self._ereco_cum_num_vals,
                        self._ereco_cum_num_edges,
                        self._ereco_num_vals,
                        self._ereco_num_edges,
                    ],
                ):
                    self._make_lookup_functions(name, array, self._season)

                # call histogramm with appropriate values/edges
                declination = ForwardVariableDef("declination", "real")
                declination << FunctionCall(["omega"], "omega_to_dec")
                dec_idx = ForwardVariableDef("dec_idx", "int")
                dec_idx << FunctionCall(["declination"], f"{self._season}_dec_lookup")

                ereco_hist_idx = ForwardVariableDef("ereco_hist_idx", "int")
                etrue_idx = ForwardVariableDef("etrue_idx", "int")
                etrue_idx << FunctionCall(
                    ["log_true_energy"], f"{self._season}_etrue_lookup"
                )

                if self.mode == DistributionMode.PDF:
                    with IfBlockContext(["etrue_idx == 0 || etrue_idx > 14"]):
                        ReturnStatement(["negative_infinity()"])

                ereco_hist_idx << FunctionCall(
                    [etrue_idx, dec_idx], f"{self._season}_ereco_get_ragged_index"
                )

                if self.mode == DistributionMode.PDF:
                    ereco_idx = ForwardVariableDef("ereco_idx", "int")
                    ereco_idx << FunctionCall(
                        [
                            "log_reco_energy",
                            FunctionCall(
                                [ereco_hist_idx],
                                f"{self._season}_ereco_get_ragged_edges",
                            ),
                        ],
                        "binary_search",
                    )

                    # Intercept outside of hist range here:
                    with IfBlockContext(
                        [
                            f"ereco_idx == 0 || ereco_idx > {self._season}_ereco_get_num_vals(ereco_hist_idx)"
                        ]
                    ):
                        ReturnStatement(["negative_infinity()"])

                    return_value = ForwardVariableDef("return_value", "real")
                    return_value << StringExpression(
                        [
                            FunctionCall(
                                [ereco_hist_idx],
                                f"{self._season}_ereco_get_ragged_hist",
                            ),
                            "[ereco_idx]",
                        ]
                    )

                    with IfBlockContext(["return_value == 0."]):
                        ReturnStatement(["negative_infinity()"])

                    with ElseBlockContext():
                        ReturnStatement([FunctionCall([return_value], "log")])

                else:
                    # Discard all events below lowest Ereco of data in the respective Aeff declination bin,
                    # Sample until an event passes the cut, return this Ereco
                    if self._make_ereco_cuts:
                        ereco_cuts = StanArray("ereco_cuts", "real", self._ereco_cuts)
                        aeff_dec_bins = StanArray(
                            "aeff_dec_bins", "real", self._aeff_dec_bins
                        )
                        aeff_dec_idx = ForwardVariableDef("aeff_dec_idx", "int")
                        aeff_dec_idx << FunctionCall(
                            [declination, aeff_dec_bins], "binary_search"
                        )

                    ereco = ForwardVariableDef("ereco", "real")

                    if self._make_ereco_cuts:
                        with WhileLoopContext([1]):
                            ereco << FunctionCall(
                                [
                                    FunctionCall(
                                        [ereco_hist_idx],
                                        f"{self._season}_ereco_get_ragged_hist",
                                    ),
                                    FunctionCall(
                                        [ereco_hist_idx],
                                        f"{self._season}_ereco_get_ragged_edges",
                                    ),
                                ],
                                "histogram_rng",
                            )
                            # Apply lower energy cut, Ereco_sim >= Ereco_data at the appropriate Aeff declination bin
                            # Only apply this lower limit
                            with IfBlockContext(
                                [ereco, " >= ", ereco_cuts[aeff_dec_idx, 1]]
                            ):
                                StringExpression(["break"])
                    else:
                        ereco << FunctionCall(
                            [
                                FunctionCall(
                                    [ereco_hist_idx],
                                    f"{self._season}_ereco_get_ragged_hist",
                                ),
                                FunctionCall(
                                    [ereco_hist_idx],
                                    f"{self._season}_ereco_get_ragged_edges",
                                ),
                            ],
                            "histogram_rng",
                        )
                    ReturnStatement([ereco])

    def setup(self) -> None:
        """
        Setup all data fields, load data from cached file or create from scratch.
        """

        if self.gen_type == "lognorm":
            # Create empty lists
            self._fit_params = []
            # self._eres = []
            self._rE_bin_edges = []
            self._rE_binc = []
            self._rebin_tE_binc = []
            self._tE_binc = []
            # Generate data and ragged arrays from icecube_tools IRF
            self._generate_ragged_ereco_data(self.irf)

            # Check cache
            if self.CACHE_FNAME_LOGNORM in Cache and not self._rewrite:
                logger.info("Loading energy lognorm data from file.")
                with Cache.open(self.CACHE_FNAME_LOGNORM, "rb") as fr:
                    data = np.load(fr, allow_pickle=True)
                    self._tE_bin_edges = data["tE_bin_edges"]
                    self._poly_params_mu = data["poly_params_mu"]
                    self._poly_params_sd = data["poly_params_sd"]
                    self._poly_limits = data["poly_limits"]
                    fit_params_0 = data["fit_params_0"]
                    fit_params_1 = data["fit_params_1"]
                    fit_params_2 = data["fit_params_2"]
                    tE_binc_0 = data["tE_binc_0"]
                    tE_binc_1 = data["tE_binc_0"]
                    tE_binc_2 = data["tE_binc_0"]

                self._tE_binc = [tE_binc_0, tE_binc_1, tE_binc_2]
                self._fit_params = [fit_params_0, fit_params_1, fit_params_2]

                self._poly_params_mu__ = self._poly_params_mu.copy()
                self._poly_params_sd__ = self._poly_params_sd.copy()
                self._poly_limits__ = self._poly_limits.copy()
                self._fit_params__ = self._fit_params.copy()
                self._tE_binc__ = self._tE_binc.copy()

            else:
                logger.info("Re-doing energy lognorm data and saving files.")

                self._dec_minuits = []

                for c_dec, (dec_low, dec_high) in enumerate(
                    zip(self._declination_bins[:-1], self._declination_bins[1:])
                ):
                    true_energy_bins = []
                    for c_e, tE in enumerate(self.irf.true_energy_bins):
                        if (c_e, c_dec) not in self.irf.faulty:
                            true_energy_bins.append(tE)
                        else:
                            logger.warning(f"Faulty bin at {c_e, c_dec}")

                    tE_bin_edges = np.array(true_energy_bins)
                    tE_binc = np.power(10, 0.5 * (tE_bin_edges[:-1] + tE_bin_edges[1:]))

                    # Fit lognormal mixture to pdf(reco|true) for each true energy bin
                    # do not rebin -> rebin=1
                    fit_params_temp, minuits = self._fit_energy_res(
                        tE_binc, c_dec, self._n_components
                    )
                    self._dec_minuits.append(minuits)
                    # check for label switching
                    fit_params = np.zeros_like(fit_params_temp)
                    for c, params in enumerate(fit_params_temp):
                        idx = np.argsort(params[::2])
                        fit_params[c, ::2] = params[::2][idx]
                        fit_params[c, 1::2] = params[1::2][idx]

                    self._fit_params.append(fit_params)

                    # take entire range
                    imin = 0
                    imax = -1

                    Emin = tE_binc[imin]
                    Emax = tE_binc[imax]

                    # Fit polynomial:
                    poly_params_mu, poly_params_sd, poly_limits = self._fit_polynomial(
                        fit_params, tE_binc, Emin, Emax, polydeg=6
                    )

                    self._poly_params_mu.append(poly_params_mu)
                    self._poly_params_sd.append(poly_params_sd)
                    self._poly_limits.append(poly_limits)
                    self._tE_binc.append(tE_binc)

                self._poly_limits__ = self._poly_limits.copy()
                self._poly_params_mu__ = self._poly_params_mu.copy()
                self._poly_params_sd__ = self._poly_params_sd.copy()
                self._tE_binc__ = self._tE_binc.copy()
                self._fit_params__ = self._fit_params.copy()

                # Save values
                self._tE_bin_edges = tE_bin_edges
                if self.make_plots:
                    for c, dec in enumerate(self._declination_bins[:-1]):
                        self.set_fit_params((dec + 0.01) * u.rad)

                        fig = self.plot_fit_params(self._fit_params, self._tE_binc)
                        plt.show()

                        fig = self.plot_parameterizations(
                            self._fit_params,
                            self._tE_binc,
                            c,
                        )
                        plt.show()

                self._poly_params_mu = self._poly_params_mu__.copy()
                self._poly_params_sd = self._poly_params_sd__.copy()
                self._poly_limits = self._poly_limits__.copy()
                self._fit_params = self._fit_params__.copy()
                self._tE_binc = self._tE_binc__.copy()

                # Save polynomial
                with Cache.open(self.CACHE_FNAME_LOGNORM, "wb") as fr:
                    np.savez(
                        fr,
                        tE_bin_edges=self._tE_bin_edges,
                        tE_binc_0=self._tE_binc[0],
                        tE_binc_1=self._tE_binc[1],
                        tE_binc_2=self._tE_binc[2],
                        poly_params_mu=self._poly_params_mu,
                        poly_params_sd=self._poly_params_sd,
                        poly_limits=self.poly_limits,
                        fit_params_0=self._fit_params[0],
                        fit_params_1=self._fit_params[1],
                        fit_params_2=self._fit_params[2],
                    )

        else:
            # Check cache
            if self.CACHE_FNAME_HISTOGRAM in Cache and not self._rewrite:
                logger.info("Loading energy pdf data from file.")

                with Cache.open(self.CACHE_FNAME_HISTOGRAM, "rb") as fr:
                    data = np.load(fr, allow_pickle=True)
                    self._ereco_cum_num_vals = data["cum_num_of_values"]
                    self._ereco_cum_num_edges = data["cum_num_of_bins"]
                    self._ereco_num_vals = data["num_of_values"]
                    self._ereco_num_edges = data["num_of_bins"]
                    self._ereco_hist = data["values"]
                    self._ereco_edges = data["bins"]
                    self._tE_bin_edges = np.power(10, self.irf.true_energy_bins)

            else:
                self._generate_ragged_ereco_data(self.irf)
                with Cache.open(self.CACHE_FNAME_HISTOGRAM, "wb") as fr:
                    np.savez(
                        fr,
                        bins=self._ereco_edges,
                        values=self._ereco_hist,
                        num_of_bins=self._ereco_num_edges,
                        num_of_values=self._ereco_num_vals,
                        cum_num_of_bins=self._ereco_cum_num_edges,
                        cum_num_of_values=self._ereco_cum_num_vals,
                        tE_bin_edges=self._tE_bin_edges,
                    )

            self._Emin = np.power(10, self.irf.true_energy_bins[0])
            self._Emax = np.power(10, self.irf.true_energy_bins[-1])

    @u.quantity_input
    def prob_Edet_above_threshold(
        self,
        true_energy: u.GeV,
        lower_threshold_energy: u.GeV,
        dec: u.rad,
        upper_threshold_energy=None,
        use_lognorm: bool = False,
    ):
        """
        P(Edet > Edet_min | E) for use in precomputation.
        Needs to be adapted for declination dependent cuts
        based on the detected events. Per declination,
        find lowest and highest reconstructed energy
        and restrict the threshold energy by the found values.
        Optional argument upper_threshold_energy used for debugging and diagnostic plots
        :param true_energy: True neutrino energy in GeV
        :param lower_threshold_energy: Lower reconstructed muon energy in GeV
        :param dec: Declination of event in radian
        :param upper_threshold_energy: Optional upper reconstructe muon energy in GeV,
                                       if none provided, use highest possible value
        :param use_lognorm: bool, if True use lognormal parameterisation
        """
        # Truncate input energies to safe range
        energy_trunc = true_energy.to(u.GeV).value
        energy_trunc = np.atleast_1d(energy_trunc)
        energy_trunc[energy_trunc < self._pdet_limits[0]] = self._pdet_limits[0]
        energy_trunc[energy_trunc > self._pdet_limits[1]] = self._pdet_limits[1]
        energy_trunc = energy_trunc * u.GeV
        dec = np.atleast_1d(dec)

        assert dec.shape == energy_trunc.shape

        if len(energy_trunc.shape) > 0 and len(lower_threshold_energy.shape) == 0:
            lower_threshold_energy = (
                np.full(energy_trunc.shape, lower_threshold_energy.to_value(u.GeV))
                * u.GeV
            )
        else:
            lower_threshold_energy = np.atleast_1d(lower_threshold_energy)

        assert energy_trunc.shape == lower_threshold_energy.shape

        if upper_threshold_energy is not None:
            upper_threshold_energy = np.atleast_1d(upper_threshold_energy)

        # Limits of Ereco in dec binning of effective area
        idx_dec_aeff = np.digitize(dec.to_value(u.rad), self._aeff_dec_bins) - 1
        # Get the according IRF dec bins (there are only 3)
        idx_dec_eres = np.digitize(dec.to_value(u.rad), self._declination_bins) - 1
        idx_dec_aeff[
            np.nonzero(
                (idx_dec_aeff == self._aeff_dec_bins.size - 1)
                & (np.isclose(dec.to_value(u.rad), self._aeff_dec_bins[-1]))
            )
        ] -= 1

        # Create output array
        prob = np.zeros(energy_trunc.shape)

        ## Make strongest limits on ereco_low
        # limits from exp data selection
        e_low = self._icecube_tools_eres._ereco_limits[idx_dec_aeff, 0]
        # make log of input value
        ethr_low = np.log10(lower_threshold_energy.to_value(u.GeV))
        # apply stronger limit
        e_low[ethr_low > e_low] = ethr_low[ethr_low > e_low]

        # Get the according IRF dec bins (there are only 3)
        irf_dec_idx = np.digitize(dec.to_value(u.rad), self._declination_bins) - 1

        if use_lognorm:
            for c, d in enumerate(self._declination_bins[:-1]):
                # Otherwise we will cause errors
                if c not in irf_dec_idx:
                    continue
                self.set_fit_params((d + 0.01) * u.rad)

                model = self.make_cumulative_model(self.n_components)

                model_params: List[float] = []

                for comp in range(self.n_components):
                    mu = np.poly1d(self._poly_params_mu[comp])(
                        np.log10(energy_trunc[irf_dec_idx == c].to(u.GeV).value)
                    )
                    sigma = np.poly1d(self._poly_params_sd[comp])(
                        np.log10(energy_trunc[irf_dec_idx == c].to(u.GeV).value)
                    )
                    model_params += [mu, sigma]

                # Limits are in log10(E/GeV)
                e_low = self._icecube_tools_eres._ereco_limits[
                    idx_dec_aeff[np.nonzero(irf_dec_idx == c)], 0
                ]
                ethr_low = np.log10(lower_threshold_energy.to_value(u.GeV))
                # print(ethr_low)
                e_low[ethr_low > e_low] = ethr_low
                if upper_threshold_energy is None:
                    prob[irf_dec_idx == c] = 1.0 - model(e_low, model_params)

                else:
                    prob[irf_dec_idx == c] = model(
                        np.log10(upper_threshold_energy.to_value(u.GeV)), model_params
                    ) - model(e_low, model_params)

            return prob

        else:
            idx_tE = (
                np.digitize(
                    np.log10(energy_trunc.to_value(u.GeV)), self.irf.true_energy_bins
                )
                - 1
            )

            for cE, cD in product(
                range(self.irf.true_energy_bins.size - 1),
                range(self.irf.declination_bins.size - 1),
            ):
                if cE not in idx_tE and cD not in idx_dec_eres:
                    continue

                if upper_threshold_energy is None:
                    prob[
                        (cE == idx_tE) & (cD == idx_dec_eres)
                    ] = 1.0 - self.irf.reco_energy[cE, cD].cdf(
                        e_low[(cE == idx_tE) & (cD == idx_dec_eres)]
                    )
                else:
                    pdf = self.irf.reco_energy[cE, cD]
                    prob[(cE == idx_tE) & (cD == idx_dec_eres)] = pdf.cdf(
                        np.log10(upper_threshold_energy.to_value(u.GeV))[
                            (cE == idx_tE) & (cD == idx_dec_eres)
                        ]
                    ) - pdf.cdf(e_low[(cE == idx_tE) & (cD == idx_dec_eres)])

            return prob

    @staticmethod
    def make_fit_model(n_components):
        """
        Lognormal mixture with n_components.
        """

        # s is width of lognormal
        # scale is ~expectation value
        def _model(x, *pars):
            result = 0
            for i in range(n_components):
                result += (1 / n_components) * stats.lognorm.pdf(
                    x, scale=pars[2 * i], s=pars[2 * i + 1]
                )

            return result

        return _model

    @staticmethod
    def make_cumulative_model(n_components):
        """
        Cumulative Lognormal mixture above xth with n_components.
        """

        def _cumulative_model(xth, pars):
            result = 0
            for i in range(n_components):
                result += (1 / n_components) * stats.lognorm.cdf(
                    xth, scale=pars[2 * i], s=pars[2 * i + 1]
                )
            return result

        return _cumulative_model

    @staticmethod
    def make_cumulative_fit_model(bin_edges, n_components):
        """
        Integrate within the bin edges and make a fit model out of it
        """

        def _cumulative_model(*pars):
            result = np.zeros(bin_edges.size - 1)
            for i in range(n_components):
                result += (
                    (1 / n_components)
                    * (
                        stats.lognorm.cdf(
                            bin_edges[1:], scale=pars[2 * i], s=pars[2 * i + 1]
                        )
                        - stats.lognorm.cdf(
                            bin_edges[:-1], scale=pars[2 * i], s=pars[2 * i + 1]
                        )
                    )
                    / np.diff(bin_edges)
                )
            return result

        return _cumulative_model

    @staticmethod
    def make_likelihood_function(model, counts):
        def likelihood_func(*pars):
            eval = model(*pars)
            mask = eval != 0.0
            return -np.sum(counts[mask] * np.log(eval[mask]))

        return likelihood_func

    def _fit_energy_res(
        self,
        tE_binc: np.ndarray,
        c_dec: int,
        n_components: int,
        fit_type: str = "likelihood",
    ) -> np.ndarray:
        """
        Fit a lognormal mixture to P(Ereco | Etrue) in given Etrue bins.
        A maximum likelihood approach is used. Although the data is binned in Ereco,
        it is an unbinned likelihood: $L(\theta) = \prod_i f(x_i; \theta)$
        with one data point $x_i$ per bin.
        $f(x_i; \theta)$ here is the bin-averaged lognormal mixture evaluated at $\theta$.
        Each factor of the product over the data points (or here conversely the bins $i$)
        is weighted with the histogram's value in the bin, i.e. the histogram's pdf.
        If we had the detector MC used to generate the histogram, we would have data points $x_i$
        with relative frequency of the histogram's value in each bin.
        This weighting in the above product is an exponential (more data points equals more factors),
        thus the likelihood reads $L(\theta) = \prod_i f(x_i; \theta)^{h_i}$ with the histograms height $h_i$.
        The loglike then trivially reads $\log{L} = \sum_i h_i \log{f(x_i; \theta)}$.

        fit_type = "chi2" implements the previously used least square fit.
        """

        from scipy.optimize import least_squares
        from iminuit import Minuit
        from iminuit.cost import LeastSquares

        fit_params = []
        minuits = []
        # Fitting loop
        for tE in tE_binc:
            # print(tE)
            tE_idx = np.digitize(np.log10(tE), self.irf.true_energy_bins) - 1
            # print(tE_idx)

            # Ereco bins and fractional counts per bin
            n, bins = self.irf._marginalisation(tE_idx, c_dec)
            # normalisation to pdf
            e_reso = n / np.sum(n * np.diff(bins))
            bins_c = bins[:-1] + np.diff(bins) / 2
            log10_rE_bin_edges = bins
            log10_rE_binc = bins_c

            if e_reso.sum() > 0:
                if fit_type == "likelihood":
                    # Lognormal mixture, averages the proposed mixture pdf over each of the histogram's bins.
                    model = self.make_cumulative_fit_model(
                        log10_rE_bin_edges, n_components
                    )

                    # Make the likelihood function as described above
                    llh = self.make_likelihood_function(model, e_reso)

                elif fit_type == "chi2":
                    model = self.make_fit_model(n_components)
                    # residuals = Residuals((log10_rE_binc, e_reso), model)
                    ls = LeastSquares(
                        log10_rE_binc, e_reso, np.ones_like(e_reso), model
                    )

                # Calculate seed as mean of the resolution to help minimizer
                seed_mu = np.average(log10_rE_binc, weights=e_reso)
                if ~np.isfinite(seed_mu):
                    seed_mu = 3

                seed = np.zeros(n_components * 2)
                bounds_lo: List[float] = []
                bounds_hi: List[float] = []
                names: List[str] = []
                for i in range(n_components):
                    seed[2 * i] = seed_mu + 0.1 * (i + 1)
                    seed[2 * i + 1] = (i + 1) * 0.05
                    names += [f"scale_{i}", f"s_{i}"]
                    bounds_lo += [1, 0.01]
                    bounds_hi += [8, 1]

                limits = [(l, h) for (l, h) in zip(bounds_lo, bounds_hi)]

                if fit_type == "likelihood":
                    m = Minuit(llh, *tuple(seed), name=names)
                    m.errordef = 0.5
                elif fit_type == "chi2":
                    m = Minuit(ls, *tuple(seed), name=names)
                    m.errordef = 1
                m.errors = 0.05 * np.asarray(seed)
                m.limits = limits
                m.migrad()

                if not m.fmin.is_valid:
                    # if not converged, give it one more try, seems to do the trick
                    m.migrad()

                # Check for convergence
                if not m.fmin.is_valid:
                    logger.warning(
                        f"Fit at {tE:.1f}GeV has not converged, please inspect."
                    )
                    fit_params.append(np.zeros(2 * n_components))
                else:
                    temp = []

                    for i in range(n_components):
                        temp += [m.values[f"scale_{i}"], m.values[f"s_{i}"]]

                    # Check for label switching, ascending order of scale needs to be enforced
                    # carry over possible swaps to the `s` parameter.
                    hat = np.argsort(temp[::2])
                    new_fit_pars = []
                    for i in range(n_components):
                        new_fit_pars.append(temp[::2][hat[i]])
                        new_fit_pars.append(temp[1::2][hat[i]])
                    fit_params.append(new_fit_pars)
                minuits.append(m)

            else:
                fit_params.append(np.zeros(2 * n_components))
        fit_params = np.asarray(fit_params)

        return fit_params, minuits

    def _fit_polynomial(
        self,
        fit_params: np.ndarray,
        tE_binc: np.ndarray,
        Emin: float,
        Emax: float,
        polydeg: int,
    ):
        """
        Fit polynomial to energy dependence of lognorm mixture params.
        """

        def find_nearest_idx(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        imin = find_nearest_idx(tE_binc, Emin)
        imax = find_nearest_idx(tE_binc, Emax)

        # Mask out entries where the lognormal-mixture-fit has not converged,
        # i.e. zero entriesin fit_params
        mask = np.nonzero(np.all(fit_params[imin:imax, ::2], axis=1))

        log10_tE_binc = np.log10(tE_binc)
        poly_params_mu = np.zeros((self._n_components, polydeg + 1))

        # Fit polynomial
        poly_params_sd = np.zeros_like(poly_params_mu)
        for i in range(self.n_components):
            poly_params_mu[i] = np.polyfit(
                log10_tE_binc[imin:imax][mask],
                fit_params[:, 2 * i][imin:imax][mask],
                polydeg,
            )
            poly_params_sd[i] = np.polyfit(
                log10_tE_binc[imin:imax][mask],
                fit_params[:, 2 * i + 1][imin:imax][mask],
                polydeg,
            )

            poly_limits = (Emin, Emax)

        return poly_params_mu, poly_params_sd, poly_limits

    def plot_fit_params(self, fit_params: np.ndarray, tE_binc: np.ndarray) -> None:
        """
        Plot the evolution of the lognormal parameters with true energy,
        for each mixture component.
        """

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        xs = np.linspace(*np.log10(self._poly_limits), num=100)

        if self._poly_params_mu is None:
            raise RuntimeError("Run setup() first")

        # Plot polynomial fits for each mixture component.
        for comp in range(self._n_components):
            params_mu = self._poly_params_mu[comp]
            axs[0].plot(
                xs, np.poly1d(params_mu)(xs), label="poly, mean", color=f"C{comp}"
            )
            axs[0].plot(
                np.log10(tE_binc),
                fit_params[:, 2 * comp],
                label="Mean {}".format(comp),
                color=f"C{comp}",
                ls=":",
            )

            params_sigma = self._poly_params_sd[comp]
            axs[1].plot(
                xs, np.poly1d(params_sigma)(xs), label="poly, sigma", color=f"C{comp}"
            )
            axs[1].plot(
                np.log10(tE_binc),
                fit_params[:, 2 * comp + 1],
                label="SD {}".format(comp),
                color=f"C{comp}",
                ls=":",
            )

        axs[0].set_xlabel("log10(True Energy / GeV)")
        axs[0].set_ylabel("Parameter Value")
        axs[0].legend()
        axs[1].legend()
        plt.tight_layout()
        return fig

    def plot_parameterizations(
        self,
        fit_params: np.ndarray,
        tE_binc: np.ndarray,
        c_dec: int,
    ):
        """
        Plot fitted parameterizations
        Args:
            fit_params: np.ndarray
                Fitted parameters for mu and sigma
        """

        import matplotlib.pyplot as plt

        plot_energies = np.power(10, np.arange(3.25, 7.75, step=0.5))  # GeV
        # plot_energies = [1e5, 3e5, 5e5, 8e5, 1e6, 3e6, 5e6, 8e6]  # GeV

        if self._poly_params_mu is None:
            raise RuntimeError("Run setup() first")

        plot_indices = np.digitize(plot_energies, tE_binc) - 1

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))

        model = self.make_fit_model(self._n_components)
        fl_ax = axs.ravel()

        for i, p_i in enumerate(plot_indices):
            log_plot_e = np.log10(plot_energies[i])

            model_params: List[float] = []
            for comp in range(self.n_components):
                mu = np.poly1d(self._poly_params_mu[comp])(log_plot_e)
                sigma = np.poly1d(self._poly_params_sd[comp])(log_plot_e)
                model_params += [mu, sigma]

            res = fit_params[p_i]
            irf_tE_idx = (
                np.digitize(np.log10(plot_energies[i]), self.irf.true_energy_bins) - 1
            )
            log10_rE_bin_edges = self.irf.reco_energy_bins[irf_tE_idx, c_dec]
            log10_rE_binc = log10_rE_bin_edges[:-1] + np.diff(log10_rE_bin_edges) / 2.0
            xs = np.linspace(log10_rE_bin_edges[0], log10_rE_bin_edges[-1], num=100)

            e_reso = self.irf.reco_energy[irf_tE_idx, c_dec].pdf(log10_rE_binc)
            fl_ax[i].plot(log10_rE_binc, e_reso, label="input eres")
            fl_ax[i].plot(xs, model(xs, *model_params), label="poly evaluated")
            fl_ax[i].plot(xs, model(xs, *res), label="nearest bin's parameters")
            fl_ax[i].set_ylim(1e-4, 5)
            fl_ax[i].set_yscale("log")
            fl_ax[i].set_title("True E: {:.1E}".format(plot_energies[i]))
            fl_ax[i].set_xlim(1.5, 8.5)
            fl_ax[i].legend()

        ax = fig.add_subplot(111, frameon=False)

        # Hide tick and tick label of the big axes
        ax.tick_params(
            labelcolor="none", top="off", bottom="off", left="off", right="off"
        )
        ax.grid(False)
        ax.set_xlabel("log10(Reconstructed Energy /GeV)")
        ax.set_ylabel("PDF")
        plt.tight_layout()
        return fig

    @classmethod
    def rewrite_files(cls, season: str = "IC86_II") -> None:
        # call this to rewrite npz files
        cls(DistributionMode.PDF, rewrite=True, season=season)
        cls(DistributionMode.RNG, rewrite=True, season=season)

    @u.quantity_input
    def set_fit_params(self, dec: u.rad) -> None:
        """
        Used in `sim_interface.py`
        """
        dec_idx = np.digitize(dec.to_value(u.rad), self._declination_bins) - 1
        if dec == np.pi / 2:
            dec_idx -= 1

        self._poly_params_mu = self._poly_params_mu__[dec_idx]
        self._poly_params_sd = self._poly_params_sd__[dec_idx]
        self._poly_limits = self._poly_limits__[dec_idx]
        self._fit_params = self._fit_params__[dec_idx]
        self._tE_binc = self._tE_binc__[dec_idx]


class R2021AngularResolution(AngularResolution, HistogramSampler):
    """
    Angular resolution for the ten-year All Sky Point Source release:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    """

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        season: str = "IC86_II",
    ) -> None:
        """
        Instanciate class.
        :param mode: DistributionMode.PDF or .RNG (fitting or simulating)
        :parm rewrite: bool, True if cached files should be overwritten,
                       if there are no cached files they will be generated either way
        :param season: String identifying the detector season
        """

        self._season = season
        self.CACHE_FNAME = f"angular_reso_{season}.npz"

        self.irf = R2021IRF.from_period(season)
        self.mode = mode
        self._rewrite = rewrite
        logger.info("Forced angular rewriting: {}".format(rewrite))

        if mode == DistributionMode.PDF:
            self._func_name = f"{season}AngularResolution"
        else:
            self._func_name = f"{season}AngularResolution_rng"

        self._Emin: float = float("nan")
        self._Emax: float = float("nan")

        self.setup()

    def generate_code(self) -> None:
        """
        Generates stan code by instanciating parent class and the other things.
        """

        if self.mode == DistributionMode.PDF:
            super().__init__(
                f"{self._season}AngularResolution",
                ["true_dir", "reco_dir", "kappa"],
                ["vector", "vector", "real"],
                "real",
            )

        else:
            super().__init__(
                f"{self._season}AngularResolution_rng",
                ["log_true_energy", "log_reco_energy", "true_dir"],
                ["real", "real", "vector"],
                "vector",
            )

        # Define Stan interface
        with self:
            if self.mode == DistributionMode.PDF:
                # Is never used anyways.
                vmf = VMFParameterization(["reco_dir", "true_dir"], "kappa", self.mode)
                ReturnStatement([vmf])

            elif self.mode == DistributionMode.RNG:
                # Create vmf parameterisation, to be fed with kappa calculated from ang_err
                vmf = VMFParameterization(["true_dir"], "kappa", self.mode)

                # Create all psf histogram
                self._make_histogram(
                    "psf", self._psf_hist, self._psf_edges, self._season
                )
                # Create indexing function
                self._make_psf_hist_index(self._season)

                # Create lookup functions used for indexing
                for name, array in zip(
                    [
                        "psf_get_cum_num_vals",
                        "psf_get_cum_num_edges",
                        "psf_get_num_vals",
                        "psf_get_num_edges",
                    ],
                    [
                        self._psf_cum_num_vals,
                        self._psf_cum_num_edges,
                        self._psf_num_vals,
                        self._psf_num_edges,
                    ],
                ):
                    self._make_lookup_functions(name, array, self._season)

                # Create ang_err histogram
                self._make_histogram(
                    "ang", self._ang_hist, self._ang_edges, self._season
                )
                # You know the drill by now
                self._make_ang_hist_index(self._season)
                for name, array in zip(
                    [
                        "ang_get_cum_num_vals",
                        "ang_get_cum_num_edges",
                        "ang_get_num_vals",
                        "ang_get_num_edges",
                    ],
                    [
                        self._ang_cum_num_vals,
                        self._ang_cum_num_edges,
                        self._ang_num_vals,
                        self._ang_num_edges,
                    ],
                ):
                    self._make_lookup_functions(name, array, self._season)

                # Re-uses lookup functions from energy resolution
                etrue_idx = ForwardVariableDef("etrue_idx", "int")
                etrue_idx << FunctionCall(
                    ["log_true_energy"], f"{self._season}_etrue_lookup"
                )

                declination = ForwardVariableDef("declination", "real")
                declination << FunctionCall(["true_dir"], "omega_to_dec")
                dec_idx = ForwardVariableDef("dec_idx ", "int")
                dec_idx << FunctionCall(["declination"], f"{self._season}_dec_lookup")

                ereco_hist_idx = ForwardVariableDef("ereco_hist_idx", "int")
                ereco_hist_idx << FunctionCall(
                    [etrue_idx, dec_idx], f"{self._season}_ereco_get_ragged_index"
                )
                ereco_idx = ForwardVariableDef("ereco_idx", "int")
                ereco_idx << FunctionCall(
                    [
                        "log_reco_energy",
                        FunctionCall(
                            [ereco_hist_idx], f"{self._season}_ereco_get_ragged_edges"
                        ),
                    ],
                    "binary_search",
                )

                # Find appropriate section of psf ragged hist for sampling
                psf_hist_idx = ForwardVariableDef("psf_hist_idx", "int")
                psf_hist_idx << FunctionCall(
                    [etrue_idx, dec_idx, ereco_idx],
                    f"{self._season}_psf_get_ragged_index",
                )
                psf_idx = ForwardVariableDef("psf_idx", "int")
                psf_idx << FunctionCall(
                    [
                        FunctionCall(
                            [psf_hist_idx], f"{self._season}_psf_get_ragged_hist"
                        ),
                        FunctionCall(
                            [psf_hist_idx], f"{self._season}_psf_get_ragged_edges"
                        ),
                    ],
                    "hist_cat_rng",
                )

                # Repeat with angular error
                ang_hist_idx = ForwardVariableDef("ang_hist_idx", "int")
                ang_hist_idx << FunctionCall(
                    [etrue_idx, dec_idx, ereco_idx, psf_idx],
                    f"{self._season}_ang_get_ragged_index",
                )
                ang_err = ForwardVariableDef("ang_err", "real")
                ang_err << FunctionCall(
                    [
                        FunctionCall(
                            [ang_hist_idx], f"{self._season}_ang_get_ragged_hist"
                        ),
                        FunctionCall(
                            [ang_hist_idx], f"{self._season}_ang_get_ragged_edges"
                        ),
                    ],
                    "histogram_rng",
                )

                kappa = ForwardVariableDef("kappa", "real")
                # Convert angular error to kappa
                # Hardcoded p=0.5 (log(1-p)) from the tabulated data of release
                kappa << StringExpression(
                    ["- (2 / (pi() * pow(10, ang_err) / 180)^2) * log(1 - 0.5)"]
                )

                # Stan code needs both deflected direction and kappa
                # Make a vector of length 4, last component is kappa
                return_vec = ForwardVectorDef("return_this", [4])
                # Deflect true direction
                StringExpression(["return_this[1:3] = ", vmf])
                StringExpression(["return_this[4] = kappa"])
                ReturnStatement([return_vec])

    def setup(self) -> None:
        """
        Setup all data fields, load data from cached file or create from scratch.
        """

        self._pdet_limits = (1e2, 1e8)
        self._Emin, self._Emax = self._pdet_limits

        if self.mode == DistributionMode.PDF:
            pass

        elif self.mode == DistributionMode.RNG:
            # party in the back
            # extract *all* the histograms bar ereco
            # check for loading of data
            if self.CACHE_FNAME in Cache and not self._rewrite:
                logger.info("Loading angular data from file.")
                with Cache.open(self.CACHE_FNAME, "rb") as fr:
                    data = np.load(fr, allow_pickle=True)
                    self._psf_cum_num_edges = data["psf_cum_num_edges"]
                    self._psf_cum_num_vals = data["psf_cum_num_vals"]
                    self._psf_num_vals = data["psf_num_vals"]
                    self._psf_num_edges = data["psf_num_edges"]
                    self._psf_hist = data["psf_vals"]
                    self._psf_edges = data["psf_edges"]
                    self._ang_edges = data["ang_edges"]
                    self._ang_hist = data["ang_vals"]
                    self._ang_num_vals = data["ang_num_vals"]
                    self._ang_num_edges = data["ang_num_edges"]
                    self._ang_cum_num_vals = data["ang_cum_num_vals"]
                    self._ang_cum_num_edges = data["ang_cum_num_edges"]

            else:
                logger.info("Re-doing angular data and saving to file.")
                self._generate_ragged_psf_data(self.irf)
                with Cache.open(self.CACHE_FNAME, "wb") as fr:
                    np.savez(
                        fr,
                        psf_cum_num_edges=self._psf_cum_num_edges,
                        psf_cum_num_vals=self._psf_cum_num_vals,
                        psf_num_vals=self._psf_num_vals,
                        psf_num_edges=self._psf_num_edges,
                        psf_vals=self._psf_hist,
                        psf_edges=self._psf_edges,
                        ang_edges=self._ang_edges,
                        ang_vals=self._ang_hist,
                        ang_num_vals=self._ang_num_vals,
                        ang_num_edges=self._ang_num_edges,
                        ang_cum_num_vals=self._ang_cum_num_vals,
                        ang_cum_num_edges=self._ang_cum_num_edges,
                    )

        else:
            raise ValueError("You weren't supposed to do that.")

    def kappa(self) -> None:
        """
        Dummy method s.t. the parents don't complain
        """
        pass

    @classmethod
    def rewrite_files(cls, season: str = "IC86_II"):
        """
        Rewrite cached file
        """

        cls(DistributionMode.RNG, rewrite=True, season=season)


class R2021DetectorModel(ABC, DetectorModel):
    """
    Detector model class of ten-year All Sky Point Source release:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    Only knows muon track events.
    """

    logger = logging.getLogger(__name__ + ".R2021DetectorModel")
    logger.setLevel(logging.DEBUG)

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
        season: str = "IC86_II",
    ) -> None:
        """
        Instantiate R2021 detector model
        :param mode: DistributionMode.PDF (for fits) or .RNG (for simulations)
        :param rewrite: bool, if True rewrites all related cache files
        :param make_plots: bool, if True creates diagnostic plots of the energy parameterisation
        :param n_components: integer number of the energy resolution's lognormal mixture components
        :param ereco_cuts: bool, if True applies exp-data Ereco cuts on simulated events
        :param season: String identifying the detector season
        """

        self._season = season
        super().__init__(mode)

        if mode == DistributionMode.PDF:
            self._func_name = f"{season}PDF"
        elif mode == DistributionMode.RNG:
            self._func_name = f"{season}_rng"

        self._angular_resolution = R2021AngularResolution(mode, rewrite, season=season)

        self._energy_resolution = R2021EnergyResolution(
            mode, rewrite, make_plots, n_components, ereco_cuts, season=season
        )

        self._eff_area = R2021EffectiveArea(mode, season=season)

    def _get_effective_area(self) -> R2021EffectiveArea:
        return self._eff_area

    def _get_energy_resolution(self) -> R2021EnergyResolution:
        return self._energy_resolution

    def _get_angular_resolution(self) -> R2021AngularResolution:
        return self._angular_resolution

    @staticmethod
    def _RNG_FILENAME(season: str):
        return f"{season}_rng.stan"

    @staticmethod
    def _PDF_FILENAME(season: str):
        return f"{season}_pdf.stan"

    @classmethod
    def __generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
        season: str = "IC86_II",
    ) -> None:
        """
        Classmethod to generate stan code of entire detector.
        Will be written to package's usual stan folder, i.e. <current directory>/.stan_files/
        All code is inside a seperate file, included by `sim_interface.py` or `fit_interface.py`,
        therefore the functions block statement is deleted before writing the code to a file.
        """

        # check if stan code is already generated, delegating the task of checking for correct version
        # to the end-user
        try:
            files = os.listdir(path)
        except FileNotFoundError:
            files = []
            os.makedirs(path)
        finally:
            if not rewrite:
                if mode == DistributionMode.PDF and cls.PDF_FILENAME in files:
                    return os.path.join(path, cls.PDF_FILENAME)
                elif mode == DistributionMode.RNG and cls.RNG_FILENAME in files:
                    return os.path.join(path, cls.RNG_FILENAME)

            else:
                cls.logger.info("Generating r2021 stan code.")

        with StanGenerator() as cg:
            instance = R2021DetectorModel(
                mode=mode,
                rewrite=rewrite,
                make_plots=make_plots,
                n_components=n_components,
                ereco_cuts=ereco_cuts,
                season=season,
            )
            instance.effective_area.generate_code()
            instance.angular_resolution.generate_code()
            instance.energy_resolution.generate_code()
            code = cg.generate()
        code = code.removeprefix("functions\n{")
        code = code.removesuffix("\n}\n")
        if not os.path.isdir(path):
            os.makedirs(path)
        if mode == DistributionMode.PDF:
            with open(os.path.join(path, cls._PDF_FILENAME(season)), "w+") as f:
                f.write(code)
            return os.path.join(path, cls._PDF_FILENAME(season))
        else:
            with open(os.path.join(path, cls._RNG_FILENAME(season)), "w+") as f:
                f.write(code)
            return os.path.join(path, cls._RNG_FILENAME(season))

    def generate_pdf_function_code(self, sources: Sources = Sources()):
        """
        Generate a wrapper for the IRF in `DistributionMode.PDF`.
        Takes `Sources` instance as argument to generate energy likelihood
        and effective area for all point sources.
        Assumes that astro diffuse and atmo diffuse model components are present.
        If not, they are disregarded by the model likelihood.
        Has signature
        real true_energy [Gev] : true neutrino energy
        real detected_energy [GeV] : detected muon energy
        unit_vector[3] : detected direction of event
        array[] unit_vector[3] : array of point source's positions
        Returns a tuple of type
        1 array[Ns] real : log(energy likelihood) of all point sources
        2 array[Ns] real : log(effective area) of all point sources
        3 array[3] real : array with log(energy likelihood), log(effective area)
            and log(effective area) for atmospheric component.
        For cascades the last entry is negative_infinity().
        """

        Ns = len(sources.point_source)

        UserDefinedFunction.__init__(
            self,
            self._func_name,
            ["true_energy", "detected_energy", "omega_det", "src_pos"],
            ["real", "real", "vector", "array[] vector"],
            "tuple(array[] real, array[] real, array[] real)",
        )
        with self:
            ps_eres = ForwardArrayDef("ps_eres", "real", ["[", Ns, "]"])
            ps_aeff = ForwardArrayDef("ps_aeff", "real", ["[", Ns, "]"])
            diff = ForwardArrayDef("diff", "real", ["[3]"])
            with ForLoopContext(1, Ns, "i") as i:
                ps_eres[i] << self.energy_resolution(
                    "log10(true_energy)", "log10(detected_energy)", "src_pos[i]"
                )
                ps_aeff[i] << FunctionCall(
                    [
                        self.effective_area("true_energy", "src_pos[i]"),
                    ],
                    "log",
                )

            diff[1] << self.energy_resolution(
                "log10(true_energy)", "log10(detected_energy)", "omega_det"
            )

            diff[2] << FunctionCall(
                [
                    self.effective_area("true_energy", "omega_det"),
                ],
                "log",
            )

            diff[3] << diff[2]

            ReturnStatement(["(ps_eres, ps_aeff, diff)"])

    def generate_rng_function_code(self):
        """
        Generate a wrapper for the IRF in `DistributionMode.RNG`.
        Has signature
        real true_energy [GeV], unit_vector[3] source position
        Returns a vector with entries
        1 reconstructed energy [GeV]
        2:4 reconstructed direction [unit_vector]
        5 kappa
        """

        UserDefinedFunction.__init__(
            self,
            self._func_name,
            ["true_energy", "omega"],
            ["real", "vector"],
            "vector",
        )

        with self:
            return_this = ForwardVariableDef("return_this", "vector[5]")
            return_this[1] << FunctionCall(
                [10.0, self.energy_resolution("log10(true_energy)", "omega")], "pow"
            )
            return_this[2:5] << self.angular_resolution(
                "log10(true_energy)", "log10(return_this[1])", "omega"
            )
            ReturnStatement([return_this])


class IC40DetectorModel(R2021DetectorModel):
    RNG_FILENAME = "IC40_rng.stan"
    PDF_FILENAME = "IC40_pdf.stan"

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC40",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC40",
            path=path,
        )


class IC59DetectorModel(R2021DetectorModel):
    RNG_FILENAME = "IC59_rng.stan"
    PDF_FILENAME = "IC59_pdf.stan"

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC59",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC59",
            path=path,
        )


class IC79DetectorModel(R2021DetectorModel):
    RNG_FILENAME = "IC79_rng.stan"
    PDF_FILENAME = "IC79_pdf.stan"

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC79",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC79",
            path=path,
        )


class IC86_IDetectorModel(R2021DetectorModel):
    RNG_FILENAME = "IC86_I_rng.stan"
    PDF_FILENAME = "IC86_I_pdf.stan"

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC86_I",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC86_I",
            path=path,
        )


class IC86_IIDetectorModel(R2021DetectorModel):
    RNG_FILENAME = "IC86_II_rng.stan"
    PDF_FILENAME = "IC86_II_pdf.stan"

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC86_II",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        make_plots: bool = False,
        n_components: int = 3,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
            make_plots=make_plots,
            n_components=n_components,
            ereco_cuts=ereco_cuts,
            season="IC86_II",
            path=path,
        )
