from typing import Sequence, Tuple, Iterable, List, Callable
import os
from itertools import product

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
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
    RayleighParameterization,
    TruncatedParameterization,
    SimpleHistogram,
    ReturnStatement,
    FunctionCall,
    DistributionMode,
    LognormalMixture,
    ForLoopContext,
    WhileLoopContext,
    ForwardVariableDef,
    InstantVariableDef,
    ForwardArrayDef,
    ForwardVectorDef,
    StanArray,
    StringExpression,
    UserDefinedFunction,
    TwoDimHistInterpolation,
)
from .detector_model import (
    EffectiveArea,
    GridInterpolationEnergyResolution,
    LogNormEnergyResolution,
    AngularResolution,
    DetectorModel,
)

from ..source.source import Sources

from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.point_source_likelihood.energy_likelihood import (
    MarginalisedIntegratedEnergyLikelihood,
)
from skyllh.analyses.i3.publicdata_ps.utils import FctSpline1D, FctSpline2D
from icecube_tools.detector.r2021 import R2021IRF
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import PDSmearingMatrix

from icecube_tools.utils.data import data_directory

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
Cache.set_cache_dir(".cache")


"""
Implements the 10 year muon track point source data set of IceCube.
Makes use of existing `icecube_tools` package.
Classes implement organisation of data and stan code generation.
"""


class HistogramSampler:
    """
    Class to create histograms in stan-readable format.
    """

    # These are fitted correction factors to low energy IRFs
    # used to better model the atmospheric background at ~1TeV

    CORRECTION_FACTOR = {}
    """
        # Season
        "IC86_II": {
            # dec bin
            1: {
                # Etrue bin
                0: {"a": 0.688, "b": 0.91},
                1: {"a": 0.82, "b": 0.64},
            }
        }
    }
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
            apply_roi = np.all(list(_.apply_roi for _ in ROIList.STACK))
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


class R2021EnergyResolution(GridInterpolationEnergyResolution, HistogramSampler):
    """
    Energy resolution for the ten-year All Sky Point Source release:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    """

    _logEreco_grid = np.linspace(2, 8, 400)

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
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
        self.CACHE_FNAME_LOGNORM = f"energy_reso_grid_interp_{season}.npz"
        self.CACHE_FNAME_HISTOGRAM = f"energy_reso_histogram_{season}.npz"
        # Instantiate the icecube_tools IRFs with the appropriate re-binning
        # of reconstructed energies according to the correction factors
        try:
            corr = self.CORRECTION_FACTOR[self._season]
            self.irf = R2021IRF.from_period(self._season, correction_factor=corr)
        except KeyError:
            self.irf = R2021IRF.from_period(self._season)
        self._icecube_tools_eres = MarginalisedIntegratedEnergyLikelihood(
            season, np.linspace(1, 9, 25)
        )
        # Copy true energy bins from IRF
        self._log_tE_bin_edges = self.irf.true_energy_bins
        self._log_tE_binc = self.irf.true_energy_values
        self._tE_bin_edges = np.power(10, self._log_tE_bin_edges)

        # Setup reconstructed energy bins
        self._log_rE_binc = self._logEreco_grid
        diff = np.diff(self._log_rE_binc)[0]
        self._log_rE_edges = np.linspace(
            self._log_rE_binc[0] - diff / 2,
            self._log_rE_binc[-1] + diff / 2,
            self._log_rE_binc.size + 1,
        )
        self._fill_index = 7
        self._dec_bin_edges = self.irf.declination_bins << u.rad
        self._dec_binc = self._dec_bin_edges[:-1] + np.diff(self._dec_bin_edges) / 2
        self._dec_binc << u.rad
        self._sin_dec_edges = np.sin(self._dec_bin_edges.to_value(u.rad))
        self._sin_dec_binc = self._sin_dec_edges[:-1] + np.diff(self._sin_dec_edges) / 2

        self._make_ereco_cuts = ereco_cuts
        self._ereco_cuts = self._icecube_tools_eres._ereco_limits
        self._aeff_dec_bins = self._icecube_tools_eres.declination_bins_aeff

        self.mode = mode
        if self.mode == DistributionMode.PDF:
            self._func_name = f"{season}EnergyResolution"
        elif self.mode == DistributionMode.RNG:
            self._func_name = f"{season}EnergyResolution_rng"
        self._rewrite = rewrite
        logger.info("Forced energy rewriting: {}".format(rewrite))

        self._pdet_limits = (1e2, 1e9)

        self.setup()

    def __call__(self, log_rE, log_tE):
        pass

    def generate_code(self) -> None:
        """
        Generates stan code by instanciating parent class and the other things.
        """

        # initialise parent classes with proper signature for stan functions
        if self.mode == DistributionMode.PDF:
            super().__init__(
                self._func_name,
                ["log_true_energy", "log_reco_energy", "omega", "ereco_idx"],
                ["real", "real", "vector", "int"],
                "real",
            )
        elif self.mode == DistributionMode.RNG:
            super().__init__(
                self._func_name,
                ["log_true_energy", "omega"],
                ["real", "vector"],
                "real",
            )

        if self.mode == DistributionMode.PDF:
            logger.info("Generating pdf code using spline evaluations.")
            with self:
                ereco_idx = StringExpression(["ereco_idx"])
                # Argument `omega` is cartesian vector, cos(z) (z is direction) is theta in spherical coords
                # declination = ForwardVariableDef("declination", "real")
                # declination << FunctionCall(["omega"], "omega_to_dec")

                # return 2d interpolated grid evals
                logEreco_c = StanArray("logEreco_c", "real", self._logEreco_grid)
                logEtrue_c = StanArray("logEtrue_c", "real", self._log_tE_binc)
                # TODO fix dec index, hard-coded 1 rn
                grid_evals = StanArray("grid_evals", "real", self._evaluations[1])
                # TODO: replace 2d interpolation with finding the correct slices of
                # Ereco enclosing the actual value, hand these over to 2d interpolation
                # make another 2d interpolation function skipping the useless first binary search?
                likelihood = ForwardVariableDef("loglike", "real")
                likelihood << FunctionCall(
                    [
                        FunctionCall([logEtrue_c], "to_vector"),
                        FunctionCall([grid_evals[ereco_idx]], "to_vector"),
                        "log_true_energy",
                    ],
                    "interpolate",
                )
                ReturnStatement([FunctionCall([likelihood], "log")])

        if self.mode == DistributionMode.RNG:
            logger.info("Generating simulation code using histograms")
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

        self._ereco_splines = []
        self._2dsplines = []

        # Loop over declinations
        for c, dec in enumerate(self._dec_binc):
            # Create the IRF spline for each true energy
            self._ereco_splines.append(
                [
                    create_reco_e_pdf_for_true_e(
                        np.power(10, _) * u.GeV, dec, season=self._season
                    )
                    for _ in self._log_tE_binc
                ]
            )
        # Create empty array to store evaluated splines
        self._evaluations = np.zeros(
            (
                self._dec_binc.size,
                self._log_rE_binc.size,
                self._log_tE_binc.size,
            )
        )

        # Loop over the declination bins of the IRF
        for c, dec in enumerate(self._dec_binc):
            # Evaluate the splines over a grid of Ereco x Etrue
            # Etrue is determined by the provided IRFs
            # Ereco is set to a dense grid
            self._evaluations[c] = np.vstack(
                [
                    self._ereco_splines[c][c_E](self._log_rE_binc)
                    / self._ereco_splines[c][c_E].norm
                    for c_E, E in enumerate(self._log_tE_binc)
                ]
            ).T
            for c_E, logE in enumerate(self._log_tE_binc):
                # Find the highest and lowest non-zero entry
                # in each spline's evaluation, outside all values are zero
                idx = np.max(np.argwhere(self._evaluations[c, :, c_E]))
                # Needs linear scale because of power law definition
                E_cont = np.power(
                    10, self._log_rE_binc[idx + 1 :] - self._log_rE_binc[idx]
                )
                self._evaluations[c, idx + 1 :, c_E] = (
                    np.power(
                        E_cont,
                        -self._fill_index,
                    )
                    * self._evaluations[c, idx, c_E]
                )
                # Again with low energy end
                # at this index, there already is a non-zero entry
                # so the range indexing should be :idx
                idx = np.min(np.argwhere(self._evaluations[c, :, c_E]))
                # Needs linear scale because of power law definition
                E_cont = np.power(10, self._log_rE_binc[:idx] - self._log_rE_binc[idx])
                self._evaluations[c, :idx, c_E] = (
                    # switch sign of index here because it should be a rising flank
                    np.power(E_cont, self._fill_index)
                    * self._evaluations[c, idx, c_E]
                )
            # Spline the evaluations linearly, used in `prob_Edet_above_threshold`
            self._2dsplines.append(
                RectBivariateSpline(
                    self._log_rE_binc,
                    self._log_tE_binc,
                    self._evaluations[c],
                    kx=1,
                    ky=1,
                )
            )

    @u.quantity_input
    def prob_Edet_above_threshold(
        self,
        true_energy: u.GeV,
        lower_threshold_energy: u.GeV,
        dec: u.rad,
        upper_threshold_energy=None,
        use_interpolation: bool = False,
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
            if none provided, use highest possible value,
            assumes only one value is provided
        :param use_lognorm: bool, if True use lognormal parameterisation
        """
        # Truncate input energies to safe range, i.e. range covered by IRFs
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
        else:
            upper_threshold_energy = np.array([1e9]) * u.GeV

        if upper_threshold_energy.size == 1:
            upper_threshold_energy = (
                np.full(energy_trunc.shape, upper_threshold_energy.to_value(u.GeV))
                * u.GeV
            )
        else:
            assert upper_threshold_energy.shape == energy_trunc.shape
        # Limits of Ereco in dec binning of effective area
        idx_dec_aeff = np.digitize(dec.to_value(u.rad), self._aeff_dec_bins) - 1
        # Get the according IRF dec bins (there are only 3)
        idx_dec_eres = np.digitize(dec.to(u.rad), self._dec_bin_edges) - 1
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

        e_high = np.log10(upper_threshold_energy.to_value(u.GeV))
        e_trunc = np.log10(energy_trunc.to_value(u.GeV))

        if use_interpolation:

            for cE, cD in product(
                range(self.irf.true_energy_bins.size - 1),
                range(self.irf.declination_bins.size - 1),
            ):
                if cD not in idx_dec_eres:
                    continue

                # find the slices of evaluations with Etrue including the queried
                for c, (Et, logErl, logErh) in enumerate(zip(e_trunc, e_low, e_high)):
                    if cD != idx_dec_eres[c]:
                        continue

                    bins_per_dec = 40
                    n_bins = int(np.ceil((logErh - logErl) * bins_per_dec))
                    loge_edges = np.linspace(logErl, logErh, n_bins + 1)
                    dloge = np.diff(loge_edges)
                    loge_c = loge_edges[:-1] + dloge / 2

                    evals = self._2dsplines[cD](loge_c, Et, grid=False)
                    integral = np.sum(evals * dloge)

                    prob[c] = integral

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
                    prob[(cE == idx_tE) & (cD == idx_dec_eres)] = (
                        1.0
                        - self.irf.reco_energy[cE, cD].cdf(
                            e_low[(cE == idx_tE) & (cD == idx_dec_eres)]
                        )
                    )
                else:
                    pdf = self.irf.reco_energy[cE, cD]
                    prob[(cE == idx_tE) & (cD == idx_dec_eres)] = pdf.cdf(
                        np.log10(upper_threshold_energy.to_value(u.GeV))[
                            (cE == idx_tE) & (cD == idx_dec_eres)
                        ]
                    ) - pdf.cdf(e_low[(cE == idx_tE) & (cD == idx_dec_eres)])

        return prob

    @classmethod
    def rewrite_files(cls, season: str = "IC86_II") -> None:
        # call this to rewrite npz files
        cls(DistributionMode.PDF, rewrite=True, season=season)
        cls(DistributionMode.RNG, rewrite=True, season=season)


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
                ["true_dir", "reco_dir", "sigma", "kappa"],
                ["vector", "vector", "real", "real"],
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
                angular_parameterisation = RayleighParameterization(
                    ["true_dir", "reco_dir"], "sigma", self.mode
                )
                ReturnStatement([angular_parameterisation])

            elif self.mode == DistributionMode.RNG:
                angular_parameterisation = RayleighParameterization(
                    ["true_dir"], "ang_err", self.mode
                )

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
                # Samples angular uncertainty in degrees
                ang_err << FunctionCall(
                    [
                        10,
                        FunctionCall(
                            [
                                FunctionCall(
                                    [ang_hist_idx],
                                    f"{self._season}_ang_get_ragged_hist",
                                ),
                                FunctionCall(
                                    [ang_hist_idx],
                                    f"{self._season}_ang_get_ragged_edges",
                                ),
                            ],
                            "histogram_rng",
                        ),
                    ],
                    "pow",
                )

                # Convert to radian
                ang_err << StringExpression(["pi() * ang_err / 180.0"])
                kappa = ForwardVariableDef("kappa", "real")
                # Convert angular error to kappa
                kappa << StringExpression(["- (2 / ang_err^2) * log(1 - 0.683)"])

                # Stan code needs both deflected direction and kappa
                # Make a vector of length 4, last component is kappa
                return_vec = ForwardVectorDef("return_this", [4])
                # Deflect true direction
                StringExpression(["return_this[1:3] = ", angular_parameterisation])
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

    _logEreco_grid = R2021EnergyResolution._logEreco_grid

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

        self._energy_resolution = R2021EnergyResolution(mode, rewrite, season=season)

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
        """
        finally:
            if not rewrite:
                if mode == DistributionMode.PDF and cls.PDF_FILENAME in files:
                    return os.path.join(path, cls.PDF_FILENAME)
                elif mode == DistributionMode.RNG and cls.RNG_FILENAME in files:
                    return os.path.join(path, cls.RNG_FILENAME)

            else:
                cls.logger.info("Generating r2021 stan code.")
        """
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

    def generate_pdf_function_code(self, single_ps: bool = False):
        """
        Generate a wrapper for the IRF in `DistributionMode.PDF`.
        Assumes that astro diffuse and atmo diffuse model components are present.
        If not, they are disregarded by the model likelihood.
        Has signature dependent on the parameter `single_ps`, defaulting to False:
        real true_energy [Gev] : true neutrino energy
        real detected_energy [GeV] : detected muon energy
        unit_vector[3] : detected direction of event
        array[] unit_vector[3] : array of point source's positions
        Returns a tuple of type
        1 array[Ns] real : log(energy likelihood) of all point sources
        2 array[Ns] real : log(effective area) of all point sources
        3 array[3] real : array with log(energy likelihood), log(effective area)
            and log(effective area) for atmospheric component.
        If `single_ps==True`, all arrays regarding the PS are instead reals.
        For cascades the last entry is negative_infinity().
        """

        if not single_ps:
            UserDefinedFunction.__init__(
                self,
                self._func_name,
                ["true_energy", "detected_energy", "omega_det", "src_pos", "reco_idx"],
                ["real", "real", "vector", "array[] vector", "int"],
                "tuple(array[] real, array[] real, array[] real)",
            )
        else:
            UserDefinedFunction.__init__(
                self,
                self._func_name,
                ["true_energy", "detected_energy", "omega_det", "src_pos", "reco_idx"],
                ["real", "real", "vector", "vector", "int"],
                "tuple(real, real, array[] real)",
            )

        with self:
            log10Ereco = InstantVariableDef(
                "log10Ereco", "real", ["log10(detected_energy)"]
            )
            log10Etrue = InstantVariableDef(
                "log10Etrue", "real", ["log10(true_energy)"]
            )
            dec_bins_eres = StanArray(
                "dec_bins", "real", self.energy_resolution._declination_bins
            )
            # dec_eres_idx = ForwardVariableDef("dec_ind", "int")
            # declination = ForwardVariableDef("declination", "real")
            # declination << FunctionCall(["omega"], "omega_to_dec")
            # dec_eres_idx << FunctionCall([declination, dec_bins_eres], "binary_search")
            if not single_ps:
                Ns = InstantVariableDef("Ns", "int", ["size(src_pos)"])
                ps_eres = ForwardArrayDef("ps_eres", "real", ["[", Ns, "]"])
                ps_aeff = ForwardArrayDef("ps_aeff", "real", ["[", Ns, "]"])
                with ForLoopContext(1, Ns, "i") as i:
                    ps_eres[i] << self.energy_resolution(
                        log10Etrue, log10Ereco, "src_pos[i]", "reco_idx"
                    )
                    ps_aeff[i] << FunctionCall(
                        [
                            self.effective_area("true_energy", "src_pos[i]"),
                        ],
                        "log",
                    )

            else:
                ps_eres = ForwardVariableDef("ps_eres", "real")
                ps_aeff = ForwardVariableDef("ps_aeff", "real")
                ps_eres << self.energy_resolution(
                    log10Etrue, log10Ereco, "src_pos", "reco_idx"
                )
                ps_aeff << FunctionCall(
                    [
                        self.effective_area("true_energy", "src_pos"),
                    ],
                    "log",
                )

            diff = ForwardArrayDef("diff", "real", ["[3]"])

            diff[1] << self.energy_resolution(
                log10Etrue, log10Ereco, "omega_det", "reco_idx"
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
            log10Ereco = ForwardVariableDef("log10Ereco", "real")
            log10Etrue = InstantVariableDef(
                "log10Etrue", "real", ["log10(true_energy)"]
            )
            log10Ereco << self.energy_resolution(log10Etrue, "omega")
            return_this[1] << FunctionCall([10.0, log10Ereco], "pow")
            return_this[2:5] << self.angular_resolution(log10Etrue, log10Ereco, "omega")
            ReturnStatement([return_this])


class IC40DetectorModel(R2021DetectorModel):
    RNG_FILENAME = "IC40_rng.stan"
    PDF_FILENAME = "IC40_pdf.stan"

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            ereco_cuts=ereco_cuts,
            season="IC40",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
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
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            ereco_cuts=ereco_cuts,
            season="IC59",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
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
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            ereco_cuts=ereco_cuts,
            season="IC79",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
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
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            ereco_cuts=ereco_cuts,
            season="IC86_I",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
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
        n_components: int = 3,
        ereco_cuts: bool = True,
    ):
        super().__init__(
            mode=mode,
            rewrite=rewrite,
            ereco_cuts=ereco_cuts,
            season="IC86_II",
        )

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        rewrite: bool = False,
        ereco_cuts: bool = True,
        path: str = STAN_GEN_PATH,
    ):
        cls._R2021DetectorModel__generate_code(
            mode=mode,
            rewrite=rewrite,
            ereco_cuts=ereco_cuts,
            season="IC86_II",
            path=path,
        )
