from ast import Return
from pyclbr import Function
from tokenize import String
from typing import Sequence, Tuple, Iterable
import os
import pandas as pd
import numpy as np
import sys

from hierarchical_nu.stan.interface import STAN_GEN_PATH

from icecube_tools.detector.r2021 import R2021IRF

from hierarchical_nu.backend.stan_generator import ElseBlockContext, ElseIfBlockContext, IfBlockContext, StanGenerator

from hierarchical_nu.stan.interface import STAN_PATH

from ..utils.cache import Cache
from ..backend import (
    FunctionsContext,
    VMFParameterization,
    PolynomialParameterization,
    TruncatedParameterization,
    LogParameterization,
    SimpleHistogram,
    SimpleHistogram_rng,
    ReturnStatement,
    FunctionCall,
    DistributionMode,
    LognormalMixture,
    ForLoopContext,
    ForwardVariableDef,
    ForwardArrayDef,
    ForwardVectorDef,
    StanArray,
    StringExpression,
    UserDefinedFunction,
)
from .detector_model import (
    EffectiveArea,
    EnergyResolution,
    AngularResolution,
    DetectorModel,
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Cache.set_cache_dir(".cache")


class HistogramSampler():
    def __init__(self, rewrite: bool):
        #should include some data season for icecube_tools
        self._rewrite = rewrite


    def _generate_ragged_ereco_data(self, irf: R2021IRF):
        """
        Should take the R2021 icecube_tools irf and operate on its arrays.
        """

        logger.debug("Creating ragged arrays for reco energy.")

        num_of_bins = []
        num_of_values = []
        cum_num_of_values = []
        cum_num_of_bins = []
        counter = 0   #TODO change to try-except
        bins = []
        values = []
        for c_e, etrue in enumerate(irf.true_energy_values):
            for c_d, dec in enumerate(irf.declination_bins[:-1]):
                b = irf.reco_energy_bins[c_e, c_d]
                n = irf.reco_energy[c_e, c_d].pdf(irf.reco_energy_bins[c_e, c_d][:-1]+0.01)
                #n, b = irf._marginalisation(c_e, c_d)
                bins.append(b)
                values.append(n)
                """
                if counter != 0:
                    bins = np.concatenate((bins, b))
                    values = np.concatenate((values, n))
                else:
                    bins = b.copy()
                    values = n.copy()
                """
                num_of_values.append(n.size)
                num_of_bins.append(b.size)
                try:
                    cum_num_of_values.append(cum_num_of_values[-1]+n.size)
                    cum_num_of_bins.append(cum_num_of_bins[-1]+b.size)
                except IndexError:
                    cum_num_of_values.append(n.size)
                    cum_num_of_bins.append(b.size)

        self._ereco_cum_num_vals = cum_num_of_values
        self._ereco_cum_num_edges = cum_num_of_bins
        self._ereco_num_vals = num_of_values
        self._ereco_num_edges = num_of_bins
        self._ereco_hist = np.concatenate(values)
        self._ereco_edges = np.concatenate(bins)
        self._tE_bin_edges = np.power(10, irf.true_energy_bins)


    def _generate_ragged_psf_data(self, irf: R2021IRF):
        """
        Should take the R2021 icecube_tools irf and operate on its arrays.
        """

        #TODO cleanup
        logger.debug("Creating ragged arrays for angular parts.")
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

        #this does work! extend for ang_err!
        for etrue, _ in enumerate(irf.true_energy_values):  # 0->13, 1->14
            #iterate through etrue bins
            for dec, _ in enumerate(irf.declination_bins[:-1]):  # 0->2, 1->3
                #iterate through dec bins
                
                #get ereco bins, if value is nonzero -> get psf, else -> skip
                n_reco, bins_reco = irf._marginalisation(etrue, dec)
            
                #print(n_reco)
                for c, v in enumerate(n_reco):   #0->19, 1->20
                    #iterate through ereco bins
                    if v != 0.:
                        #print(etrue, dec, c)
                        #get psf distribution
                        n_psf, bins_psf = irf._marginalize_over_angerr(etrue, dec, c)
                        #shorten the arrays: get rid of zeros from the beginning and the end
                        #psf_val_start = np.nonzero(n_psf!=0)[0].min()
                        #psf_val_end = n_psf.size - np.nonzero(np.flip(n_psf)!=0)[0].min()
                        #n = n_psf[psf_val_start:psf_val_end]
                        #bins = bins_psf[psf_val_start:psf_val_end+1]
                        n = n_psf.copy()
                        bins = bins_psf.copy()
                        psf_vals.append(n)
                        psf_edges.append(bins)
                        psf_num_vals.append(n.size)
                        psf_num_edges.append(bins.size)
                        try:
                            psf_cum_num_vals.append(psf_cum_num_vals[-1]+n.size)
                        except IndexError:
                            psf_cum_num_vals.append(n.size)
                        try:
                            psf_cum_num_edges.append(psf_cum_num_edges[-1]+bins.size)
                        except IndexError:
                            psf_cum_num_edges.append(bins.size)
                        
                        #do it again for ang_err
                        for c_psf, v_psf in enumerate(n_psf):
                            if v_psf != 0.:
                                n_ang, bins_ang = irf._get_angerr_dist(etrue, dec, c, c_psf)
                                #shorten n_ang, bins_ang with previous method
                                #ang_val_start = np.nonzero(n_ang!=0)[0].min()
                                #ang_val_end = n_ang.size - np.nonzero(np.flip(n_ang)!=0)[0].min()
                                #n = n_ang[ang_val_start:ang_val_end]
                                #bins = bins_psf[ang_val_start:ang_val_end+1]
                                n = n_ang.copy()
                                bins = bins_ang.copy()
                                ang_vals.append(n)
                                ang_edges.append(bins)
                                ang_num_vals.append(n.size)
                                ang_num_edges.append(bins.size)
                            else:
                                ang_num_vals.append(0)
                                ang_num_edges.append(0)
                        
                    else:
                        #do counter stuff, add only zeros/previous values
                        psf_num_vals.append(0)
                        psf_num_edges.append(0)
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
                #break
            #break
        
        for v in ang_num_vals:
            try:
                ang_cum_num_vals.append(ang_cum_num_vals[-1]+v)
            except IndexError:
                ang_cum_num_vals.append(v)
        for v in ang_num_edges:
            try:
                ang_cum_num_edges.append(ang_cum_num_edges[-1]+v)
            except IndexError:
                ang_cum_num_edges.append(v)
        

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
        

    def _make_hist_lookup_functions(self):
        logger.debug("Making etrue/dec lookup functions.")
        self._etrue_lookup = UserDefinedFunction(
            "etrue_lookup", ["true_energy"], ["real"], "int"
        )
        with self._etrue_lookup:
            #truncated_e = TruncatedParameterization("true_energy", *self._poly_limits)
            #log_trunc_e = LogParameterization(truncated_e)
            #do binary search for bin of true energy
            etrue_bins = StanArray("log_etrue_bins", "real", np.log10(self._tE_bin_edges))
            ReturnStatement(["binary_search(", "true_energy", ", ", etrue_bins, ")"])

        self._dec_lookup = UserDefinedFunction("dec_lookup", ["declination"], ["real"], "int")
        with self._dec_lookup:
            #do binary search for bin of declination
            declination_bins = StanArray("dec_bins", "real", self._declination_bins)
            ReturnStatement(["binary_search(declination, ", declination_bins, ")"])


        """
            #find appropriate section in ragged array structure
            hist_ind = ForwardVariableDef("hist_ind", "int")
            hist_ind << FunctionCall([etrue_ind, dec_ind], "ereco_get_ragged_index")

            with IfBlockContext(["type == 0"]):
                ReturnStatement([FunctionCall([hist_ind], "ereco_get_ragged_hist")])
            with ElseIfBlockContext(["type == 1"]):
                ReturnStatement([FunctionCall([hist_ind], "ereco_get_hist_edges")])
        """


    def _make_histogram(self, data_type: str, hist_values: Iterable[float], hist_bins: Iterable[float]):
        logger.debug("Making histograms.")
        self._ragged_hist = UserDefinedFunction("{}_get_ragged_hist".format(data_type), ["idx"], ["int"], "real[]")
        with self._ragged_hist:
            arr = StanArray("arr", "real", hist_values)
            self._make_ragged_start_stop(data_type, "vals")
            ReturnStatement(["arr[start:stop]"])

        self._ragged_edges = UserDefinedFunction("{}_get_ragged_edges".format(data_type), ["idx"], ["int"], "real[]")
        with self._ragged_edges:
            arr = StanArray("arr", "real", hist_bins)
            self._make_ragged_start_stop(data_type, "edges")
            ReturnStatement(["arr[start:stop]"])


    def _make_ereco_hist_index(self):
        logger.debug("Making ereco histogram indexing function.")
        get_ragged_index = UserDefinedFunction(
            "ereco_get_ragged_index", ["etrue", "dec"], ["int", "int"], "int"
        )
        # Takes indices of etrue and dec (to be determined elsewhere!)
        with get_ragged_index:
            ReturnStatement(["dec + (etrue - 1) * 3"])

        
    def _make_psf_hist_index(self):
        logger.debug("Making psf histogram indexing function.")
        get_ragged_index = UserDefinedFunction(
            "psf_get_ragged_index", ["etrue", "dec", "ereco"], ["int", "int", "int"], "int"
        )
        with get_ragged_index:
            #find appropriate expression
            l_etrue = 14
            l_dec = 3
            l_ereco = 20
            l_psf = 20
            ReturnStatement(["ereco + (dec - 1) * 20 + (etrue - 1) * 3  * 20"])

    
    def _make_ang_hist_index(self):
        logger.debug("Making ang histogram indexing function.")
        get_ragged_index = UserDefinedFunction(
            "ang_get_ragged_index", ["etrue", "dec", "ereco", "psf"], ["int", "int", "int", "int"], "int"
        )
        with get_ragged_index:
            ReturnStatement(["psf + (ereco - 1) * 20 + (dec - 1) * 20 * 20 + (etrue - 1) * 3 * 20 * 20"])


    def _make_lookup_functions(self, name, array):
        logger.debug("Making generic lookup functions.")
        #DONE
        #look-up functions for ragged arrays
        f = UserDefinedFunction(name, ["idx"], ["int"], "int")
        with f:
            arr = StanArray("arr", "int", array)
            with IfBlockContext(["idx > ", len(array), "|| idx < 0"]):
                FunctionCall(['"idx outside range, "', "idx"], "reject")
            ReturnStatement(["arr[idx]"])

        
    def _make_ragged_start_stop(self, data, hist):
        logger.debug("Making ragged array indexing.")
        start = ForwardVariableDef("start", "int")
        stop = ForwardVariableDef("stop", "int")
        if hist == "edges" or hist == "vals":
            start << StringExpression(["{}_get_cum_num_{}(idx)-{}_get_num_{}(idx)+1".format(data, hist, data, hist)])
            stop << StringExpression(["{}_get_cum_num_{}(idx)".format(data, hist)])
        else:
            raise ValueError("No other type available.")



class R2021EffectiveArea(EffectiveArea):
    """
    Effective area for the ten-year All Sky Point Source release:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    """

    local_path = "input/tracks/IC86_II_effectiveArea.csv"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "aeff_r2021.npz"

    def __init__(self) -> None:

        #what does this? super() is ABC with no arguments specified
        self._func_name = "R2021EffectiveArea"

        self.setup()


    def generate_code(self):

        super().__init__(
            "R2021EffectiveArea",
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real",
        )

        # Define Stan interface.
        with self:
            hist = SimpleHistogram(
                self._eff_area,
                [self._tE_bin_edges, self._cosz_bin_edges],
                "R2021EffAreaHist",
            )

            cos_dir = "cos(pi() - acos(true_dir[3]))"

            _ = ReturnStatement([hist("true_energy", cos_dir)])


    def setup(self) -> None:

        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                data = np.load(fr)
                eff_area = data["eff_area"]
                tE_bin_edges = data["tE_bin_edges"]
                cosz_bin_edges = data["cosz_bin_edges"]
        else:

            from icecube_tools.detector.effective_area import EffectiveArea

            #cut the arrays short because of numerical inaccuracies in comparing large floats
            aeff = EffectiveArea.from_dataset("20210126")
            eff_area = aeff.values[:-5]
            tE_bin_edges = aeff.true_energy_bins[:-5]
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
        self._rs_bbpl_params["gamma2_scale"] = 1.2

        

class R2021EnergyResolution(EnergyResolution, HistogramSampler):

    """
    Energy resolution for Northern Tracks Sample

    Data from https://arxiv.org/pdf/1811.07979.pdf
    """

    local_path = "input/tracks/IC86_II_smearing.csv"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME_LOGNORM = "energy_reso_lognorm_r2021.npz"
    CACHE_FNAME_HISTOGRAM = "energy_reso_histogram_r2021.npz"

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        rewrite: bool = False,
        gen_type: str = "histogram"
    ) -> None:

        self.irf = R2021IRF()
        self.gen_type = gen_type    # either "histogram" or "lognorm"
        print(gen_type)
        self.mode = mode
        self._rewrite = rewrite
        logger.info("Forced energy rewriting: {}".format(rewrite))
        self.mode = mode
        self._poly_params_mu: Sequence = []
        self._poly_params_sd: Sequence = []
        self._poly_limits: Tuple[float, float] = (float("nan"), float("nan"))
        self._poly_limits_battery: Sequence = []
        self._declination_bins = self.irf.declination_bins

        # For prob_Edet_above_threshold
        self._pdet_limits = (1e2, 1e9)

        self._n_components = 3
        self.setup()

        #mis-use inheritence without initialising parent class
        if self.mode == DistributionMode.PDF:
            self._func_name = "R2021EnergyResolution_lpdf"
        elif self.mode == DistributionMode.RNG:
            self._func_name = "R2021EnergyResolution_rng"

        
    def generate_code(self):

        #initialise parent classes with proper signature for stan functions
        if self.mode == DistributionMode.PDF:
            self.mixture_name = "r2021_energy_res_mix"
            super().__init__(
                self._func_name,
                ["true_energy", "reco_energy", "omega"],
                ["real", "real", "vector"],
                "real",
            )
        elif self.mode == DistributionMode.RNG:
            self.mixture_name = "r2021_energy_res_mix_rng"
            super().__init__(
                self._func_name,
                ["true_energy", "omega"],
                ["real", "vector"],
                "real",
            )

        #actual code generation
        
        if self.gen_type == "lognorm":
            logger.info("Using lognorm")
            with self:
                lognorm = LognormalMixture(self.mixture_name, self.n_components, self.mode)
                truncated_e = TruncatedParameterization("true_energy", *self._poly_limits)
                log_trunc_e = LogParameterization(truncated_e)

                #self._poly_params_mu should have shape (3, n_components, poly_deg+1)s
                mu_poly_coeffs = StanArray(
                    "R2021EnergyResolutionMuPolyCoeffs",
                    "real",
                    self._poly_params_mu,
                )
                #same as above
                sd_poly_coeffs = StanArray(
                    "R2021EnergyResolutionSdPolyCoeffs",
                    "real",
                    self._poly_params_sd,
                )

                mu = ForwardArrayDef("mu_e_res", "real", ["[", self._n_components, "]"])
                sigma = ForwardArrayDef(
                    "sigma_e_res", "real", ["[", self._n_components, "]"]
                )

                weights = ForwardVariableDef(
                    "weights", "vector[" + str(self._n_components) + "]"
                )

                declination = ForwardVariableDef("declination", "real")
                declination << FunctionCall(["omega"], "omega_to_dec")

                declination_bins = StanArray("dec_bins", "real", self._declination_bins)
                declination_index = ForwardVariableDef("dec_ind", "int")
                declination_index << StringExpression(["binary_search(declination, ", declination_bins, ")"])

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
                    log_reco_e = LogParameterization("reco_energy")
                    ReturnStatement([lognorm(log_reco_e, log_mu_vec, sigma_vec, weights)])
                else:
                    ReturnStatement([lognorm(log_mu_vec, sigma_vec, weights)])
                
        elif self.gen_type == "histogram":
            logger.info("Using histograms")
            with self:
                self._make_hist_lookup_functions()
                self._make_histogram("ereco", self._ereco_hist, self._ereco_edges)
                self._make_ereco_hist_index()
                for name, array in zip(["ereco_get_cum_num_vals", "ereco_get_cum_num_edges",
                    "ereco_get_num_vals", "ereco_get_num_edges"],
                    [self._ereco_cum_num_vals, self._ereco_cum_num_edges, self._ereco_num_vals, self._ereco_num_edges]
                ):
                    self._make_lookup_functions(name, array)
                
                #call histogramm with appropriate values/edges
                declination = ForwardVariableDef("declination", "real")
                declination << FunctionCall(["omega"], "omega_to_dec")
                etrue_idx = ForwardVariableDef("etrue_idx", "int")
                dec_idx = ForwardVariableDef("dec_idx", "int")
                ereco_hist_idx = ForwardVariableDef("ereco_hist_idx", "int")
                etrue_idx << FunctionCall(["log10(true_energy)"], "etrue_lookup")
                if self.mode == DistributionMode.PDF:
                    with IfBlockContext(["etrue_idx == 0 || etrue_idx > 14"]):
                        ReturnStatement(["negative_infinity()"])
                dec_idx << FunctionCall(["declination"], "dec_lookup")
                ereco_hist_idx << FunctionCall([etrue_idx, dec_idx], "ereco_get_ragged_index")
                if self.mode == DistributionMode.PDF:
                    ereco_idx = ForwardVariableDef("ereco_idx", "int")
                    ereco_idx << StringExpression(["binary_search(log10(reco_energy), ",
                        FunctionCall([ereco_hist_idx], "ereco_get_ragged_edges"), ")"]
                    )
                    #intercept outside of hist range here:
                    with IfBlockContext(["ereco_idx == 0 || ereco_idx > ereco_get_num_vals(ereco_hist_idx)"]):
                        ReturnStatement(["negative_infinity()"])

                    return_value = ForwardVariableDef("return_value", "real")
                    return_value << StringExpression([FunctionCall([ereco_hist_idx], "ereco_get_ragged_hist"), "[ereco_idx]"])
                    with IfBlockContext(["return_value == 0."]):
                        ReturnStatement(["negative_infinity()"])
                    with ElseBlockContext():
                        ReturnStatement([FunctionCall([return_value], "log")])
                else:
                    ReturnStatement([FunctionCall([FunctionCall([ereco_hist_idx], "ereco_get_ragged_hist"), FunctionCall([ereco_hist_idx], "ereco_get_ragged_edges")], "histogram_rng")])

        '''            
        elif self.mode == DistributionMode.RNG:

            with self:
                """
                #histogram "histogram_rng(array[] real hist_array, array[] real hist_edges)"
                #is defined in utils.stan, is included anway
                self._make_hist_lookup_functions()
                self._make_histogram("ereco", self._ereco_hist, self._ereco_edges)
                self._make_ereco_hist_index()
                for name, array in zip(["ereco_get_cum_num_vals", "ereco_get_cum_num_edges",
                    "ereco_get_num_vals", "ereco_get_num_edges"],
                    [self._ereco_cum_num_vals, self._ereco_cum_num_edges, self._ereco_num_vals, self._ereco_num_edges]
                ):
                    self._make_lookup_functions(name, array)
                

                #call histogramm with appropriate values/edges
                declination = ForwardVariableDef("declination", "real")
                declination << FunctionCall(["omega"], "omega_to_dec")
                etrue_idx = ForwardVariableDef("etrue_idx", "int")
                dec_idx = ForwardVariableDef("dec_idx", "int")
                ereco_hist_idx = ForwardVariableDef("ereco_hist_idx", "int")
                etrue_idx << FunctionCall(["log10(true_energy)"], "etrue_lookup")
                #StringExpression(['print("Etrue ", log10(true_energy))'])
                with IfBlockContext(["etrue_idx == 0 || etrue_idx > 14"]):
                    StringExpression(['reject("True energy outside IRF range", true_energy)'])
                with ElseBlockContext():
                    dec_idx << FunctionCall(["declination"], "dec_lookup")
                    ereco_hist_idx << FunctionCall([etrue_idx, dec_idx], "ereco_get_ragged_index")
                    #StringExpression(['print("etrue_idx ", etrue_idx)'])
                    #StringExpression(['print("dec_idx ", dec_idx)'])
                    #StringExpression(['print("ereco_hist_idx ", ereco_hist_idx)'])
                    #return is log(E/GeV)
                    ReturnStatement([FunctionCall([FunctionCall([ereco_hist_idx], "ereco_get_ragged_hist"), FunctionCall([ereco_hist_idx], "ereco_get_ragged_edges")], "histogram_rng")])
                    #ReturnStatement([FunctionCall([FunctionCall(["true_energy", "declination", "0"], "ereco_lookup"), FunctionCall(["true_energy", "declination", "1"], "ereco_lookup")], "histogram_rng")])

                #append lookup for inputted energy -> already done, etrue_ind
                #append lookup for outputted energy
                # -> use as input for angular stuff
                """
                #use alternatively for testing the lognormal approach
                lognorm = LognormalMixture(self.mixture_name, self.n_components, self.mode)
                truncated_e = TruncatedParameterization("true_energy", *self._poly_limits)
                log_trunc_e = LogParameterization(truncated_e)

                #self._poly_params_mu should have shape (3, 3, 6)
                #3: declination bins, 3: components, 6: poly-coeffs
                mu_poly_coeffs = StanArray(
                    "R2021EnergyResolutionMuPolyCoeffs",
                    "real",
                    self._poly_params_mu,
                )
                #same as above
                sd_poly_coeffs = StanArray(
                    "R2021EnergyResolutionSdPolyCoeffs",
                    "real",
                    self._poly_params_sd,
                )

                mu = ForwardArrayDef("mu_e_res", "real", ["[", self._n_components, "]"])
                sigma = ForwardArrayDef(
                    "sigma_e_res", "real", ["[", self._n_components, "]"]
                )

                weights = ForwardVariableDef(
                    "weights", "vector[" + str(self._n_components) + "]"
                )

                declination = ForwardVariableDef("declination", "real")
                declination << FunctionCall(["omega"], "omega_to_dec")

                declination_bins = StanArray("dec_bins", "real", self._declination_bins)
                declination_index = ForwardVariableDef("dec_ind", "int")
                declination_index << StringExpression(["binary_search(declination, ", declination_bins, ")"])

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

                #log_reco_e = LogParameterization("reco_energy")
                ReturnStatement([lognorm(log_mu_vec, sigma_vec, weights)])


        else:
            raise RuntimeError("mode must be DistributionMode.PDF or DistributionMode.RNG")
        '''

    def setup(self) -> None:

        if self.gen_type == "lognorm":
            self.fit_params = []
            self._eres = []
            self._rE_bin_edges = []
            self._rE_binc = []
            self._generate_ragged_ereco_data(self.irf)
        
            # Check cache
            if self.CACHE_FNAME_LOGNORM in Cache and not self._rewrite:
                logger.info("Loading energy pdf data from file.")
                with Cache.open(self.CACHE_FNAME_LOGNORM, "rb") as fr:
                    data = np.load(fr)
                    self._eres = data["eres"]
                    self._tE_bin_edges = data["tE_bin_edges"]
                    self._rE_bin_edges = data["rE_bin_edges"]
                    self._poly_params_mu = data["poly_params_mu"]
                    self._poly_params_sd = data["poly_params_sd"]
                    self._poly_limits = (float(data["Emin"]), float(data["Emax"]))

            else:
                logger.info("Re-doing energy lognorm data and saving files.")
                tE_bin_edges = np.power(10, self.irf.true_energy_bins)
                tE_binc = 0.5 * (tE_bin_edges[:-1] + tE_bin_edges[1:])
                
                for c_dec, (dec_low, dec_high) in enumerate(
                    zip(self._declination_bins[:-1], self._declination_bins[1:])
                ):

                    # Find common rE binning per declination
                    _min = 10
                    _max = 0
                    _diff = 0

                    for bins in self.irf.reco_energy_bins[:, c_dec]:
                        c_min = bins.min()
                        c_max = bins.max()
                        c_diff = np.diff(bins).max()
                        _min = c_min if c_min < _min else _min
                        _max = c_max if c_max > _max else _max
                        _diff = c_diff if c_diff > _diff else _diff

                    # use bins encompassing the entire range
                    # round number to next integer, should that be +1 bc bin edges and not bins?
                    num_bins = int(np.ceil((_max - _min) / _diff))
                    rE_bin_edges = np.logspace(_min, _max, num_bins)
                    self._rE_bin_edges.append(rE_bin_edges)
                    rE_binc = 0.5 * (rE_bin_edges[:-1] + rE_bin_edges[1:])
                    self._rE_binc.append(rE_binc)
                    print("rE_binc.shape", rE_binc.shape)
                    # Normalize along Ereco
                    # log10 because rE_bin_edges live in linear space
                    # weird choice to me, everything else with these pdfs is done in log space
                    bin_width = np.log10(rE_bin_edges[1]) - np.log10(rE_bin_edges[0])
                    eres = np.zeros((tE_bin_edges.size-1, rE_bin_edges.size-1))
                    for i, pdf in enumerate(self.irf.reco_energy[:, c_dec]):
                        for j, (elow, ehigh) in enumerate(zip(
                            rE_bin_edges[:-1], rE_bin_edges[1:]
                            )
                        ):
                            eres[i, j] = pdf.cdf(np.log10(ehigh)) - pdf.cdf(np.log10(elow))
                        #eres should already be normalised bc the pdfs (or rather cdfs)
                        #of which the values are taken are already noralised, nevertheless:
                        eres[i, :] = eres[i, :] / np.sum(eres[i] * bin_width)
                    self._eres.append(eres)
                
                    # do not rebin -> rebin=1
                    fit_params, rebin_tE_binc = self._fit_energy_res(
                        tE_binc, rE_binc, eres, self._n_components, rebin=1
                    )
                    self.fit_params.append(fit_params)                    
                    self.rebin_tE_binc = rebin_tE_binc
                    # take entire range
                    imin = 0
                    imax = -1

                    # I get that Emin, Emax for fitting might be the bin centers
                    # but why for the truncated parameterisation?
                    # the histogram data covers all the bin, not just up to/down to the center of last/first bin
                    
                    #Emin = rebin_tE_binc[imin]
                    #Emax = rebin_tE_binc[imax]
                    Emin = tE_bin_edges[imin]
                    Emax = tE_bin_edges[imax]

                    # Fit polynomial:
                    poly_params_mu, poly_params_sd, poly_limits = self._fit_polynomial(
                        fit_params, rebin_tE_binc, Emin, Emax, polydeg=5
                    )
                    
                    self._poly_params_mu.append(poly_params_mu)
                    self._poly_params_sd.append(poly_params_sd)
                    self._poly_limits_battery.append(poly_limits)
                    #self._poly_params_mu = poly_params_mu
                    #self._poly_params_sd = poly_params_sd
                    #self._poly_limits = poly_limits
                    #break

                #find smallest range of poly limits to use globally
                
                poly_low = [i[0] for i in self._poly_limits_battery]
                poly_high = [i[1] for i in self._poly_limits_battery]
                poly_limits = (max(poly_low), min(poly_high))
                
                # Save values
                self._poly_limits = poly_limits
                self._tE_bin_edges = tE_bin_edges
                #self._rE_bin_edges = rE_bin_edges
                

                # Save polynomial

                with Cache.open(self.CACHE_FNAME_LOGNORM, "wb") as fr:

                    np.savez(
                        fr,
                        eres=eres,
                        tE_bin_edges=tE_bin_edges,
                        rE_bin_edges=rE_bin_edges,
                        poly_params_mu=self._poly_params_mu,
                        poly_params_sd=self._poly_params_sd,
                        Emin=Emin,
                        Emax=Emax,
                    )
            self._poly_params_mu__ = self.poly_params_mu
            self._poly_params_sd__ = self.poly_params_sd
            self._eres__ = self._eres

            
            for c, dec in enumerate(self._declination_bins[:-1]):
                self.set_fit_params(dec+0.01)
                fig = self.plot_fit_params(self.fit_params[c], self.rebin_tE_binc)
                fig.savefig(f"/Users/David/Documents/phd/icecube/hi_nu_plots/fit_params_{c}.png")
                fig = self.plot_parameterizations(
                    tE_binc,
                    self._rE_binc[c],
                    self.fit_params[c],
                    #rebin_tE_binc=rebin_tE_binc,
                )
                fig.savefig(f"/Users/David/Documents/phd/icecube/hi_nu_plots/parameterisation_{c}.png")
            self._poly_params_mu = self._poly_params_mu__
            self._poly_params_sd = self._poly_params_sd__
            
        else:
            # Check cache
            if self.CACHE_FNAME_HISTOGRAM in Cache and not self._rewrite:
                logger.info("Loading energy pdf data from file.")
                
                with Cache.open(self.CACHE_FNAME_HISTOGRAM, "rb") as fr:
                    data = np.load(fr)
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
                        bins = self._ereco_edges,
                        values = self._ereco_hist,
                        num_of_bins = self._ereco_num_edges,
                        num_of_values = self._ereco_num_vals,
                        cum_num_of_bins = self._ereco_cum_num_edges,
                        cum_num_of_values = self._ereco_cum_num_vals,
                        tE_bin_edges = self._tE_bin_edges,
                    )
            
            self._Emin = np.power(10, self.irf.true_energy_bins[0])
            self._Emax = np.power(10, self.irf.true_energy_bins[-1])
        


    @classmethod
    def rewrite_files(cls):
        #call this to rewrite npz files
        cls(DistributionMode.PDF, rewrite=True)
        cls(DistributionMode.RNG, rewrite=True)


    def set_fit_params(self, dec):
        """
        Used in `sim_interface.py`
        """
        dec_idx = np.digitize(dec, self._declination_bins) - 1
        if dec == np.pi / 2:
            dec_idx -= 1
        
        self._poly_params_mu = self._poly_params_mu__[dec_idx]
        self._poly_params_sd = self._poly_params_sd__[dec_idx]
        self._eres = self._eres__[dec_idx]


    # def marginalise_over_dec(self):
    #     cos_theta_bins = np.cos(np.pi / 2 - self._declination_bins)



class R2021AngularResolution(AngularResolution, HistogramSampler):
    """
    Angular resolution for Northern Tracks Sample

    Data from https://arxiv.org/pdf/1811.07979.pdf
    Fits a polynomial to the median angular resolution converted to
    `kappa` parameter of a VMF distribution

    Attributes:
        poly_params: Coefficients of the polynomial
        e_min: Lower energy bound of the polynomial
        e_max: Upper energy bound of the polynomial

    """

    local_path = "input/tracks/IC86_II_smearing.csv"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "angular_reso_r2021.npz"
    #only one file here for the rng data

    def __init__(self, mode: DistributionMode = DistributionMode.PDF, rewrite: bool = False) -> None:

        self.irf = R2021IRF()
        self.mode = mode
        self._rewrite = rewrite
        logger.info("Forced angular rewriting: {}".format(rewrite))

        if mode == DistributionMode.PDF:
            self._func_name = "R2021AngularResolution"

        else:
            self._func_name = "R2021AngularResolution_rng"

        #self._kappa_grid: np.ndarray = None
        #self._Egrid: np.ndarray = None
        #self._poly_params: Sequence = []
        #TODO check for lin/log scale of energy
        self._Emin: float = float("nan")
        self._Emax: float = float("nan")

        self.setup()



    def generate_code(self):

        if self.mode == DistributionMode.PDF:
            #TODO check pdf signature
            super().__init__(
                "R2021AngularResolution",
                ["true_dir", "reco_dir", "kappa"],
                ["vector", "vector", "real"],
                "real",
            )

        else:

            super().__init__(
                "R2021AngularResolution_rng",
                ["true_energy", "reco_energy", "true_dir"],
                ["real", "real", "vector"],
                "vector",
            )

        # Define Stan interface
        with self:

            # Clip true energy
            #clipped_e = TruncatedParameterization("true_energy", self._Emin, self._Emax)

            #clipped_log_e = LogParameterization(clipped_e)
            """
            kappa = PolynomialParameterization(
                clipped_log_e,
                self._poly_params,
                "NorthernTracksAngularResolutionPolyCoeffs",
            )
            """
            if self.mode == DistributionMode.PDF:
                # VMF expects x_obs, x_true -> true i.e. test source
                vmf = VMFParameterization(["reco_dir", "true_dir"], "kappa", self.mode)
                #calculation of kappa goes here
                #use approximate formulas from icecube_tools
                ReturnStatement([vmf])
                


            elif self.mode == DistributionMode.RNG:
                vmf = VMFParameterization(["true_dir"], "kappa", self.mode)
                self._make_histogram("psf", self._psf_hist, self._psf_edges)
                self._make_psf_hist_index()
                for name, array in zip(["psf_get_cum_num_vals", "psf_get_cum_num_edges",
                    "psf_get_num_vals", "psf_get_num_edges"],
                    [self._psf_cum_num_vals, self._psf_cum_num_edges, self._psf_num_vals, self._psf_num_edges]
                ):
                    self._make_lookup_functions(name, array)
                
                self._make_histogram("ang", self._ang_hist, self._ang_edges)
                self._make_ang_hist_index()
                for name, array in zip(["ang_get_cum_num_vals", "ang_get_cum_num_edges",
                    "ang_get_num_vals", "ang_get_num_edges"],
                    [self._ang_cum_num_vals, self._ang_cum_num_edges, self._ang_num_vals, self._ang_num_edges]
                ):
                    self._make_lookup_functions(name, array)
                
                #convert input energies to logs
                #log_etrue = ForwardVariableDef("log_etrue", "real")
                #log_etrue = LogParameterization("true_energy")

                log_ereco = LogParameterization("reco_energy")
                
                #get ereco index from eres-defined functions
                etrue_idx = ForwardVariableDef("etrue_idx", "int")
                etrue_idx << FunctionCall(["log10(true_energy)"], "etrue_lookup")
                #StringExpression(['print("etrue_idx ", etrue_idx)'])
                declination = ForwardVariableDef("declination", "real")
                declination << FunctionCall(["true_dir"], "omega_to_dec")
                dec_idx = ForwardVariableDef("dec_idx ", "int")
                dec_idx << FunctionCall(["declination"], "dec_lookup")
                ereco_hist_idx = ForwardVariableDef("ereco_hist_idx", "int")
                ereco_hist_idx << FunctionCall([etrue_idx, dec_idx], "ereco_get_ragged_index")
                ereco_idx = ForwardVariableDef("ereco_idx", "int")
                ereco_idx << FunctionCall([log_ereco, FunctionCall([ereco_hist_idx], "ereco_get_ragged_edges")], "binary_search")
                #StringExpression(['print("ereco_idx ", ereco_idx)'])

                #lookup psf stuff
                psf_hist_idx = ForwardVariableDef("psf_hist_idx", "int")
                psf_hist_idx << FunctionCall([etrue_idx, dec_idx, ereco_idx], "psf_get_ragged_index")
                #StringExpression(['print("psfhistidx ", psf_hist_idx)'])
                #StringExpression(["print(", FunctionCall([psf_hist_idx], "psf_get_ragged_hist"), ")"])
                #StringExpression(["print(", FunctionCall([psf_hist_idx], "psf_get_ragged_edges"), ")"])
                #call histogramm with appropriate values/edges
                psf_idx = ForwardVariableDef("psf_idx", "int")
                psf_idx << FunctionCall([FunctionCall([psf_hist_idx], "psf_get_ragged_hist"), FunctionCall([psf_hist_idx], "psf_get_ragged_edges")], "hist_cat_rng")
                #StringExpression(['print("psfidx", psf_idx)'])
                
                #lookup ang err stuff
                ang_hist_idx = ForwardVariableDef("ang_hist_idx", "int")
                ang_hist_idx << FunctionCall([etrue_idx, dec_idx, ereco_idx, psf_idx], "ang_get_ragged_index")
                #StringExpression(['print("anghist", ang_hist_idx)'])
                #StringExpression(["print(", FunctionCall([ang_hist_idx], "ang_get_ragged_hist"), ")"])
                #StringExpression(["print(", FunctionCall([ang_hist_idx], "ang_get_ragged_edges"), ")"])
                ang_err = ForwardVariableDef("ang_err", "real")
                ang_err << FunctionCall([FunctionCall([ang_hist_idx], "ang_get_ragged_hist"), FunctionCall([ang_hist_idx], "ang_get_ragged_edges")], "histogram_rng")
                
                kappa = ForwardVariableDef("kappa", "real")
                #hardcoded p=0.5 (log(1-p)) from the tabulated data of release
                kappa << StringExpression(["- (2 / (pi() * pow(10, ang_err) / 180)^2) * log(1 - 0.5)"])
                #StringExpression(["print(ang_err)"])
                #StringExpression(["print(kappa)"])
                return_vec = ForwardVectorDef("return_this", [4])
                StringExpression(["return_this[1:3] = ", vmf])
                StringExpression(["return_this[4] = kappa"])
                ReturnStatement([return_vec])

            #ReturnStatement([ang_err])


    def setup(self):
        self._pdet_limits = (1e2, 1e8)
        self._Emin, self._Emax = self._pdet_limits

        if self.mode == DistributionMode.PDF:
            #what needs to be set up?
            #only thing needed is gaussian/vMF for reconstruction
            #that's done in init itself
            pass

        elif self.mode == DistributionMode.RNG:
            #party in the back
            #extract *all* the histograms bar ereco
            #check for loading of data
            if self.CACHE_FNAME in Cache and not self._rewrite:
                logger.info("Loading angular data from file.")
                with Cache.open(self.CACHE_FNAME, "rb") as fr:
                    data = np.load(fr)
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
                        psf_cum_num_edges = self._psf_cum_num_edges,
                        psf_cum_num_vals = self._psf_cum_num_vals,
                        psf_num_vals = self._psf_num_vals,
                        psf_num_edges = self._psf_num_edges,
                        psf_vals = self._psf_hist,
                        psf_edges = self._psf_edges,
                        ang_edges = self._ang_edges,
                        ang_vals = self._ang_hist,
                        ang_num_vals = self._ang_num_vals,
                        ang_num_edges = self._ang_num_edges,
                        ang_cum_num_vals = self._ang_cum_num_vals,
                        ang_cum_num_edges = self._ang_cum_num_edges,
                    )

        else:
            raise ValueError("You weren't supposed to do that.")


    def kappa(self):
        pass


    @classmethod
    def rewrite_files(cls):
        #call this to rewrite npz files
        cls(DistributionMode.RNG, rewrite=True)
    


class R2021DetectorModel(DetectorModel):

    RNG_FILENAME = "r2021_rng.stan"
    PDF_FILENAME = "r2021_pdf.stan"

    event_types = ["tracks"]

    logger = logging.getLogger(__name__+".R2021DetectorModel")
    logger.setLevel(logging.INFO)

    def __init__(self,
        mode: DistributionMode = DistributionMode.PDF,
        event_type = None,
        rewrite = False,
        gen_type = "histogram"):

        super().__init__(mode, event_type="tracks")

        #ang_res = R2021AngularResolution(mode, rewrite)
        #self._angular_resolution = ang_res

        energy_res = R2021EnergyResolution(mode, rewrite, gen_type)
        self._energy_resolution = energy_res

        #if mode == DistributionMode.PDF:
        #self._eff_area = R2021EffectiveArea()


    def _get_effective_area(self):
        return self._eff_area


    def _get_energy_resolution(self):
        return self._energy_resolution


    def _get_angular_resolution(self):
        return self._angular_resolution


    @classmethod
    def generate_code(cls, mode: DistributionMode, rewrite: bool = False, gen_type: str = "histogram"):
        cls.logger.info("Generating r2021 stan code.")
        with StanGenerator() as cg:
            instance = cls(mode=mode, rewrite=rewrite, gen_type=gen_type)
            #instance.effective_area.generate_code()
            #instance.angular_resolution.generate_code()
            instance.energy_resolution.generate_code()
            code = cg.generate()
        code = code.removeprefix("functions\n{")
        code = code.removesuffix("\n}\n")
        if not os.path.isdir(STAN_GEN_PATH):
            os.makedirs(STAN_GEN_PATH)
        if mode == DistributionMode.PDF:
            with open(os.path.join(STAN_GEN_PATH, cls.PDF_FILENAME), "w+") as f:
                f.write(code)
            #with Cache.open(cls.PDF_FILENAME, "w+") as f:
            #    f.write(code)
        else:
            with open(os.path.join(STAN_GEN_PATH, cls.RNG_FILENAME), "w+") as f:
                f.write(code)
            #with Cache.open(cls.RNG_FILENAME, "w+") as f:
            #    f.write(code)
        #with open(os.path.join(STAN_PATH, "r2021.stan"), 'w+') as f:
        #    f.write(code)
