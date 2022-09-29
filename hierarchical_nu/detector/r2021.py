from typing import Sequence, Tuple, Iterable
import os
import pandas as pd
import numpy as np
import sys

from icecube_tools.detector.r2021 import R2021IRF

from hierarchical_nu.backend.stan_generator import IfBlockContext

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
Cache.set_cache_dir(".cache")


class HistogramSampler():
    def __init__(self):
        pass


    def _generate_ragged_ereco_data(self, irf: R2021IRF):
        """
        Should take the R2021 icecube_tools irf and operate on its arrays.
        """
        #TODO abstract names
        num_of_bins = []
        num_of_values = []
        cum_num_of_values = []
        cum_num_of_bins = []
        indices = np.zeros((14, 3), dtype=int)
        counter = 0
        # irf = R2021IRF()   # substitute by provided irf later
        for c_e, etrue in enumerate(irf.true_energy_values):
            #inner loop is declination
            for c_d, dec in enumerate(irf.declination_bins[:-1]):
                n, b = irf._marginalisation(c_e, c_d)
                if counter != 0:
                    bins = np.concatenate((bins, b))
                    values = np.concatenate((values, n))
                else:
                    bins = b.copy()
                    values = n.copy()
                num_of_values.append(n.size)
                num_of_bins.append(b.size)
                if counter != 0:
                    cum_num_of_values.append(cum_num_of_values[-1]+n.size)
                    cum_num_of_bins.append(cum_num_of_bins[-1]+b.size)
                else:
                    cum_num_of_values.append(n.size)
                    cum_num_of_bins.append(b.size)
                indices[c_e, c_d] = counter
                counter += 1
        self._cum_num_values = cum_num_of_values
        self._cum_num_edges = cum_num_of_bins
        self._num_values = num_of_values
        self._num_edges = num_of_bins
        self._ereco_hist = values
        self._ereco_edges = bins


    def _make_histogram(self, data_type: str, hist_values: Iterable[float], hist_bins: Iterable[float]):
        #TODO needs to be abstracted in name
        #this should be a funcion with the edges below
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
        #TODO needs to be abstracted for use with angles
        #another function for the specific histogram
        self._get_ragged_index = UserDefinedFunction("ereco_get_ragged_index", ["etrue", "dec"], ["int", "int"], "int")
        # Takes indices of etrue and dec (to be determined elsewhere!)
        with self._get_ragged_index:
            ReturnStatement(["3 * etrue + dec - 3"])


    def _make_lookup_functions(self, name, array):
        #DONE
        #look-up functions for ragged arrays
        f = UserDefinedFunction(name, ["idx"], ["int"], "int")
        with f:
            arr = StanArray("arr", "int", array)
            ReturnStatement(["arr[idx]"])

        
    def _make_ragged_start_stop(self, data, hist):
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

        super().__init__(
            "R2021EffectiveArea",
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real",
        )

        self.setup()

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

            aeff = EffectiveArea.from_dataset("20210126")
            eff_area = aeff.values
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
        self._rs_bbpl_params["gamma2_scale"] = 1.2

        

class R2021EnergyResolution(EnergyResolution, HistogramSampler):

    """
    Energy resolution for Northern Tracks Sample

    Data from https://arxiv.org/pdf/1811.07979.pdf
    """

    local_path = "input/tracks/IC86_II_smearing.csv"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "energy_reso_r2021.npz"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:
        """
        Args:
            inputs: List[TExpression]
                First item is true energy, second item is reco energy
        """

        self.irf = R2021IRF()
        #super().__init__()


        #TODO edit type hints later when everything else works
        self._mode = mode
        self._poly_params_mu: Sequence = []
        self._poly_params_sd: Sequence = []
        self._poly_limits: Tuple[float, float] = (float("nan"), float("nan"))
        self._poly_limits_battery: Sequence = []
        self._declination_bins = self.irf.declination_bins

        # For prob_Edet_above_threshold
        self._pdet_limits = (1e2, 1e8)

        self._n_components = 3
        self.setup()

        # need to change rng mode to actually be sampling from the histogram
        if mode == DistributionMode.PDF:
            mixture_name = "r2021_energy_res_mix"
            super().__init__(
                "R2021EnergyResolution",
                ["true_energy", "reco_energy", "declination"],
                ["real", "real", "real"],
                "real",
            )
            lognorm = LognormalMixture(mixture_name, self.n_components, self._mode)
        elif mode == DistributionMode.RNG:
            mixture_name = "r2021_energy_res_mix_rng"
            super().__init__(
                "R2021EnergyResolution_rng",
                ["true_energy", "declination"],
                ["real", "real"],
                "real",
            )
        else:
            raise RuntimeError("mode must be DistributionMode.PDF or DistributionMode.RNG")

        # Define Stan interface
        if self._mode == DistributionMode.PDF:
            with self:
                truncated_e = TruncatedParameterization("true_energy", *self._poly_limits)
                log_trunc_e = LogParameterization(truncated_e)

                #self._poly_params_mu should have shape (3, 3, 6)
                #3: declination bins, 3: components, 6: poly-coeffs
                mu_poly_coeffs = StanArray(
                    "NorthernTracksEnergyResolutionMuPolyCoeffs",
                    "real",
                    self._poly_params_mu,
                )
                #same as above
                sd_poly_coeffs = StanArray(
                    "NorthernTracksEnergyResolutionSdPolyCoeffs",
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

                log_reco_e = LogParameterization("reco_energy")
                ReturnStatement([lognorm(log_reco_e, log_mu_vec, sigma_vec, weights)])

        elif mode == DistributionMode.RNG:
            with self:
                #histogram "histogram_rng(array[] real hist_array, array[] real hist_edges)"
                #is defined in utils.stan, is included anway
                self._generate_ragged_ereco_data(self.irf)
                self._make_histogram("ereco", self._ereco_hist, self._ereco_edges)
                self._make_ereco_hist_index()
                for name, array in zip(["ereco_get_cum_num_vals", "ereco_get_cum_num_edges",
                    "ereco_get_num_vals", "ereco_get_num_edges"],
                    [self._cum_num_values, self._cum_num_edges, self._num_values, self._num_edges]
                ):
                    self._make_lookup_functions(name, array)

                #TODO truncate or not?
                truncated_e = TruncatedParameterization("true_energy", *self._poly_limits)
                log_trunc_e = LogParameterization(truncated_e)

                #do binary search for bin of true energy
                etrue_ind = ForwardVariableDef("etrue_ind", "int")
                etrue_bins = StanArray("log_etrue_bins", "real", np.log10(self._tE_bin_edges))
                etrue_ind << StringExpression(["binary_search(", log_trunc_e, ", ", etrue_bins, ")"])

                #do binary search for bin of declination
                declination_bins = StanArray("dec_bins", "real", self._declination_bins)
                dec_ind = ForwardVariableDef("dec_ind", "int")
                dec_ind << StringExpression(["binary_search(declination, ", declination_bins, ")"])

                #find appropriate section in ragged array structure
                hist_ind = ForwardVariableDef("hist_ind", "int")
                hist_ind << FunctionCall([etrue_ind, dec_ind], "ereco_get_ragged_index")

                #call histogramm with appropriate values/edges
                ReturnStatement([FunctionCall([FunctionCall([hist_ind], "ereco_get_ragged_hist"), FunctionCall([hist_ind], "ereco_get_ragged_edges")], "histogram_rng")])

                #append lookup for inputted energy -> already done, etrue_ind
                #append lookup for outputted energy
                # -> use as input for angular stuff


    def setup(self) -> None:
     
        # Check cache
        if self.CACHE_FNAME in Cache:

            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                data = np.load(fr)
                self._eres = data["eres"]
                self._tE_bin_edges = data["tE_bin_edges"]
                self._rE_bin_edges = data["rE_bin_edges"]
                self._poly_params_mu = data["poly_params_mu"]
                self._poly_params_sd = data["poly_params_sd"]
                self._poly_limits = (float(data["Emin"]), float(data["Emax"]))

        # Or load from file
        else:

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
                rE_binc = 0.5 * (rE_bin_edges[:-1] + rE_bin_edges[1:])

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

            
                # do not rebin -> rebin=1
                fit_params, rebin_tE_binc = self._fit_energy_res(
                    tE_binc, rE_binc, eres, self._n_components, rebin=1
                )

                # take entire range
                imin = 0
                imax = -1

                # I get that Emin, Emax for fitting might be the bin centers
                # but why for the truncated parameterisation?
                # the histogram data covers all the bin, not just up to/down to the center of last/first bin
                Emin = rebin_tE_binc[imin]
                Emax = rebin_tE_binc[imax]

                # Fit polynomial:
                poly_params_mu, poly_params_sd, poly_limits = self._fit_polynomial(
                    fit_params, rebin_tE_binc, Emin, Emax, polydeg=5
                )
                self._poly_params_mu.append(poly_params_mu)
                self._poly_params_sd.append(poly_params_sd)
                self._poly_limits_battery.append(poly_limits)

            #find smallest range of poly limits to use globally
            poly_low = [i[0] for i in self._poly_limits_battery]
            poly_high = [i[1] for i in self._poly_limits_battery]
            poly_limits = (max(poly_low), min(poly_high))

            # Save values
            self._poly_limits = poly_limits
            self._eres = eres
            self._tE_bin_edges = tE_bin_edges
            self._rE_bin_edges = rE_bin_edges

            

            # Save polynomial

            with Cache.open(self.CACHE_FNAME, "wb") as fr:

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

            # Show results
            # this isn't working properly right now, why though?
            """
            self.plot_fit_params(fit_params, rebin_tE_binc)
            self.plot_parameterizations(
                tE_binc,
                rE_binc,
                fit_params,
                rebin_tE_binc=rebin_tE_binc,
            )
            """



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

    local_path = "input/tracks/NorthernTracksAngularRes.csv"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "angular_reso_r2021.npz"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:

        self.irf = R2021IRF()

        if mode == DistributionMode.PDF:

            super().__init__(
                "NorthernTracksAngularResolution",
                ["true_energy", "true_dir", "reco_dir"],
                ["real", "vector", "vector"],
                "real",
            )

        else:

            super().__init__(
                "NorthernTracksAngularResolution_rng",
                ["true_energy", "true_dir"],
                ["real", "vector"],
                "vector",
            )

        self._kappa_grid: np.ndarray = None
        self._Egrid: np.ndarray = None
        self._poly_params: Sequence = []
        self._Emin: float = float("nan")
        self._Emax: float = float("nan")

        self.setup()

        # Define Stan interface
        with self:

            # Clip true energy
            clipped_e = TruncatedParameterization("true_energy", self._Emin, self._Emax)

            clipped_log_e = LogParameterization(clipped_e)

            kappa = PolynomialParameterization(
                clipped_log_e,
                self._poly_params,
                "NorthernTracksAngularResolutionPolyCoeffs",
            )

            if mode == DistributionMode.PDF:
                # VMF expects x_obs, x_true
                vmf = VMFParameterization(["reco_dir", "true_dir"], kappa, mode)

            elif mode == DistributionMode.RNG:
                vmf = VMFParameterization(["true_dir"], kappa, mode)

            ReturnStatement([vmf])

    def kappa(self):

        clipped_e = TruncatedParameterization("E[i]", self._Emin, self._Emax)

        clipped_log_e = LogParameterization(clipped_e)

        kappa = PolynomialParameterization(
            clipped_log_e,
            self._poly_params,
            "NorthernTracksAngularResolutionPolyCoeffs",
        )

        return kappa

    def setup(self) -> None:

        # Check cache
        if self.CACHE_FNAME in Cache:

            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                data = np.load(fr)
                self._kappa_grid = data["kappa_grid"]
                self._Egrid = data["Egrid"]
                self._poly_params = data["poly_params"]
                self._Emin = float(data["Emin"])
                self._Emax = float(data["Emax"])
                self._poly_limits = (self._Emin, self._Emax)

        else:

            # Load input data and fit polynomial
            if not os.path.exists(self.DATA_PATH):

                raise RuntimeError(self.DATA_PATH, "is not a valid path")

            data = pd.read_csv(
                self.DATA_PATH,
                sep=";",
                decimal=",",
                header=None,
                names=["log10energy", "resolution"],
            )

            # Kappa parameter of VMF distribution
            data["kappa"] = 1.38 / np.radians(data.resolution) ** 2

            self._kappa_grid = data.kappa.values
            self._Egrid = 10**data.log10energy.values
            self._poly_params = np.polyfit(
                data.log10energy.values, data.kappa.values, 5
            )
            self._Emin = 10 ** float(data.log10energy.min())
            self._Emax = 10 ** float(data.log10energy.max())
            self._poly_limits = (self._Emin, self._Emax)

            # Save polynomial
            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                np.savez(
                    fr,
                    kappa_grid=self._kappa_grid,
                    Egrid=self._Egrid,
                    poly_params=self._poly_params,
                    Emin=self._Emin,
                    Emax=self._Emax,
                )


class NorthernTracksDetectorModel(DetectorModel):
    """
    Implements the detector model for the NT sample

    Parameters:
        mode: DistributionMode
            Set mode to either RNG or PDF

    """

    event_types = ["tracks"]

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        event_type=None,
    ):

        super().__init__(mode, event_type="tracks")

        ang_res = NorthernTracksAngularResolution(mode)
        self._angular_resolution = ang_res

        energy_res = NorthernTracksEnergyResolution(mode)
        self._energy_resolution = energy_res

        if mode == DistributionMode.PDF:
            self._eff_area = NorthernTracksEffectiveArea()

    def _get_effective_area(self):
        return self._eff_area

    def _get_energy_resolution(self):
        return self._energy_resolution

    def _get_angular_resolution(self):
        return self._angular_resolution


# Testing.
# @TODO: This needs updating.
if __name__ == "__main__":

    e_true_name = "e_true"
    e_reco_name = "e_reco"
    true_dir_name = "true_dir"
    reco_dir_name = "reco_dir"
    # ntp = NorthernTracksAngularResolution([e_true, pos_true])

    # print(ntp.to_stan())
    from backend.stan_generator import (
        StanGenerator,
        GeneratedQuantitiesContext,
        DataContext,
        FunctionsContext,
        Include,
    )

    logging.basicConfig(level=logging.DEBUG)
    import pystan

    with StanGenerator() as cg:

        with FunctionsContext() as fc:
            Include("utils.stan")
            Include("vMF.stan")

        with DataContext() as dc:
            e_true = ForwardVariableDef(e_true_name, "real")
            e_reco = ForwardVariableDef(e_reco_name, "real")
            true_dir = ForwardVariableDef(true_dir_name, "vector[3]")
            reco_dir = ForwardVariableDef(reco_dir_name, "vector[3]")

        with GeneratedQuantitiesContext() as gq:
            ntd = NorthernTracksDetectorModel()

            """
            ang_res_result = ForwardVariableDef("ang_res", "real")
            ang_res = FunctionCall(
                [e_reco], ntd.angular_resolution, 1)
            ang_res_result = AssignValue(
                [ang_res], ang_res_result)

            """
            e_res_result = ForwardVariableDef("e_res", "real")
            e_res_result << ntd.energy_resolution(e_true, e_reco)

            ang_res_result = ForwardVariableDef("ang_res", "real")
            ang_res_result << ntd.angular_resolution(e_true, true_dir, reco_dir)

            eff_area_result = ForwardVariableDef("eff_area", "real")
            eff_area_result << ntd.effective_area(e_true, true_dir)

        model = cg.generate()

    print(model)
    this_dir = os.path.abspath("")
    sm = pystan.StanModel(
        model_code=model,
        include_paths=[
            os.path.join(
                this_dir, "../dev/statistical_model/4_tracks_and_cascades/stan/"
            )
        ],  # noqa: E501
        verbose=False,
    )

    dir1 = np.array([1, 0, 0])
    dir2 = np.array([1, 0.1, 0])
    dir2 /= np.linalg.norm(dir2)

    data = {"e_true": 1e5, "e_reco": 1e5, "true_dir": dir1, "reco_dir": dir2}
    fit = sm.sampling(data=data, iter=1, chains=1, algorithm="Fixed_param")
    print(fit)