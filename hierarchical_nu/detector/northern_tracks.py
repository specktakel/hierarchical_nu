from typing import Sequence, Tuple
import os
import pandas as pd
import numpy as np

from ..utils.cache import Cache
from ..backend import (
    VMFParameterization,
    PolynomialParameterization,
    TruncatedParameterization,
    LogParameterization,
    SimpleHistogram,
    ReturnStatement,
    FunctionCall,
    DistributionMode,
    LognormalMixture,
    ForLoopContext,
    ForwardVariableDef,
    ForwardArrayDef,
    StanArray,
    StringExpression,
    TwoDimHistInterpolation,
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


class NorthernTracksEffectiveArea(EffectiveArea):
    """
    Effective area for the two-year Northern Tracks release:
    https://icecube.wisc.edu/science/data/HE_NuMu_diffuse
    """

    local_path = "input/tracks/effective_area.h5"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "aeff_tracks.npz"

    NAME = "NorthernTracksEffectiveArea"

    def __init__(self) -> None:
        super().__init__(
            self.NAME,
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real",
        )

        self.setup()

        self._make_spline()

        # Define Stan interface.
        with self:
            hist = TwoDimHistInterpolation(
                self._eff_area,
                [self._tE_bin_edges, self._cosz_bin_edges],
                "NorthernTracksEffAreaHist",
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
            import h5py  # type: ignore

            with h5py.File(self.DATA_PATH, "r") as f:
                aeff_numu = f["2010/nu_mu/area"][()]
                aeff_numubar = f["2010/nu_mu_bar/area"][()]

                # Sum over reco energy and average numu/numubar
                eff_area = 0.5 * (aeff_numu.sum(axis=2) + aeff_numubar.sum(axis=2))

                # True Energy [GeV]
                tE_bin_edges = f["2010/nu_mu/bin_edges_0"][:]

                # cos(zenith)
                cosz_bin_edges = f["2010/nu_mu/bin_edges_1"][:]

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


class NorthernTracksEnergyResolution(EnergyResolution):

    """
    Energy resolution for Northern Tracks Sample

    Data from https://arxiv.org/pdf/1811.07979.pdf
    """

    local_path = "input/tracks/effective_area.h5"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "energy_reso_tracks.npz"

    RNG_NAME = "NorthernTracksEnergyResolution_rng"
    PDF_NAME = "NorthernTracksEnergyResolution"

    def __init__(
        self, mode: DistributionMode = DistributionMode.PDF, make_plots: bool = False
    ) -> None:
        """
        Args:
            inputs: List[TExpression]
                First item is true energy, second item is reco energy
        """

        self._mode = mode
        self._poly_params_mu: Sequence = []
        self._poly_params_sd: Sequence = []
        self._poly_limits: Tuple[float, float] = (float("nan"), float("nan"))

        self.make_plots = make_plots

        # For prob_Edet_above_threshold
        self._pdet_limits = (1e2, 1e8)

        self._n_components = 3
        self.setup()

        if mode == DistributionMode.PDF:
            mixture_name = "nt_energy_res_mix"
        elif mode == DistributionMode.RNG:
            mixture_name = "nt_energy_res_mix_rng"
        else:
            RuntimeError("This should never happen")

        lognorm = LognormalMixture(mixture_name, self.n_components, self._mode)

        if mode == DistributionMode.PDF:
            super().__init__(
                self.PDF_NAME,
                ["true_energy", "reco_energy"],
                ["real", "real"],
                "real",
            )

        elif mode == DistributionMode.RNG:
            super().__init__(
                self.RNG_NAME,
                ["true_energy"],
                ["real"],
                "real",
            )

            mixture_name = "nt_energy_res_mix_rng"

        else:
            RuntimeError("mode must be DistributionMode.PDF or DistributionMode.RNG")

        # Define Stan interface.
        with self:
            truncated_e = TruncatedParameterization("true_energy", *self._poly_limits)
            log_trunc_e = LogParameterization(truncated_e)

            mu_poly_coeffs = StanArray(
                "NorthernTracksEnergyResolutionMuPolyCoeffs",
                "real",
                self._poly_params_mu,
            )

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

            # for some reason stan complains about weights not adding to 1 if
            # implementing this via StanArray
            with ForLoopContext(1, self._n_components, "i") as i:
                weights[i] << StringExpression(["1.0/", self._n_components])

            log_mu = FunctionCall([mu], "log")

            with ForLoopContext(1, self._n_components, "i") as i:
                mu[i] << [
                    "eval_poly1d(",
                    log_trunc_e,
                    ", ",
                    "to_vector(",
                    mu_poly_coeffs[i],
                    "))",
                ]

                sigma[i] << [
                    "eval_poly1d(",
                    log_trunc_e,
                    ", ",
                    "to_vector(",
                    sd_poly_coeffs[i],
                    "))",
                ]

            log_mu_vec = FunctionCall([log_mu], "to_vector")
            sigma_vec = FunctionCall([sigma], "to_vector")

            if mode == DistributionMode.PDF:
                log_reco_e = LogParameterization("reco_energy")
                ReturnStatement([lognorm(log_reco_e, log_mu_vec, sigma_vec, weights)])
            elif mode == DistributionMode.RNG:
                ReturnStatement([lognorm(log_mu_vec, sigma_vec, weights)])

    def setup(self) -> None:
        # Check cache
        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                data = np.load(fr)
                eres = data["eres"]
                tE_bin_edges = data["tE_bin_edges"]
                rE_bin_edges = data["rE_bin_edges"]
                poly_params_mu = data["poly_params_mu"]
                poly_params_sd = data["poly_params_sd"]
                poly_limits = (float(data["Emin"]), float(data["Emax"]))

        # Or load from file
        else:
            import h5py

            with h5py.File(self.DATA_PATH, "r") as f:
                aeff_numu = f["2010/nu_mu/area"][()]
                aeff_numubar = f["2010/nu_mu_bar/area"][()]

                # Sum over cosz and average over numu/numubar
                eff_area = 0.5 * (aeff_numu.sum(axis=1) + aeff_numubar.sum(axis=1))

                # True Energy [GeV]
                tE_bin_edges = f["2010/nu_mu/bin_edges_0"][:]

                # Reco Energy [GeV]
                rE_bin_edges = f["2010/nu_mu/bin_edges_2"][:]

            # Normalize along Ereco
            bin_width = np.log10(rE_bin_edges[1]) - np.log10(rE_bin_edges[0])
            eres = np.zeros_like(eff_area)
            for i, pdf in enumerate(eff_area):
                if pdf.sum() > 0:
                    eres[i] = pdf / (pdf.sum() * bin_width)

            tE_binc = 0.5 * (tE_bin_edges[:-1] + tE_bin_edges[1:])
            rE_binc = 0.5 * (rE_bin_edges[:-1] + rE_bin_edges[1:])

            fit_params, rebin_tE_binc = self._fit_energy_res(
                tE_binc, rE_binc, eres, self._n_components, rebin=3
            )

            # Min and max values
            imin = 5
            imax = -15

            Emin = rebin_tE_binc[imin]
            Emax = rebin_tE_binc[imax]

            # Fit polynomial
            poly_params_mu, poly_params_sd, poly_limits = self._fit_polynomial(
                fit_params, rebin_tE_binc, Emin, Emax, polydeg=5
            )

            # Save polynomial
            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                np.savez(
                    fr,
                    eres=eres,
                    tE_bin_edges=tE_bin_edges,
                    rE_bin_edges=rE_bin_edges,
                    poly_params_mu=poly_params_mu,
                    poly_params_sd=poly_params_sd,
                    Emin=Emin,
                    Emax=Emax,
                )

            # Set properties
            self._eres = eres
            self._tE_bin_edges = tE_bin_edges
            self._rE_bin_edges = rE_bin_edges

            self._poly_params_mu = poly_params_mu
            self._poly_params_sd = poly_params_sd
            self._poly_limits = poly_limits

            # Show results
            if self.make_plots:
                self.plot_fit_params(fit_params, rebin_tE_binc)
                self.plot_parameterizations(
                    tE_binc,
                    rE_binc,
                    fit_params,
                    rebin_tE_binc=rebin_tE_binc,
                )

        # Set properties
        self._eres = eres
        self._tE_bin_edges = tE_bin_edges
        self._rE_bin_edges = rE_bin_edges

        self._poly_params_mu = poly_params_mu
        self._poly_params_sd = poly_params_sd
        self._poly_limits = poly_limits


class NorthernTracksAngularResolution(AngularResolution):
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

    CACHE_FNAME = "angular_reso_tracks.npz"

    PDF_NAME = "NorthernTracksAngularResolution"
    RNG_NAME = "NorthernTracksAngularResolution_rng"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:
        if mode == DistributionMode.PDF:
            super().__init__(
                self.PDF_NAME,
                ["true_energy", "true_dir", "reco_dir"],
                ["real", "vector", "vector"],
                "real",
            )

        else:
            super().__init__(
                self.RNG_NAME,
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

    PDF_NAME = "NorthernTracksIRF"

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

        # if mode == DistributionMode.PDF:
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
