import os
import numpy as np
import pandas as pd
from typing import Sequence, Tuple

from ..utils.cache import Cache
from ..backend import (
    VMFParameterization,
    PolynomialParameterization,
    TruncatedParameterization,
    LogParameterization,
    SimpleHistogram,
    ReturnStatement,
    UserDefinedFunction,
    FunctionCall,
    DistributionMode,
    LognormalMixture,
    ForLoopContext,
    ForwardVariableDef,
    ForwardArrayDef,
    StanArray,
    StringExpression,
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


class CascadesEffectiveArea(EffectiveArea):
    """
    Effective area based on the cascade_model simulation.
    """

    local_path = "input/cascades/cascade_detector_model.h5"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "aeff_cascades.npz"

    def __init__(self) -> None:

        super().__init__(
            "CascadesEffectiveArea",
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real",
        )

        self.setup()

        # Define Stan interface
        with self:
            hist = SimpleHistogram(
                self._eff_area,
                [self._tE_bin_edges, self._cosz_bin_edges],
                "CascadesEffAreaHist",
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

            import h5py

            with h5py.File(self.DATA_PATH, "r") as f:

                # Effective area [m^2]
                eff_area = f["aeff/aeff"][()]

                # True Energy [GeV]
                tE_bin_edges = f["aeff/tE_bin_edges"][()]

                # cos(zenith)
                cosz_bin_edges = f["aeff/cosz_bin_edges"][()]

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


class CascadesEnergyResolution(EnergyResolution):
    """
    Energy resolution based on the cascade_model simulation.
    """

    local_path = "input/cascades/cascade_detector_model.h5"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "energy_reso_cascades.npz"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:
        """
        Energy resolution based on the cascade_model simulation.

        Energy resolution is calculated by fitting a lognormal mixture
        with self.n_components to the normalised Ereco dependence of a given
        Etrue bin. The variation of the lognormal mixture parameters with energy
        are then fit with a polynomial for fast interpolation and evaluation.
        """

        self._mode = mode

        # Parameters of polynomials for lognormal mu and sd
        self._poly_params_mu: Sequence = []
        self._poly_params_sd: Sequence = []
        self._poly_limits: Tuple[float, float] = (float("nan"), float("nan"))

        # For prob_Edet_above_threshold
        self._pdet_limits = (5e2, 1e8)

        # Mixture of 3 lognormals
        self._n_components = 4

        # Load energy resolution and fit if not cached
        self.setup()

        if mode == DistributionMode.PDF:
            mixture_name = "c_energy_res_mix"
        elif mode == DistributionMode.RNG:
            mixture_name = "c_energy_res_mix_rng"
        else:
            RuntimeError("This should never happen")

        lognorm = LognormalMixture(mixture_name, self._n_components, self._mode)

        if mode == DistributionMode.PDF:

            super().__init__(
                "CascadeEnergyResolution",
                ["true_energy", "reco_energy"],
                ["real", "real"],
                "real",
            )

        elif mode == DistributionMode.RNG:

            super().__init__(
                "CascadeEnergyResolution_rng", ["true_energy"], ["real"], "real"
            )
            mixture_name = "nt_energy_res_mix_rng"

        else:

            RuntimeError(
                "mode must be either DistributionMode.PDF or DistributionMode.RNG"
            )

        with self:

            # Define parametrization in Stan.
            truncated_e = TruncatedParameterization("true_energy", *self._poly_limits)
            log_trunc_e = LogParameterization(truncated_e)

            mu_poly_coeffs = StanArray(
                "CascadesEnergyResolutionMuPolyCoeffs", "real", self._poly_params_mu
            )

            sd_poly_coeffs = StanArray(
                "CascadesEnergyResolutionSdPolyCoeffs", "real", self._poly_params_sd
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
        """
        Load Eres data and perform fits, or load from cache
        if already stored.
        """

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

        # Load energy resolution from file
        else:

            import h5py

            with h5py.File(self.DATA_PATH, "r") as f:

                # P(Ereco | Etrue), normalised along Ereco
                eres = f["eres/eres"][()]

                # True Energy [GeV]
                tE_bin_edges = f["eres/tE_bin_edges"][()]

                # Reco Energy [GeV]
                rE_bin_edges = f["eres/rE_bin_edges"][()]

            tE_binc = 0.5 * (tE_bin_edges[:-1] + tE_bin_edges[1:])
            rE_binc = 0.5 * (rE_bin_edges[:-1] + rE_bin_edges[1:])

            fit_params, _ = self._fit_energy_res(
                tE_binc, rE_binc, eres, self._n_components, rebin=1
            )

            Emin = np.min(tE_bin_edges)  # GeV
            Emax = np.max(tE_bin_edges)  # GeV

            poly_params_mu, poly_params_sd, poly_limits = self._fit_polynomial(
                fit_params, tE_binc, Emin=Emin, Emax=Emax, polydeg=3
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

            # Store params
            self._eres = eres
            self._tE_bin_edges = tE_bin_edges
            self._rE_bin_edges = rE_bin_edges

            self._poly_params_mu = poly_params_mu
            self._poly_params_sd = poly_params_sd
            self._poly_limits = poly_limits

            # Show results
            self.plot_fit_params(fit_params, tE_binc)
            self.plot_parameterizations(tE_binc, rE_binc, fit_params)

        # poly params are now set
        self._poly_params_mu = poly_params_mu
        self._poly_params_sd = poly_params_sd
        self._poly_limits = poly_limits


class CascadesAngularResolution(AngularResolution):
    """
    Angular resolution for Cascades
    Data from https://arxiv.org/pdf/1311.4767.pdf (Fig. 14)
    Extrapolated using a complementary error function
    Fits a polynomial to the median angular resolution converted to
    `kappa` parameter of a VMF distribution
    Attributes:
        poly_params: Coefficients of the polynomial
        e_min: Lower energy bound of the polynomial
        e_max: Upper energy bound of the polynomial
    """

    local_path = "input/cascades/CascadesAngularResolution.csv"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "angular_reso_cascades.npz"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:

        if mode == DistributionMode.PDF:

            super().__init__(
                "CascadesAngularResolution",
                ["true_energy", "true_dir", "reco_dir"],
                ["real", "vector", "vector"],
                "real",
            )

        else:

            super().__init__(
                "CascadesAngularResolution_rng",
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
                clipped_log_e, self._poly_params, "CascadesAngularResolutionPolyCoeffs"
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
            clipped_log_e, self._poly_params, "CascadesAngularResolutionPolyCoeffs"
        )

        return kappa

    def setup(self) -> None:
        """
        Load angular resolution and fit or
        use cache if already stored.
        """

        # Check cache
        if self.CACHE_FNAME in Cache:

            with Cache.open(self.CACHE_FNAME, "rb") as fr:

                data = np.load(fr)
                self._kappa = data["kappa_grid"]
                self._Egrid = data["Egrid"]
                self._poly_params = data["poly_params"]
                self._Emin = float(data["Emin"])
                self._Emax = float(data["Emax"])
                self._poly_limits = (self._Emin, self._Emax)

        # Load input data and fit polynomial
        else:

            if not os.path.exists(self.DATA_PATH):

                raise RuntimeError(self.DATA_PATH, "is not a valid path")

            data = pd.read_csv(
                self.DATA_PATH,
                header=None,
                names=["log10energy", "resolution"],
            )

            # Kappa parameter of VMF distribution
            data["kappa"] = 1.38 / np.radians(data.resolution) ** 2

            self._kappa_grid = data.kappa.values
            self._Egrid = 10 ** data.log10energy.values
            self._poly_params = np.polyfit(
                data.log10energy.values, data.kappa.values, 5
            )
            self._Emin = 10 ** float(data.log10energy.min())
            self._Emax = 10 ** float(data.log10energy.max())

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


class CascadesDetectorModel(DetectorModel):
    """
    Implements the detector model for the cascades.
    Parameters:
        mode: DistributionMode
            Set mode to either RNG or PDF
    """

    event_types = ["cascades"]

    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        event_type=None,
    ):

        super().__init__(mode, event_type="cascades")

        ang_res = CascadesAngularResolution(mode)
        self._angular_resolution = ang_res

        energy_res = CascadesEnergyResolution(mode)
        self._energy_resolution = energy_res

        if mode == DistributionMode.PDF:
            self._eff_area = CascadesEffectiveArea()

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

    from backend.stan_generator import (
        StanGenerator,
        GeneratedQuantitiesContext,
        DataContext,
        FunctionsContext,
        Include,
    )

    logging.basicConfig(level=logging.DEBUG)
    import pystan  # type: ignore
    import numpy as np

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
            ntd = CascadesDetectorModel()

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
