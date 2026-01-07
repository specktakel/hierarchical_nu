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
    TwoDimHistInterpolation,
    ReturnStatement,
    UserDefinedFunction,
    FunctionCall,
    DistributionMode,
    LognormalMixture,
    ForLoopContext,
    ForwardVariableDef,
    InstantVariableDef,
    ForwardArrayDef,
    StanArray,
    StringExpression,
)
from .detector_model import (
    EffectiveArea,
    LogNormEnergyResolution,
    AngularResolution,
    DetectorModel,
)

from ..source.source import Sources

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

    NAME = "CascadesEffectiveArea"

    def __init__(self) -> None:
        logger.warning("Cascades detector model is no longer maintained")
        self._func_name = self.NAME
        self.setup()

        self._make_spline()

        # Define Stan interface

    def generate_code(self):
        super().__init__(
            self._func_name,
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real",
        )

        with self:
            hist = TwoDimHistInterpolation(
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

        self._rs_bbpl_params = {}
        self._rs_bbpl_params["threshold_energy"] = 5e4  # GeV
        self._rs_bbpl_params["gamma1"] = -0.8
        self._rs_bbpl_params["gamma2_scale"] = 0.5


class CascadesEnergyResolution(LogNormEnergyResolution):
    """
    Energy resolution based on the cascade_model simulation.
    """

    local_path = "input/cascades/cascade_detector_model.h5"
    DATA_PATH = os.path.join(os.path.dirname(__file__), local_path)

    CACHE_FNAME = "energy_reso_cascades.npz"

    PDF_NAME = "CascadeEnergyResolution"
    RNG_NAME = "CascadeEnergyResolution_rng"

    def __init__(
        self, mode: DistributionMode = DistributionMode.PDF, make_plots: bool = False
    ) -> None:
        """
        Energy resolution based on the cascade_model simulation.

        Energy resolution is calculated by fitting a lognormal mixture
        with self.n_components to the normalised Ereco dependence of a given
        Etrue bin. The variation of the lognormal mixture parameters with energy
        are then fit with a polynomial for fast interpolation and evaluation.
        """

        logger.warning("Cascades detector model is no longer maintained")
        self.mode = mode

        # Parameters of polynomials for lognormal mu and sd
        self._poly_params_mu: Sequence = []
        self._poly_params_sd: Sequence = []
        self._poly_limits: Tuple[float, float] = (float("nan"), float("nan"))

        self.make_plots = make_plots

        # For prob_Edet_above_threshold
        self._pdet_limits = (5e2, 1e8)

        # Mixture of 4 lognormals
        self._n_components = 4

        # Load energy resolution and fit if not cached
        self.setup()

        if mode == DistributionMode.PDF:
            self._func_name = self.PDF_NAME
        elif mode == DistributionMode.RNG:
            self._func_name = self.RNG_NAME
        else:
            RuntimeError("This should never happen")

    def generate_code(self):
        if self.mode == DistributionMode.PDF:
            mixture_name = "c_energy_res_mix"

            super().__init__(
                self.PDF_NAME,
                ["log_true_energy", "log_reco_energy"],
                ["real", "real"],
                "real",
            )

        elif self.mode == DistributionMode.RNG:
            super().__init__(self.RNG_NAME, ["log_true_energy"], ["real"], "real")
            mixture_name = "c_energy_res_mix_rng"

        else:
            RuntimeError(
                "mode must be either DistributionMode.PDF or DistributionMode.RNG"
            )

        with self:
            # Define parametrization in Stan.
            lognorm = LognormalMixture(mixture_name, self._n_components, self.mode)

            log_trunc_e = TruncatedParameterization(
                "log_true_energy",
                np.log10(self._poly_limits[0]),
                np.log10(self._poly_limits[1]),
            )

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

            if self.mode == DistributionMode.PDF:
                ReturnStatement(
                    [lognorm("log_reco_energy", log_mu_vec, sigma_vec, weights)]
                )
            elif self.mode == DistributionMode.RNG:
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
            if self.make_plots:
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

    PDF_NAME = "CascadesAngularResolution"
    RNG_NAME = "CascadesAngularResolution_rng"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:
        logger.warning("Cascades detector model is no longer maintained")
        self.mode = mode
        self._kappa_grid: np.ndarray = None
        self._Egrid: np.ndarray = None
        self._poly_params: Sequence = []
        self._Emin: float = float("nan")
        self._Emax: float = float("nan")

        self.setup()

        if self.mode == DistributionMode.PDF:
            self._func_name = self.PDF_NAME
        else:
            self._func_name = self.RNG_NAME

    def generate_code(self):
        if self.mode == DistributionMode.PDF:
            super().__init__(
                self._func_name,
                ["true_dir", "reco_dir", "kappa"],
                ["vector", "vector", "real"],
                "real",
            )

        else:
            super().__init__(
                self._func_name,
                ["log_true_energy", "true_dir"],
                ["real", "vector"],
                "vector",
            )

        with self:
            if self.mode == DistributionMode.PDF:
                # VMF expects x_obs, x_true
                vmf = VMFParameterization(["reco_dir", "true_dir"], "kappa", self.mode)
                ReturnStatement([vmf])

            elif self.mode == DistributionMode.RNG:
                # Clip true energy
                clipped_log_e = TruncatedParameterization(
                    "log_true_energy", np.log10(self._Emin), np.log10(self._Emax)
                )

                kappa = PolynomialParameterization(
                    clipped_log_e,
                    self._poly_params,
                    "CascadesAngularResolutionPolyCoeffs",
                )
                pre_event = ForwardVariableDef("pre_event", "vector[4]")
                vmf = VMFParameterization(["true_dir"], kappa, self.mode)
                pre_event[1:3] << vmf
                pre_event[4] << kappa

                ReturnStatement([pre_event])

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
            self._Egrid = 10**data.log10energy.values
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

    PDF_NAME = "CascadesPDF"
    RNG_NAME = "Cascades_rng"

    RNG_FILENAME = "cascades_rng.stan"
    PDF_FILENAME = "cascades_pdf.stan"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF):
        logger.warning("Cascades detector model is no longer maintained")
        super().__init__(mode, event_type="cascades")

        if self.mode == DistributionMode.PDF:
            self._func_name = self.PDF_NAME
        elif self.mode == DistributionMode.RNG:
            self._func_name = self.RNG_NAME

        ang_res = CascadesAngularResolution(mode)
        self._angular_resolution = ang_res

        energy_res = CascadesEnergyResolution(mode)
        self._energy_resolution = energy_res

        self._eff_area = CascadesEffectiveArea()

    def _get_effective_area(self):
        return self._eff_area

    def _get_energy_resolution(self):
        return self._energy_resolution

    def _get_angular_resolution(self):
        return self._angular_resolution

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
                ["true_energy", "detected_energy", "omega_det", "src_pos"],
                ["real", "real", "vector", "array[] vector"],
                "tuple(array[] real, array[] real, array[] real)",
            )
        else:
            UserDefinedFunction.__init__(
                self,
                self._func_name,
                ["true_energy", "detected_energy", "omega_det", "src_pos"],
                ["real", "real", "vector", "vector"],
                "tuple(real, real, array[] real)",
            )

        with self:
            if not single_ps:
                Ns = InstantVariableDef("Ns", "int", ["size(src_pos)"])
                ps_eres = ForwardArrayDef("ps_eres", "real", ["[", Ns, "]"])
                ps_aeff = ForwardArrayDef("ps_aeff", "real", ["[", Ns, "]"])

            else:
                ps_eres = ForwardVariableDef("ps_eres", "real")
                ps_aeff = ForwardVariableDef("ps_aeff", "real")
            diff = ForwardArrayDef("diff", "real", ["[3]"])
            eres = ForwardVariableDef("eres", "real")
            eres << self.energy_resolution(
                "log10(true_energy)", "log10(detected_energy)"
            )
            if not single_ps:
                with ForLoopContext(1, Ns, "i") as i:
                    ps_eres[i] << eres
                    ps_aeff[i] << FunctionCall(
                        [
                            self.effective_area("true_energy", "src_pos[i]"),
                        ],
                        "log",
                    )
            else:
                ps_eres << eres
                ps_aeff << FunctionCall(
                    [
                        self.effective_area("true_energy", "src_pos"),
                    ],
                    "log",
                )

            diff[1] << eres

            diff[2] << FunctionCall(
                [
                    self.effective_area("true_energy", "omega_det"),
                ],
                "log",
            )

            diff[3] << "negative_infinity()"

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
            self.RNG_NAME,
            ["true_energy", "omega"],
            ["real", "vector"],
            "vector",
        )

        with self:
            return_this = ForwardVariableDef("return_this", "vector[5]")
            return_this[1] << FunctionCall(
                [10.0, self.energy_resolution("log10(true_energy)")], "pow"
            )
            return_this[2:5] << self.angular_resolution("log10(true_energy)", "omega")
            ReturnStatement([return_this])


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
