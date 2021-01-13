from typing import List, Sequence, Tuple
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from astropy import units as u

from ..cache import Cache
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
from ..fitting_tools import Residuals
from .detector_model import EffectiveArea, DetectorModel

import logging

logger = logging.getLogger(__name__)
Cache.set_cache_dir(".cache")


class CascadesEffectiveArea(EffectiveArea):
    """
    Effective area based on the cascade_model simulation.
    """

    local_path = "input/cascades/cascade_detector_model_test.h5"
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


class CascadesEnergyResolution(UserDefinedFunction):
    """
    Energy resolution based on the cascade_model simulation.
    """

    DATA_PATH = "input/cascades/cascade_detector_model_test.h5"
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
        self.poly_params_mu: Sequence = []
        self.poly_params_sd: Sequence = []
        self.poly_limits: Tuple[float, float] = (float("nan"), float("nan"))

        # Mixture of 3 lognormals
        self.n_components = 3

        # Load energy resolution and fit if not cached
        self.setup()

        if mode == DistributionMode.PDF:
            mixture_name = "c_energy_res_mix"
        elif mode == DistributionMode.RNG:
            mixture_name = "c_energy_res_mix_rng"
        else:
            RuntimeError("This should never happen")

        lognorm = LognormalMixture(mixture_name, self.n_components, self._mode)

        if mode == DistributionMode.PDF:
            UserDefinedFunction.__init__(
                self,
                "CascadeEnergyResolution",
                ["true_energy", "reco_energy"],
                ["real", "real"],
                "real",
            )

        elif mode == DistributionMode.RNG:
            UserDefinedFunction.__init__(
                self, "CascadeEnergyResolution_rng", ["true_energy"], ["real"], "real"
            )
            mixture_name = "nt_energy_res_mix_rng"
        else:
            RuntimeError("This should never happen")

        with self:
            truncated_e = TruncatedParameterization("true_energy", *self.poly_limits)
            log_trunc_e = LogParameterization(truncated_e)

            mu_poly_coeffs = StanArray(
                "CascadesEnergyResolutionMuPolyCoeffs", "real", self.poly_params_mu
            )

            sd_poly_coeffs = StanArray(
                "CascadesEnergyResolutionSdPolyCoeffs", "real", self.poly_params_sd
            )

            mu = ForwardArrayDef("mu_e_res", "real", ["[", self.n_components, "]"])
            sigma = ForwardArrayDef(
                "sigma_e_res", "real", ["[", self.n_components, "]"]
            )

            weights = ForwardVariableDef(
                "weights", "vector[" + str(self.n_components) + "]"
            )

            # for some reason stan complains about weights not adding to 1 if
            # implementing this via StanArray
            with ForLoopContext(1, self.n_components, "i") as i:
                weights[i] << StringExpression(["1.0/", self.n_components])

            log_mu = FunctionCall([mu], "log")

            with ForLoopContext(1, self.n_components, "i") as i:
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

    @staticmethod
    def make_fit_model(n_components):
        """
        Lognormal mixture with n_components.
        """

        def _model(x, pars):
            result = 0
            for i in range(n_components):
                result += (
                    1
                    / n_components
                    * stats.lognorm.pdf(x, scale=pars[2 * i], s=pars[2 * i + 1])
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

    def _fit_energy_res(
        self,
        tE_binc: np.ndarray,
        rE_binc: np.ndarray,
        eres: np.ndarray,
        n_components: int,
    ) -> np.ndarray:
        from scipy.optimize import least_squares

        fit_params = []

        logrEbins = np.log10(rE_binc)

        # Lognormal mixture
        model = self.make_fit_model(n_components)

        # Fitting loop
        for index in range(len(tE_binc)):

            # Energy resolution for this true-energy bin
            e_reso = eres[index]

            if e_reso.sum() > 0:

                residuals = Residuals((logrEbins, e_reso), model)

                # Calculate seed as mean of the resolution to help minimizer
                seed_mu = np.average(logrEbins, weights=e_reso)
                if ~np.isfinite(seed_mu):
                    seed_mu = 3

                seed = np.zeros(n_components * 2)
                bounds_lo: List[float] = []
                bounds_hi: List[float] = []
                for i in range(n_components):
                    seed[2 * i] = seed_mu + 0.1 * (i + 1)
                    seed[2 * i + 1] = 0.02
                    bounds_lo += [0, 0.01]
                    bounds_hi += [8, 1]

                # Fit using simple least squares
                res = least_squares(
                    residuals,
                    seed,
                    bounds=(bounds_lo, bounds_hi),
                )

                # Check for label swapping
                mu_indices = np.arange(0, stop=n_components * 2, step=2)
                mu_order = np.argsort(res.x[mu_indices])

                # Store fit parameters
                this_fit_pars: List = []
                for i in range(n_components):
                    mu_index = mu_indices[mu_order[i]]
                    this_fit_pars += [res.x[mu_index], res.x[mu_index + 1]]
                fit_params.append(this_fit_pars)

            else:

                fit_params.append(np.zeros(2 * n_components))

        fit_params = np.asarray(fit_params)

        return fit_params

    def plot_fit_params(
        self, fit_params: np.ndarray, rebinned_binc: np.ndarray
    ) -> None:

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        xs = np.linspace(*np.log10(self.poly_limits), num=100)

        if self.poly_params_mu is None:

            raise RuntimeError("Run setup() first")

        # Plot polynomial fits for each mixture component.
        for comp in range(self.n_components):

            params_mu = self.poly_params_mu[comp]
            axs[0].plot(xs, np.poly1d(params_mu)(xs))
            axs[0].plot(
                np.log10(rebinned_binc),
                fit_params[:, 2 * comp],
                label="Mean {}".format(comp),
            )

            params_sigma = self.poly_params_sd[comp]
            axs[1].plot(xs, np.poly1d(params_sigma)(xs))
            axs[1].plot(
                np.log10(rebinned_binc),
                fit_params[:, 2 * comp + 1],
                label="SD {}".format(comp),
            )

        axs[0].set_xlabel("log10(True Energy / GeV)")
        axs[0].set_ylabel("Parameter Value")
        plt.tight_layout()

    def plot_parameterizations(
        self,
        tE_binc: np.ndarray,
        rE_binc: np.ndarray,
        fit_params: np.ndarray,
        eres: np.ndarray,
    ):
        """
        Plot fitted parameterizations
        Args:
            tE_binc: np.ndarray
                True energy bin centers
            rE_binc: np.ndarray
                Reconstructed energy bin centers
            fit_params: np.ndarray
                Fitted parameters for mu and sigma
            eres: np.ndarray
                P(Ereco | Etrue)
        """

        import matplotlib.pyplot as plt

        plot_energies = [1e5, 3e5, 5e5, 8e5, 1e6, 3e6, 5e6, 8e6]  # GeV

        if self.poly_params_mu is None:

            raise RuntimeError("Run setup() first")

        # Find true energy bins for the chosen plotting energies
        plot_indices = np.digitize(plot_energies, tE_binc)

        logrEbins = np.log10(rE_binc)

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        xs = np.linspace(*np.log10(self.poly_limits), num=100)

        model = self.make_fit_model(self.n_components)
        fl_ax = axs.ravel()

        for i, p_i in enumerate(plot_indices):

            log_plot_e = np.log10(plot_energies[i])

            model_params: List[float] = []
            for comp in range(self.n_components):

                mu = np.poly1d(self.poly_params_mu[comp])(log_plot_e)
                sigma = np.poly1d(self.poly_params_sd[comp])(log_plot_e)
                model_params += [mu, sigma]

            e_reso = eres[p_i]
            fl_ax[i].plot(logrEbins, e_reso)

            res = fit_params[plot_indices[i]]

            fl_ax[i].plot(xs, model(xs, model_params))
            fl_ax[i].plot(xs, model(xs, res))
            fl_ax[i].set_ylim(1e-4, 5)
            fl_ax[i].set_yscale("log")
            fl_ax[i].set_title("True E: {:.1E}".format(tE_binc[p_i]))

        ax = fig.add_subplot(111, frameon=False)

        # Hide tick and tick label of the big axes
        ax.tick_params(
            labelcolor="none", top="off", bottom="off", left="off", right="off"
        )
        ax.grid(False)
        ax.set_xlabel("log10(Reconstructed Energy /GeV)")
        ax.set_ylabel("PDF")
        plt.tight_layout()

    def setup(self) -> None:
        """
        Load Eres data and perform fits, or load from cache
        if already stored.
        """

        # Check cache
        if self.CACHE_FNAME in Cache:

            with Cache.open(self.CACHE_FNAME, "rb") as fr:

                data = np.load(fr)
                poly_params_mu = data["poly_params_mu"]
                poly_params_sd = data["poly_params_sd"]
                poly_limits = (float(data["e_min"]), float(data["e_max"]))

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

            fit_params = self._fit_energy_res(tE_binc, rE_binc, eres, self.n_components)

            def find_nearest_idx(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx

            e_min = 1e3
            e_max = 1e7
            imin = find_nearest_idx(tE_binc, e_min)
            imax = find_nearest_idx(tE_binc, e_max)

            # Degree of polynomial to fit
            polydeg = 3

            log10_tE_binc = np.log10(tE_binc)
            poly_params_mu = np.zeros((self.n_components, polydeg + 1))

            # Fit polynomial
            poly_params_sd = np.zeros_like(poly_params_mu)
            for i in range(self.n_components):
                poly_params_mu[i] = np.polyfit(
                    log10_tE_binc[imin:imax], fit_params[:, 2 * i][imin:imax], polydeg
                )
                poly_params_sd[i] = np.polyfit(
                    log10_tE_binc[imin:imax],
                    fit_params[:, 2 * i + 1][imin:imax],
                    polydeg,
                )

            poly_limits = (e_min, e_max)

            # Save polynomial
            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                np.savez(
                    fr,
                    poly_params_mu=poly_params_mu,
                    poly_params_sd=poly_params_sd,
                    e_min=e_min,
                    e_max=e_max,
                )

            # Store params
            self.poly_params_mu = poly_params_mu
            self.poly_params_sd = poly_params_sd
            self.poly_limits = poly_limits

            # Show results
            self.plot_fit_params(fit_params, tE_binc)
            self.plot_parameterizations(tE_binc, rE_binc, fit_params, eres)

        # poly params are now set
        self.poly_params_mu = poly_params_mu
        self.poly_params_sd = poly_params_sd
        self.poly_limits = poly_limits

    @u.quantity_input
    def prob_Edet_above_threshold(self, true_energy: u.GeV, threshold_energy: u.GeV):
        """
        P(Edet > Edet_min | E) for use in precomputation.
        """

        model = self.make_cumulative_model(self.n_components)

        prob = np.zeros_like(true_energy)
        model_params: List[float] = []

        for comp in range(self.n_components):

            mu = np.poly1d(self.poly_params_mu[comp])(np.log10(true_energy.value))
            sigma = np.poly1d(self.poly_params_sd[comp])(np.log10(true_energy.value))
            model_params += [mu, sigma]

        prob = 1 - model(np.log10(threshold_energy.value), model_params)

        return prob


class CascadesAngularResolution(UserDefinedFunction):
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

    DATA_PATH = "input/cascades/CascadesAngularResolution.csv"
    CACHE_FNAME = "angular_reso_cascades.npz"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:

        if mode == DistributionMode.PDF:

            UserDefinedFunction.__init__(
                self,
                "CascadesAngularResolution",
                ["true_energy", "true_dir", "reco_dir"],
                ["real", "vector", "vector"],
                "real",
            )
        else:
            UserDefinedFunction.__init__(
                self,
                "CascadesAngularResolution_rng",
                ["true_energy", "true_dir"],
                ["real", "vector"],
                "vector",
            )

        self.poly_params: Sequence = []
        self.e_min: float = float("nan")
        self.e_max: float = float("nan")

        self.setup()

        with self:

            # Clip true energy
            clipped_e = TruncatedParameterization("true_energy", self.e_min, self.e_max)

            clipped_log_e = LogParameterization(clipped_e)

            kappa = PolynomialParameterization(
                clipped_log_e, self.poly_params, "CascadesAngularResolutionPolyCoeffs"
            )

            if mode == DistributionMode.PDF:
                # VMF expects x_obs, x_true
                vmf = VMFParameterization(["reco_dir", "true_dir"], kappa, mode)

            elif mode == DistributionMode.RNG:
                vmf = VMFParameterization(["true_dir"], kappa, mode)

            ReturnStatement([vmf])

    def _calc_resolution(self):
        pass

    def setup(self) -> None:
        """
        Load angular resolution and fit or
        use cache if already stored.
        """

        # Check cache
        if self.CACHE_FNAME in Cache:

            with Cache.open(self.CACHE_FNAME, "rb") as fr:

                data = np.load(fr)
                self.poly_params = data["poly_params"]
                self.e_min = float(data["e_min"])
                self.e_max = float(data["e_max"])

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

            self.poly_params = np.polyfit(data.log10energy, data.kappa, 5)
            self.e_min = float(data.log10energy.min())
            self.e_max = float(data.log10energy.max())

            # Save polynomial
            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                np.savez(
                    fr,
                    poly_params=self.poly_params,
                    e_min=10 ** data.log10energy.min(),
                    e_max=10 ** data.log10energy.max(),
                )


class CascadesDetectorModel(DetectorModel):
    """
    Implements the detector model for the cascades.
    Parameters:
        mode: DistributionMode
            Set mode to either RNG or PDF
    """

    def __init__(self, mode: DistributionMode = DistributionMode.PDF):
        DetectorModel.__init__(self, mode)

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
