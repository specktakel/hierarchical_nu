"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Tuple
import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import scipy.stats as stats  # type: ignore
from astropy import units as u

from .cache import Cache
from .backend import (
    Expression,
    TExpression,
    TListTExpression,
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
from .fitting_tools import Residuals

import logging

logger = logging.getLogger(__name__)
Cache.set_cache_dir(".cache")


class EffectiveArea(UserDefinedFunction, metaclass=ABCMeta):
    """
    Implements baseclass for effective areas.


    Every implementation of an effective area has to define a setup method,
    that will take care of downloading required files, creating splines, etc.

    The effective areas can depend on multiple quantities (ie. energy,
    direction, time, ..)
    """

    @abstractmethod
    def _calc_effective_area(self, param_dict: dict) -> float:
        pass

    @abstractmethod
    def setup(self) -> None:
        """
        Download and or build all the required input data for calculating
        the effective area
        """
        pass


class Resolution(Expression, metaclass=ABCMeta):
    """
    Base class for parameterizing resolutions
    """

    def __init__(self, inputs: Sequence[TExpression], stan_code: TListTExpression):
        Expression.__init__(self, inputs, stan_code)

    def __call__(self, **kwargs):
        """
        Return the resolution for variables given in kwargs
        """
        return self._calc_resolution(kwargs)

    @abstractmethod
    def _calc_resolution(self, param_dict: dict):
        pass

    @abstractmethod
    def setup(self):
        """
        Download and or build all the required input data for calculating
        the resolution
        """
        pass


class NorthernTracksEffectiveArea(UserDefinedFunction):
    """
    Effective area for the two-year Northern Tracks release:
    https://icecube.wisc.edu/science/data/HE_NuMu_diffuse

    """

    DATA_PATH = "../dev/statistical_model/4_tracks_and_cascades/aeff_input_tracks/effective_area.h5"  # noqa: E501
    CACHE_FNAME = "aeff_tracks.npz"

    def __init__(self) -> None:
        UserDefinedFunction.__init__(
            self,
            "NorthernTracksEffectiveArea",
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real",
        )

        self.setup()

        with self:
            hist = SimpleHistogram(
                self._eff_area,
                [self._tE_bin_edges, self._cosz_bin_edges],
                "NorthernTracksEffAreaHist",
            )

            # z = cos(theta)
            cos_dir = "cos(pi() - acos(true_dir[3]))"
            # cos_dir = FunctionCall(["true_dir"], "cos")
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
                eff_area = f["2010/nu_mu/area"][()]
                # sum over reco energy
                eff_area = eff_area.sum(axis=2)
                # True Energy [GeV]
                tE_bin_edges = f["2010/nu_mu/bin_edges_0"][:]
                # cos(zenith)
                cosz_bin_edges = f["2010/nu_mu/bin_edges_1"][:]
                # Reco Energy [GeV]
                # rE_bin_edges = f['2010/nu_mu/bin_edges_2'][:]

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


class NorthernTracksEnergyResolution(UserDefinedFunction):

    """
    Energy resolution for Northern Tracks Sample

    Data from https://arxiv.org/pdf/1811.07979.pdf
    """

    DATA_PATH = "../dev/statistical_model/4_tracks_and_cascades/aeff_input_tracks/effective_area.h5"  # noqa: E501
    CACHE_FNAME = "energy_reso_tracks.npz"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:
        """
        Args:
            inputs: List[TExpression]
                First item is true energy, second item is reco energy
        """

        self._mode = mode
        self.poly_params_mu: Sequence = []
        self.poly_params_sd: Sequence = []
        self.poly_limits: Tuple[float, float] = (float("nan"), float("nan"))

        self.n_components = 3
        self.setup()

        if mode == DistributionMode.PDF:
            mixture_name = "nt_energy_res_mix"
        elif mode == DistributionMode.RNG:
            mixture_name = "nt_energy_res_mix_rng"
        else:
            RuntimeError("This should never happen")

        lognorm = LognormalMixture(mixture_name, self.n_components, self._mode)

        if mode == DistributionMode.PDF:
            UserDefinedFunction.__init__(
                self,
                "NorthernTracksEnergyResolution",
                ["true_energy", "reco_energy"],
                ["real", "real"],
                "real",
            )

        elif mode == DistributionMode.RNG:
            UserDefinedFunction.__init__(
                self,
                "NorthernTracksEnergyResolution_rng",
                ["true_energy"],
                ["real"],
                "real",
            )
            mixture_name = "nt_energy_res_mix_rng"
        else:
            RuntimeError("This should never happen")

        with self:
            truncated_e = TruncatedParameterization("true_energy", *self.poly_limits)
            log_trunc_e = LogParameterization(truncated_e)

            mu_poly_coeffs = StanArray(
                "NorthernTracksEnergyResolutionMuPolyCoeffs",
                "real",
                self.poly_params_mu,
            )

            sd_poly_coeffs = StanArray(
                "NorthernTracksEnergyResolutionSdPolyCoeffs",
                "real",
                self.poly_params_sd,
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
        Lognormal mixture
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
        Cumulative Lognormal mixture above xth
        """

        def _cumulative_model(xth, pars):
            result = 0
            for i in range(n_components):
                result += (
                    1
                    / n_components
                    * stats.lognorm.cdf(xth, scale=pars[2 * i], s=pars[2 * i + 1])
                )
            return result

        return _cumulative_model

    def _fit_energy_res(
        self,
        tE_binc: np.ndarray,
        rE_binc: np.ndarray,
        eff_area: np.ndarray,
        n_components: int,
    ) -> np.ndarray:
        from scipy.optimize import least_squares  # type: ignore

        fit_params = []
        # Rebin to have higher statistics at upper
        # and lower end of energy range
        rebin = 3
        rebinned_binc = np.zeros(int(len(tE_binc) / rebin))
        logrEbins = np.log10(rE_binc)

        model = self.make_fit_model(n_components)
        # Fitting loop
        for index in range(int(len(tE_binc) / rebin)):
            # Calculate rebinned bin-centers as mean of first and
            # last bin being summed
            rebinned_binc[index] = (
                0.5 * (tE_binc[[index * rebin, rebin * (index + 1) - 1]]).sum()
            )

            # Calculate the energy resolution for this true-energy bin
            e_reso = eff_area.sum(axis=1)[index * rebin : (index + 1) * rebin]
            e_reso = e_reso.sum(axis=0)
            if e_reso.sum() > 0:
                # Normalize to prob. density / bin
                e_reso = e_reso / e_reso.sum() / (logrEbins[1] - logrEbins[0])

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

                res = least_squares(residuals, seed, bounds=(bounds_lo, bounds_hi),)

                # Check for label swapping
                mu_indices = np.arange(0, stop=n_components * 2, step=2)
                mu_order = np.argsort(res.x[mu_indices])

                this_fit_pars: List = []
                for i in range(n_components):
                    mu_index = mu_indices[mu_order[i]]
                    this_fit_pars += [res.x[mu_index], res.x[mu_index + 1]]
                fit_params.append(this_fit_pars)
            else:
                fit_params.append(np.zeros(2 * n_components))
        fit_params = np.asarray(fit_params)
        return fit_params, rebinned_binc

    def plot_fit_params(
        self, fit_params: np.ndarray, rebinned_binc: np.ndarray
    ) -> None:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        xs = np.linspace(*np.log10(self.poly_limits), num=100)

        if self.poly_params_mu is None:
            raise RuntimeError("Run setup() first")
        for comp in range(self.n_components):
            params_mu = self.poly_params_mu[comp]
            axs[0].plot(xs, np.poly1d(params_mu)(xs))
            axs[0].plot(
                np.log10(rebinned_binc),
                fit_params[:, 2 * comp],
                label="Mean {}".format(comp),
            )

            params_sigma = self.poly_params_sd[comp]  # type: ignore
            axs[1].plot(xs, np.poly1d(params_sigma)(xs))
            axs[1].plot(
                np.log10(rebinned_binc),
                fit_params[:, 2 * comp + 1],
                label="SD {}".format(comp),
            )
        axs[0].set_xlabel("log10(True Energy / GeV)")
        axs[0].set_ylabel("Parameter Value")
        plt.tight_layout()
        plt.savefig("energy_fit_params.png", dpi=150)

    def plot_parameterizations(
        self,
        tE_binc: np.ndarray,
        rebinned_binc: np.ndarray,
        rE_binc: np.ndarray,
        fit_params: np.ndarray,
        eff_area: np.ndarray,
    ):
        """
        Plot fitted parameterizations

        Args:
            tE_binc: np.ndarray
                True energy bin centers
            rebinned_binc: np.ndarray:
                Rebinned true energy bin centers
            rE_binc: np.ndarray
                Reconstructed energy bin centers

            fit_params: np.ndarray
                Fitted parameters for mu and sigma
            eff_area: np.ndarray
                Effective Area

        """
        import matplotlib.pyplot as plt  # type: ignore

        plot_energies = [100, 200, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]  # GeV

        if self.poly_params_mu is None:
            raise RuntimeError("Run setup() first")

        # Find true energy bins for the chosen plotting energies
        plot_indices = np.digitize(plot_energies, tE_binc)
        # Parameters are relative to the rebinned histogram
        param_indices = np.digitize(plot_energies, rebinned_binc)

        logrEbins = np.log10(rE_binc)

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        xs = np.linspace(*np.log10(self.poly_limits), num=100)

        rebin = int(len(tE_binc) / len(rebinned_binc))

        model = self.make_fit_model(self.n_components)
        fl_ax = axs.ravel()
        for i, p_i in enumerate(plot_indices):
            log_plot_e = np.log10(plot_energies[i])
            model_params: List[float] = []
            for comp in range(self.n_components):
                mu = np.poly1d(self.poly_params_mu[comp])(log_plot_e)
                sigma = np.poly1d(self.poly_params_sd[comp])(log_plot_e)
                model_params += [mu, sigma]
            e_reso = eff_area.sum(axis=1)
            e_reso = e_reso[int(p_i / rebin) * rebin : (int(p_i / rebin) + 1) * rebin]
            e_reso = e_reso.sum(axis=0) / e_reso.sum()
            e_reso /= logrEbins[1] - logrEbins[0]
            fl_ax[i].plot(logrEbins, e_reso)

            res = fit_params[param_indices[i]]

            fl_ax[i].plot(xs, model(xs, model_params))
            fl_ax[i].plot(xs, model(xs, res))
            fl_ax[i].set_ylim(1e-4, 10)
            fl_ax[i].set_yscale("log")
            fl_ax[i].set_title("True E: {:.1E}".format(tE_binc[p_i]))

        ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        ax.tick_params(
            labelcolor="none", top="off", bottom="off", left="off", right="off"
        )
        ax.grid(False)
        ax.set_xlabel("log10(Reconstructed Energy /GeV)")
        ax.set_ylabel("PDF")
        plt.tight_layout()
        plt.savefig("energy_parameterizations.png", dpi=150)

    def setup(self) -> None:
        # Load Aeff data

        # Check cache
        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                data = np.load(fr)
                poly_params_mu = data["poly_params_mu"]
                poly_params_sd = data["poly_params_sd"]
                poly_limits = (float(data["e_min"]), float(data["e_max"]))

        else:
            import h5py  # type: ignore

            with h5py.File(self.DATA_PATH, "r") as f:
                eff_area = f["2010/nu_mu/area"][()]
                # True Energy [GeV]
                tE_bin_edges = f["2010/nu_mu/bin_edges_0"][:]
                # cos(zenith)
                # cosz_bin_edges = f['2010/nu_mu/bin_edges_1'][:]
                # Reco Energy [GeV]
                rE_bin_edges = f["2010/nu_mu/bin_edges_2"][:]

            tE_binc = 0.5 * (tE_bin_edges[:-1] + tE_bin_edges[1:])
            rE_binc = 0.5 * (rE_bin_edges[:-1] + rE_bin_edges[1:])
            n_components = 3
            fit_params, rebinned_binc = self._fit_energy_res(
                tE_binc, rE_binc, eff_area, n_components
            )

            # Min and max values
            imin = 5
            imax = -15

            e_min = rebinned_binc[imin]
            e_max = rebinned_binc[imax]

            # Degree of polynomial
            polydeg = 5

            log_rebinned = np.log10(rebinned_binc)
            poly_params_mu = np.zeros((n_components, polydeg + 1))

            poly_params_sd = np.zeros_like(poly_params_mu)
            for i in range(n_components):
                poly_params_mu[i] = np.polyfit(
                    log_rebinned[imin:imax], fit_params[:, 2 * i][imin:imax], polydeg
                )
                poly_params_sd[i] = np.polyfit(
                    log_rebinned[imin:imax],
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
            self.poly_params_mu = poly_params_mu
            self.poly_params_sd = poly_params_sd
            self.poly_limits = poly_limits
            self.plot_fit_params(fit_params, rebinned_binc)
            self.plot_parameterizations(
                tE_binc, rebinned_binc, rE_binc, fit_params, eff_area
            )

        # poly params are now set
        self.poly_params_mu = poly_params_mu
        self.poly_params_sd = poly_params_sd
        self.poly_limits = poly_limits

    def _calc_resolution(self, param_dict: dict):
        pass

    def prob_Edet_above_threshold(self, true_energy, threshold_energy):
        """
        P(Edet > Edet_min | E) for use in precomputation. 
        """

        model = self.make_cumulative_model(self.n_components)

        prob = np.zeros_like(true_energy)
        model_params: List[float] = []
        for comp in range(self.n_components):
            mu = np.poly1d(self.poly_params_mu[comp])(np.log10(true_energy))
            sigma = np.poly1d(self.poly_params_sd[comp])(np.log10(true_energy))
            model_params += [mu, sigma]

        prob = model(np.log10(threshold_energy), model_params)

        return prob


class NorthernTracksAngularResolution(UserDefinedFunction):
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

    DATA_PATH = "NorthernTracksAngularRes.csv"
    CACHE_FNAME = "angular_reso_tracks.npz"

    def __init__(self, mode: DistributionMode = DistributionMode.PDF) -> None:

        if mode == DistributionMode.PDF:

            UserDefinedFunction.__init__(
                self,
                "NorthernTracksAngularResolution",
                ["true_energy", "true_dir", "reco_dir"],
                ["real", "vector", "vector"],
                "real",
            )
        else:
            UserDefinedFunction.__init__(
                self,
                "NorthernTracksAngularResolution_rng",
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
                clipped_log_e,
                self.poly_params,
                "NorthernTracksAngularResolutionPolyCoeffs",
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
        """See base class"""

        # Check cache
        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                data = np.load(fr)
                self.poly_params = data["poly_params"]
                self.e_min = float(data["e_min"])
                self.e_max = float(data["e_max"])
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


class DetectorModel(metaclass=ABCMeta):
    def __init__(self, mode: DistributionMode = DistributionMode.PDF):
        self._mode = mode

    @property
    def effective_area(self):
        return self._get_effective_area()

    @abstractmethod
    def _get_effective_area(self):
        return self.__get_effective_area()

    @property
    def energy_resolution(self):
        return self._get_energy_resolution()

    @abstractmethod
    def _get_energy_resolution(self):
        return self._energy_resolution

    @property
    def angular_resolution(self):
        return self._get_angular_resolution()

    @abstractmethod
    def _get_angular_resolution(self):
        self._angular_resolution


class NorthernTracksDetectorModel(DetectorModel):
    """
    Implements the detector model for the NT sample

    Parameters:
        mode: DistributionMode
            Set mode to either RNG or PDF

    """

    def __init__(self, mode: DistributionMode = DistributionMode.PDF):
        DetectorModel.__init__(self, mode)

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
