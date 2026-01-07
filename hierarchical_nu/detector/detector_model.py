"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from typing import List
import os
from astropy import units as u

from ..utils.cache import Cache
from ..backend import (
    UserDefinedFunction,
    DistributionMode,
    StanGenerator,
)
from ..stan.interface import STAN_GEN_PATH

from ..utils.fitting_tools import Residuals

import logging

logger = logging.getLogger(__name__)
Cache.set_cache_dir(".cache")


class EffectiveArea(UserDefinedFunction, metaclass=ABCMeta):
    """
    Implements baseclass for effective areas.

    Every implementation of an effective area has to define a setup method,
    that will take care of downloading required files, creating parametrizations etc.

    The effective areas can depend on multiple quantities (ie. energy,
    direction, time, ..)
    """

    def _make_spline(self):
        log_tE_lower_bin_edges = np.log10(self._tE_bin_edges[:-1])
        log_tE_upper_bin_edges = np.log10(self._tE_bin_edges[1:])
        """
        log_tE_bin_c = np.concatenate(
            (
                np.array([log_tE_lower_bin_edges[0]]),
                (log_tE_lower_bin_edges + log_tE_upper_bin_edges) / 2,
                np.array([log_tE_upper_bin_edges[-1]]),
            ),
        )
        """
        log_tE_bin_c = (log_tE_lower_bin_edges + log_tE_upper_bin_edges) / 2
        log_tE_bin_c[0] = log_tE_lower_bin_edges[0]
        log_tE_bin_c[-1] = log_tE_upper_bin_edges[-1]
        tE_bin_c = np.power(10, log_tE_bin_c)

        cosz_lower = self._cosz_bin_edges[:-1]
        cosz_upper = self._cosz_bin_edges[1:]
        cosz_c = np.concatenate(
            (
                np.array([cosz_lower[0]]),
                (cosz_lower + cosz_upper) / 2,
                np.array([cosz_upper[-1]]),
            )
        )

        # Duplicate slice at lowest and highest energy
        """
        to_be_splined_aeff = np.concatenate(
            (
                np.atleast_2d(self.eff_area[0, :]),
                self.eff_area,
                (np.atleast_2d(self.eff_area[-1, :])),
            ),
            axis=0,
        )
        """
        to_be_splined_aeff = self.eff_area.copy()
        # Duplicate slice at cosz=-1 (vertical upgoing) and cosz=1 (vertical downgoing)
        to_be_splined_aeff = np.concatenate(
            (
                np.atleast_2d(to_be_splined_aeff[:, 0]).T,
                to_be_splined_aeff,
                (np.atleast_2d(to_be_splined_aeff[:, -1]).T),
            ),
            axis=1,
        )
        non_zero_min = to_be_splined_aeff[to_be_splined_aeff > 0.0].min()
        to_be_splined_aeff[to_be_splined_aeff == 0.0] = non_zero_min

        # fill_value = np.log10(np.min(to_be_splined_aeff[to_be_splined_aeff > 0.]))
        self._eff_area_spline = RegularGridInterpolator(
            (log_tE_bin_c, cosz_c),
            np.log10(to_be_splined_aeff),
            bounds_error=False,
            fill_value=np.log10(non_zero_min),
            # fill_value=0.0,
            method="linear",
        )

    def spline(self, logE, cosz):
        return self.eff_area_spline((logE, cosz))

    @abstractmethod
    def setup(self) -> None:
        """
        Download and or build all the required input data for calculating
        the effective area. Alternatively load from necessary info from
        cache if already stored.

        Setup must provide the info to fill the properties listed below,
        and be called in the initialisation.
        """

        pass

    @property
    def eff_area(self):
        """
        2D histogram of effective area values, with
        Etrue on axis 0 and cosz on the axis 1.
        """

        return self._eff_area

    @property
    def tE_bin_edges(self):
        """
        True energy bin edges corresponding the the
        histogram in eff_area.
        """

        return self._tE_bin_edges

    @property
    def cosz_bin_edges(self):
        """
        cos(zenith) bin edges corresponding to the
        histogram in eff_area.
        """

        return self._cosz_bin_edges

    @property
    def rs_bbpl_params(self):
        """
        Bounded broken power law parameters
        for suitable sampling of the effective
        area.
        """

        return self._rs_bbpl_params

    # @property
    def eff_area_spline(self, vals):
        """
        2D spline of effective area.
        """

        return np.power(10, self._eff_area_spline(vals))

    @abstractmethod
    def generate_code(self):
        """
        Generates stan code.
        """

        pass


class EnergyResolution(metaclass=ABCMeta):
    """
    Abstract base class defining the energy resolution interface.

    Signature for __call__ of UserDefinedFunction for DistributionMode.PDF is
    log10(Etrue): real, log10(Edet): real, omega_det: unit_vector[3]
    even if some parameter might not be used

    Since there are two implementations of
    """

    @abstractmethod
    def setup(self):
        """
        Download and or build all the required input data for calculating
        the energy resolution and setting the properties listed below.
        """

        pass

    @property
    def eres(self):
        """
        2D histogram of P(Ereco | Etrue) with Etrue on axis 0 and
        Ereco on axis 1. Normalised along Ereco for a given Etrue bin.
        """

        return self._eres

    @property
    def tE_bin_edges(self):
        """
        True energy bins in [GeV] corresponding to histogram in eres.
        """

        return self._tE_bin_edges

    @abstractmethod
    def generate_code(self):
        """
        Generates stan code.
        """

        pass


class GridInterpolationEnergyResolution(
    EnergyResolution, UserDefinedFunction, metaclass=ABCMeta
):

    @property
    def log_rE_bin_edges(self):
        return self._log_rE_bin_edges

    @property
    def log_rE_binc(self):
        return self._log_rE_bin_edges

    @property
    def log_tE_bin_edges(self):
        return self._log_tE_bin_edges

    @property
    def log_tE_binc(self):
        return self._log_tE_binc

    @property
    def dec_bin_edges(self):
        return self._dec_bin_edges

    @property
    def dec_binc(self):
        return self._dec_binc

    @property
    def sin_dec_edges(self):
        return self._sin_dec_edges

    @property
    def sin_dec_binc(self):
        return self._sin_dec_binc

    @property
    def evaluations(self):
        return self._evaluations


class LogNormEnergyResolution(EnergyResolution, UserDefinedFunction, metaclass=ABCMeta):

    @property
    def rE_bin_edges(self):
        """
        Reconstucted energy bins in [GeV] corresponding to histogram
        in eres.
        """

        return self._rE_bin_edges

    @property
    def n_components(self):
        """
        Number of components of the lognormal mixture model.
        """

        return self._n_components

    @property
    def poly_params_mu(self):
        """
        Polynomial parameters describing the evolution of
        the lognormal means with true energy.
        """

        return self._poly_params_mu

    @property
    def poly_params_sd(self):
        """
        Polynomial parameters dscribing the evolution of
        the lognormal standard deviations with true energy.
        """

        return self._poly_params_sd

    @property
    def poly_limits(self):
        """
        Limits in true energy [GeV] of the polynomial fit.
        """

        return self._poly_limits

    @staticmethod
    def make_fit_model(n_components):
        """
        Lognormal mixture with n_components.
        """

        # s is width of lognormal
        # scale is ~expectation value
        def _model(x, pars):
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

    def _fit_energy_res(
        self,
        tE_binc: np.ndarray,
        rE_binc: np.ndarray,
        eres: np.ndarray,
        n_components: int,
        rebin: int = 1,
    ) -> np.ndarray:
        """
        Fit a lognormal mixture to P(Ereco | Etrue) in given Etrue bins.
        """

        from scipy.optimize import least_squares

        fit_params = []

        log10_rE_binc = np.log10(rE_binc)
        log10_bin_width = log10_rE_binc[1] - log10_rE_binc[0]

        # Rebin to have higher statistics at upper
        # and lower end of energy range
        rebin_tE_binc = np.zeros(int(len(tE_binc) / rebin))

        # Lognormal mixture
        model = self.make_fit_model(n_components)

        # Fitting loop
        for index in range(len(rebin_tE_binc)):
            rebin_tE_binc[index] = (
                0.5 * (tE_binc[[index * rebin, rebin * (index + 1) - 1]]).sum()
            )

            # Energy resolution for this true-energy bin
            e_reso = eres[index * rebin : (index + 1) * rebin]
            e_reso = e_reso.sum(axis=0)

            if e_reso.sum() > 0:
                # Normalize to prob. density / bin
                e_reso = e_reso / (e_reso.sum() * log10_bin_width)

                residuals = Residuals((log10_rE_binc, e_reso), model)

                # Calculate seed as mean of the resolution to help minimizer
                seed_mu = np.average(log10_rE_binc, weights=e_reso)
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

        return fit_params, rebin_tE_binc

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

        log10_tE_binc = np.log10(tE_binc)
        poly_params_mu = np.zeros((self._n_components, polydeg + 1))

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
            axs[0].plot(xs, np.poly1d(params_mu)(xs), label="poly, mean")
            axs[0].plot(
                np.log10(tE_binc),
                fit_params[:, 2 * comp],
                label="Mean {}".format(comp),
            )

            params_sigma = self._poly_params_sd[comp]
            axs[1].plot(xs, np.poly1d(params_sigma)(xs), label="poly, sigma")
            axs[1].plot(
                np.log10(tE_binc),
                fit_params[:, 2 * comp + 1],
                label="SD {}".format(comp),
            )

        axs[0].set_xlabel("log10(True Energy / GeV)")
        axs[0].set_ylabel("Parameter Value")
        axs[0].legend()
        axs[1].legend()
        plt.tight_layout()
        return fig

    def plot_parameterizations(
        self,
        tE_binc: np.ndarray,
        rE_binc: np.ndarray,
        fit_params: np.ndarray,
        rebin_tE_binc=None,
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

        if self._poly_params_mu is None:
            raise RuntimeError("Run setup() first")

        # Find true energy bins for the chosen plotting energies
        plot_indices = np.digitize(plot_energies, tE_binc)

        if rebin_tE_binc is not None:
            # Parameters are relative to the rebinned histogram
            param_indices = np.digitize(plot_energies, rebin_tE_binc)
            rebin = int(len(tE_binc) / len(rebin_tE_binc))

        log10_rE_binc = np.log10(rE_binc)
        log10_bin_width = log10_rE_binc[1] - log10_rE_binc[0]

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        xs = np.linspace(*np.log10(self._poly_limits), num=100)

        model = self.make_fit_model(self._n_components)
        fl_ax = axs.ravel()

        for i, p_i in enumerate(plot_indices):
            log_plot_e = np.log10(plot_energies[i])

            model_params: List[float] = []
            for comp in range(self.n_components):
                mu = np.poly1d(self._poly_params_mu[comp])(log_plot_e)
                sigma = np.poly1d(self._poly_params_sd[comp])(log_plot_e)
                model_params += [mu, sigma]

            if rebin_tE_binc is not None:
                e_reso = self._eres[
                    int(p_i / rebin) * rebin : (int(p_i / rebin) + 1) * rebin
                ]
                # normalisation of e_reso in case it's not normalised yet
                e_reso = e_reso.sum(axis=0) / (e_reso.sum() * log10_bin_width)
                res = fit_params[param_indices[i]]

            else:
                e_reso = self._eres[p_i]
                res = fit_params[plot_indices[i]]

            fl_ax[i].plot(log10_rE_binc, e_reso, label="input eres")
            fl_ax[i].plot(xs, model(xs, model_params), label="poly evaluated")
            fl_ax[i].plot(xs, model(xs, res), label="nearest bin's parameters")
            fl_ax[i].set_ylim(1e-4, 5)
            fl_ax[i].set_yscale("log")
            fl_ax[i].set_title("True E: {:.1E}".format(tE_binc[p_i]))
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

    @u.quantity_input
    def prob_Edet_above_threshold(
        self,
        true_energy: u.GeV,
        threshold_energy: u.GeV,
        dec: u.rad = 0.0 * u.rad,
        use_lognorm: bool = True,
    ):
        """
        P(Edet > Edet_min | E) for use in precomputation.
        """

        # Truncate input energies to safe range
        energy_trunc = true_energy.to(u.GeV).value
        energy_trunc[energy_trunc < self._pdet_limits[0]] = self._pdet_limits[0]
        energy_trunc[energy_trunc > self._pdet_limits[1]] = self._pdet_limits[1]
        energy_trunc = energy_trunc * u.GeV

        model = self.make_cumulative_model(self.n_components)

        prob = np.zeros_like(energy_trunc)
        model_params: List[float] = []

        for comp in range(self.n_components):
            mu = np.poly1d(self.poly_params_mu[comp])(
                np.log10(energy_trunc.to(u.GeV).value)
            )
            sigma = np.poly1d(self.poly_params_sd[comp])(
                np.log10(energy_trunc.to(u.GeV).value)
            )
            model_params += [mu, sigma]

        prob = 1 - model(np.log10(threshold_energy.to(u.GeV).value), model_params)

        return prob


class AngularResolution(UserDefinedFunction, metaclass=ABCMeta):
    """
    Abstract base class for angular resolution implementation.
    """

    @abstractmethod
    def setup(self):
        """
        Load the angular resolution and transform into the
        properties listed below.
        """

        pass

    @property
    def kappa_grid(self):
        """
        1D array of kappa values with changing energy. Here,
        kappa is the scale parameter of the vMF distribution.
        """

        return self._kappa_grid

    @property
    def Egrid(self):
        """
        Energy in [GeV] corresponding to above kappa values.
        """

        return self._Egrid

    @property
    def poly_params(self):
        """
        Parameters of polynomial that describes the
        evolution of the vMF kappa parameter with energy.
        """

        return self._poly_params

    @property
    def poly_limits(self):
        """
        Mininum and maximum energy in [GeV] over which the
        polynomial is fit.
        """

        return self._poly_limits

    @abstractmethod
    def kappa(self):
        """
        Get kappa in stan sim.

        Meant to be used inside stan code generator.
        """

        pass

    @abstractmethod
    def generate_code(self):
        """
        Generates stan code.
        """

        pass


class DetectorModel(UserDefinedFunction, metaclass=ABCMeta):
    """
    Abstract base class for detector models.
    """

    @u.quantity_input
    def __init__(
        self,
        mode: DistributionMode = DistributionMode.PDF,
        event_type=None,
    ):
        self.mode = mode

        self._event_type = event_type

    @property
    def effective_area(self):
        return self._get_effective_area()

    @abstractmethod
    def _get_effective_area(self):
        return self._get_effective_area()

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
        return self._angular_resolution

    @property
    def event_type(self):
        return self._event_type

    @classmethod
    def generate_code(
        cls,
        mode: DistributionMode,
        path: str = STAN_GEN_PATH,
        rewrite: bool = False,
        **kwargs
    ):
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

            # else:
            #     cls.logger.info("Generating IRF stan code.")

        with StanGenerator() as cg:
            instance = cls(mode=mode, **kwargs)
            instance.effective_area.generate_code()
            instance.angular_resolution.generate_code()
            instance.energy_resolution.generate_code()
            code = cg.generate()
        code = code.removeprefix("functions\n{")
        code = code.removesuffix("\n}\n")
        if not os.path.isdir(path):
            os.makedirs(path)
        if mode == DistributionMode.PDF:
            with open(os.path.join(path, cls.PDF_FILENAME), "w+") as f:
                f.write(code)
            return os.path.join(path, cls.PDF_FILENAME)
        else:
            with open(os.path.join(path, cls.RNG_FILENAME), "w+") as f:
                f.write(code)
            return os.path.join(path, cls.RNG_FILENAME)

    @abstractmethod
    def generate_pdf_function_code(self):
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

        pass

    @abstractmethod
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

        pass
