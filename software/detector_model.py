"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod
from typing import Union, List, Sequence
import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from cache import Cache
from parameterizations import (
    Parameterization,
    TExpression,
    VMFParameterization,
    PolynomialParameterization,
    TruncatedParameterization,
    MixtureParameterization)
from fitting_tools import Residuals


Cache.set_cache_dir(".cache")


class EffectiveArea(metaclass=ABCMeta):
    """
    Implements baseclass for effective areas.


    Every implementation of an effective area has to define a setup method,
    that will take care of downloading required files, creating splines, etc.

    The effective areas can depend on multiple quantities (ie. energy,
    direction, time, ..)
    """

    """
    Parameters on which the effective area depends.
    Overwrite when subclassing
    """
    PARAMETERS: Union[None, List] = None

    def __call__(self, **kwargs):
        """
        Return the effective area for variables given in kwargs
        """
        if (set(self.PARAMETERS) - kwargs.keys()):
            raise ValueError("Not all required parameters passed to call")
        else:
            self._calc_effective_area(kwargs)

    @abstractmethod
    def _calc_effective_area(
            self,
            param_dict: dict) -> float:
        pass

    @abstractmethod
    def setup(self) -> None:
        """
        Download and or build all the required input data for calculating
        the effective area
        """
        pass


class Resolution(Parameterization, metaclass=ABCMeta):
    """
    Base class for parameterizing resolutions
    """

    PARAMETERS: Union[None, List] = None

    def __init__(self, inputs: List[TExpression]):
        Parameterization.__init__(self, inputs)

    def __call__(self, **kwargs):
        """
        Return the resolution for variables given in kwargs
        """
        return self._calc_resolution(kwargs)

    @abstractmethod
    def _calc_resolution(
            self,
            param_dict: dict):
        pass

    @abstractmethod
    def setup(self):
        """
        Download and or build all the required input data for calculating
        the resolution
        """
        pass


class NorthernTracksEffectiveArea(EffectiveArea):
    """
    Effective area for the two-year Northern Tracks release:
    https://icecube.wisc.edu/science/data/HE_NuMu_diffuse

    """

    PARAMETERS = ["true_energy", "true_cos_zen"]

    def _calc_effective_area(
            self,
            param_dict: dict):
        pass

    def setup(self):
        """See base class"""


class NorthernTracksEnergyResolution(  # type: ignore
        MixtureParameterization,
        Resolution):

    """
    Energy resolution for Northern Tracks Sample

    Data from https://arxiv.org/pdf/1811.07979.pdf
    """

    DATA_PATH = "../dev/statistical_model/aeff_input_tracks/effective_area.h5"
    CACHE_FNAME = "energy_reso_tracks.npz"

    def __init__(self, inputs: TExpression) -> None:
        """
        Args:
            inputs: List[TExpression]
                First item is true energy, second item is true
                position
        """
        Resolution.__init__(self, [inputs])
        self.poly_params: Union[None, np.ndarray] = None
        self.e_min: Union[None, float] = None
        self.e_max: Union[None, float] = None

        self.setup()

    def _fit_energy_res(
            self,
            tE_binc: np.ndarray,
            rE_binc: np.ndarray,
            eff_area: np.ndarray,
            n_components: int
            ) -> np.ndarray:
        import scipy.stats as stats  # type: ignore
        from scipy.optimize import least_squares  # type: ignore
        fit_params = []
        # Rebin to have higher statistics at upper
        # and lower end of energy range
        rebin = 3
        rebinned_binc = np.zeros(int(len(tE_binc)/rebin))
        logrEbins = np.log10(rE_binc)

        def _model(x, pars):
            result = 0
            for i in range(n_components):
                result += 1/n_components*stats.lognorm.pdf(x, scale=pars[2*i], s=pars[2*i+1])
            return result

        # Fitting loop
        for index in range(int(len(tE_binc)/rebin)):
            # Calculate rebinned bin-centers as mean of first and
            # last bin being summed
            rebinned_binc[index] = 0.5*(
                tE_binc[[index*rebin, rebin*(index+1)-1]]).sum()

            # Calculate the energy resolution for this true-energy bin
            e_reso = eff_area.sum(axis=1)[index*rebin:(index+1)*rebin]
            e_reso = e_reso.sum(axis=0)
            if e_reso.sum() > 0:
                # Normalize to prob. density / bin
                e_reso = e_reso/e_reso.sum() / (logrEbins[1]-logrEbins[0])

                residuals = Residuals((logrEbins, e_reso), _model)

                # Calculate seed as mean of the resolution to help minimizer
                seed_mu = np.average(logrEbins, weights=e_reso)
                if ~np.isfinite(seed_mu):
                    seed_mu = 3

                seed = np.zeros(n_components*2)
                bounds_lo: List[float] = []
                bounds_hi: List[float] = []
                for i in range(n_components):
                    seed[2*i] = seed_mu + 0.1*(i+1)
                    seed[2*i+1] = 0.02
                    bounds_lo += [0, 0.01]
                    bounds_hi += [8, 1]

                res = least_squares(
                    residuals,
                    seed,
                    bounds=(bounds_lo, bounds_hi),
                )

                # Check for label swapping
                mu_indices = np.arange(0, stop=n_components*2, step=2)
                mu_order = np.argsort(res.x[mu_indices])

                this_fit_pars: List = []
                for i in range(n_components):
                    mu_index = mu_indices[mu_order[i]]
                    this_fit_pars += [res.x[mu_index], res.x[mu_index+1]]
                fit_params.append(this_fit_pars)
            else:
                fit_params.append(np.zeros(2*n_components))
        fit_params = np.asarray(fit_params)
        return fit_params, rebinned_binc

    def plot_fit_params(
            self,
            fit_params: np.ndarray,
            poly_params: Sequence[np.ndarray],
            poly_limits: Sequence[float],
            rebinned_binc: np.ndarray) -> None:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        xs = np.linspace(*np.log10(poly_limits), num=100)
        n_comp = poly_params[0].shape[0]

        for comp in range(n_comp):
            params_mu = poly_params[0][comp]
            axs[0].plot(xs, np.poly1d(params_mu)(xs))
            axs[0].plot(
                np.log10(rebinned_binc),
                fit_params[:, 2*comp],
                label="Mean {}".format(comp))

            params_sigma = poly_params[1][comp]
            axs[1].plot(xs, np.poly1d(params_sigma)(xs))
            axs[1].plot(
                np.log10(rebinned_binc),
                fit_params[:, 2*comp+1],
                label="SD {}".format(comp))
        plt.savefig("plot.png")

    def setup(self) -> None:
        # Load Aeff data

        # Check cache
        if self.CACHE_FNAME in Cache:
            pass
        else:
            import h5py  # type: ignore
            with h5py.File("../dev/statistical_model/4_tracks_and_cascades/aeff_input_tracks/effective_area.h5", 'r') as f:
                eff_area = f['2010/nu_mu/area'][()]
                # True Energy [GeV]
                tE_bin_edges = f['2010/nu_mu/bin_edges_0'][:]
                # cos(zenith)
                # cosz_bin_edges = f['2010/nu_mu/bin_edges_1'][:]
                # Reco Energy [GeV]
                rE_bin_edges = f['2010/nu_mu/bin_edges_2'][:]

            tE_binc = 0.5*(tE_bin_edges[:-1]+tE_bin_edges[1:])
            rE_binc = 0.5*(rE_bin_edges[:-1]+rE_bin_edges[1:])
            n_components = 3
            fit_params, rebinned_binc = self._fit_energy_res(
                tE_binc,
                rE_binc,
                eff_area,
                n_components)

            # Min and max values
            imin = 5
            imax = -15

            e_min = rebinned_binc[imin]
            e_max = rebinned_binc[imax]

            # Degree of polynomial
            polydeg = 5

            log_rebinned = np.log10(rebinned_binc)
            poly_params_mu = np.zeros((n_components, polydeg+1))

            poly_params_sd = np.zeros_like(poly_params_mu)
            for i in range(n_components):
                poly_params_mu[i] = np.polyfit(
                    log_rebinned[imin:imax],
                    fit_params[:, 2*i][imin:imax],
                    polydeg)
                poly_params_sd[i] = np.polyfit(
                    log_rebinned[imin:imax],
                    fit_params[:, 2*i+1][imin:imax],
                    polydeg)

            self.plot_fit_params(fit_params, [poly_params_mu, poly_params_sd],
                                 [e_min, e_max], rebinned_binc)


    def _calc_resolution(
            self,
            param_dict: dict):
        pass


class NorthernTracksAngularResolution(  # type: ignore
        VMFParameterization,
        Resolution):
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

    def __init__(self, inputs: List[TExpression]) -> None:
        """
        Args:
            inputs: List[TExpression]
                First item is true energy, second item is true
                position
        """
        Resolution.__init__(self, inputs)
        self.poly_params: Union[None, np.ndarray] = None
        self.e_min: Union[None, float] = None
        self.e_max: Union[None, float] = None

        self.setup()

        VMFParameterization.__init__(self, inputs, self._kappa)

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
                names=["energy", "resolution"],
                )

            # Kappa parameter of VMF distribution
            data["kappa"] = 1.38 / np.radians(data.resolution)**2

            self.poly_params = np.polyfit(data.energy, data.kappa, 5)
            self.e_min = float(data.energy.min())
            self.e_max = float(data.energy.max())

            # Save polynomial
            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                np.savez(
                    fr,
                    poly_params=self.poly_params,
                    e_min=10**data.energy.min(),
                    e_max=10**data.energy.max())

        # Clip true energy
        clipped_e = TruncatedParameterization(
            self._inputs[0],
            self.e_min,
            self.e_max)

        self._kappa = PolynomialParameterization(
            clipped_e,
            self.poly_params)


class DetectorModel(metaclass=ABCMeta):

    @property
    def effective_area(self):
        return self._get_effective_area()

    @abstractmethod
    def _get_effective_area(self):
        return self.__get_effective_area

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


if __name__ == "__main__":

    e_true = "E_true"
    pos_true = "pos_true"
    ntp = NorthernTracksAngularResolution([e_true, pos_true])

    print(ntp.to_stan())

    nte = NorthernTracksEnergyResolution(e_true)
