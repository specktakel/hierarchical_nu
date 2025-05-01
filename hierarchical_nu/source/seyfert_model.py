import logging
from typing import Union, Iterable
from functools import lru_cache
import astropy.units as u
import numpy as np
import scipy
from scipy.integrate import dblquad, quad, trapezoid
from scipy.interpolate import RectBivariateSpline, make_smoothing_spline
import h5py
import logging

from .flux_model import SpectralShape
from .parameter import Parameter
from ..backend import (
    UserDefinedFunction,
    StanArray,
    StringExpression,
    FunctionCall,
    ReturnStatement,
    ForwardVariableDef,
)


logger = logging.getLogger(__name__)


class SeyfertNuMuSpectrum(SpectralShape):
    _lower_energy = 1e2 * u.GeV
    _upper_energy = 1e7 * u.GeV
    _name = "SeyfertII"

    def __init__(
        self,
        P: Parameter,
        eta: Parameter,
        energy_points: int = 80,
        eta_points: int = 100,
    ):
        super().__init__(self)
        self._parameters["eta"] = eta
        self._parameters["P"] = P
        self._energy_points = energy_points
        self._eta_points = eta_points

        self._filename = "/remote/ceph2/user/k/kuhlmann/seyfert_nu_spectra/spectra/hnu_input_ngc_1068.h5"
        # Load grid of flux values
        with h5py.File(self._filename, "r") as f:
            energy = f["Enu"][()]
            flux_grid = (
                f["numu_flux"][()] * 1e4
            )  # conversion from cm-2 to m-2, is already normalised to P_R = 1
            eta = f["eta"][()]
        flux_grid[flux_grid == 0.0] = flux_grid[flux_grid != 0.0].min()
        self._energy = energy << u.GeV
        self._flux_grid = flux_grid << 1 / u.GeV / u.s / u.m**2
        self._eta = eta

        self._spline = RectBivariateSpline(
            np.log10(energy), eta, np.log10(flux_grid).T, kx=1, ky=3
        )

        # properly normalise everything: stan needs pdf in energy on a grid of eta
        # thus divide all slices by their integral over energy -> pdf(E;eta)
        # pass to stan implementation
        pdf_grid = np.zeros(flux_grid.shape)
        integral_grid = np.zeros(self._eta.shape)
        energy_flux_grid = np.zeros(self._eta.shape)
        eta = self._parameters["eta"]
        eta_init_val = eta.value
        fixed = eta.fixed
        eta.fixed = False

        P = self._parameters["P"]
        fixed_P = P.fixed
        P_init_val = P.value
        P.fixed = False
        P.value = 1.0

        for c, e in enumerate(self._eta):
            eta.value = e
            integral = self.total_flux_int.to_value(1 / u.m**2 / u.s)
            pdf_grid[c] = (
                self._flux_grid[c].to_value(1 / u.GeV / u.m**2 / u.s) / integral
            )
            integral_grid[c] = integral
            e_flux_int = self.total_flux_density.to_value(u.GeV / u.m**2 / u.s)
            energy_flux_grid[c] = e_flux_int
        eta.value = eta_init_val
        eta.fixed = fixed
        P.value = P_init_val
        P.fixed = fixed_P

        self._integral_grid = integral_grid
        self._energy_flux_grid = energy_flux_grid
        self._log_pdf_spline = RectBivariateSpline(
            np.log10(energy), self._eta, np.log(pdf_grid).T, kx=1, ky=3
        )

        log_energy_grid = np.linspace(
            np.log10(self._lower_energy.to_value(u.GeV)),
            np.log10(self._upper_energy.to_value(u.GeV)),
            self.energy_points,
        )
        eta_grid = np.linspace(self._eta.min(), self._eta.max(), self.eta_points)
        self.eta_grid = eta_grid
        self.log_pdf_grid = self._log_pdf_spline(log_energy_grid, eta_grid)
        self.log_energy_grid = log_energy_grid

        # Smoothen integral_grid
        self._integral_grid_spline = make_smoothing_spline(
            self._eta, self._integral_grid
        )
        self._flux_conv_spline = make_smoothing_spline(
            self._eta, self._integral_grid / self._energy_flux_grid
        )
        # make spline out of energy density

    @property
    def eta_points(self):
        return self._eta_points

    @property
    def energy_points(self):
        return self._energy_points

    def make_stan_lpdf_func(self, f_name):
        return

    def make_stan_flux_conv_func(self, f_name):
        return

    def make_stan_functions(self):
        lpdf = self._make_stan_lpdf_func()
        flux_tab, flux_conv = self._make_stan_flux_conv_func()
        return lpdf, flux_tab, flux_conv

    def pdf(self, energy: u.GeV, Emin: u.GeV, Emax: u.GeV, apply_lim: bool = True):
        logE = np.log10(energy.to_value(u.GeV))
        eta = self._parameters["eta"].value
        return np.exp(self._log_pdf_spline(logE, eta).squeeze())

    def _make_stan_lpdf_func(self):
        # PDF
        func = UserDefinedFunction(
            "SeyfertNuMu_logpdf",
            ["true_energy", "eta"],
            ["real", "real"],
            "real",
        )

        with func:
            # takes true energy and eta as arguments
            etrue = StringExpression("true_energy")
            log_energy = FunctionCall([etrue], "log10")
            eta = StringExpression("eta")
            eta_grid = StanArray("eta_grid", "real", self.eta_grid)
            log_energy_grid = StanArray("log_energy_grid", "real", self.log_energy_grid)
            log_pdf_grid = StanArray("pdf_grid", "real", self.log_pdf_grid)

            ReturnStatement(
                [
                    FunctionCall(
                        [log_energy, eta, log_energy_grid, eta_grid, log_pdf_grid],
                        "interp2d",
                    )
                ]
            )
        return func

    def _make_stan_flux_conv_func(self):
        # Integrated flux, needed for likelihood
        flux_tab = UserDefinedFunction(
            "integrated_flux",
            ["eta"],
            ["real"],
            "real",
        )
        # make a spline out of self._integral_grid, evaluate at eta points and use linear interpolation

        splined_values = self._integral_grid_spline(self.eta_grid)
        with flux_tab:
            eta = StringExpression("eta")
            eta_grid = StanArray("eta_grid", "real", self.eta_grid)
            integral_grid = StanArray("integral_grid", "real", splined_values)
            ReturnStatement(
                [FunctionCall([eta_grid, integral_grid, eta], "interpolate")]
            )

        # Flux conv, only needed when using luminosities as derived parameters
        splined_values = self._flux_conv_spline(self.eta_grid)
        flux_conv = UserDefinedFunction(
            "flux_conv",
            ["eta"],
            ["real"],
            "real",
        )
        with flux_conv:
            eta = StringExpression("eta")
            eta_grid = StanArray("eta_grid", "real", self.eta_grid)
            conv_grid = StanArray("conv_grid", "real", splined_values)
            ReturnStatement([FunctionCall([eta_grid, conv_grid, eta], "interpolate")])
        return flux_tab, flux_conv

    def _spline_log_interpolation(self, logE: Union[Iterable, float], eta: float):
        """
        Evaluate spline representation of flux model
        :param logE: log10(E/GeV), Iterable or float
        :param eta: eta
        """

        logE = np.atleast_1d(logE)
        return np.power(10, self._spline(logE, eta)).squeeze()

    @u.quantity_input
    def __call__(self, energy: u.GeV) -> 1 / (u.m**2 * u.s * u.GeV):
        """
        Returns differential flux. Uses eta and P from self._parameters
        :param energy: Energy at which to evaluate flux
        """

        eta = self._parameters["eta"].value
        P = self._parameters["P"].value
        eval = (
            self._spline_log_interpolation(
                np.log10(energy.to_value(u.GeV)),
                eta,
            )
            << 1 / u.GeV / u.s / u.m**2
        )
        flux = eval * P
        return flux

    @u.quantity_input
    def integral(self, lower: u.GeV, upper: u.GeV) -> 1 / u.m**2 / u.s:
        logElow = np.log10(lower.to_value(u.GeV))
        logEhigh = np.log10(upper.to_value(u.GeV))
        eta = self._parameters["eta"].value
        P = self._parameters["P"].value

        def integrand(logE, eta):
            return (
                self._spline_log_interpolation(logE, eta)
                * np.power(10, logE)
                * np.log(10)
            )

        integral = quad(integrand, logElow, logEhigh, (eta))[0] << 1 / u.s / u.m**2
        return P * integral

    @property
    def total_flux_int(self) -> 1 / (u.m**2 * u.s):
        """
        Return number flux integrated over energy. Uses eta and P from self._parameters
        """

        integral = self.integral(self._lower_energy, self._upper_energy)
        return integral

    @property
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        """
        Returns energy flux over the entire energy range. Uses eta and P from self._parameters
        """

        eta = self._parameters["eta"].value
        P = self._parameters["P"].value
        logElow = np.log10(self._lower_energy.to_value(u.GeV))
        logEhigh = np.log10(self._upper_energy.to_value(u.GeV))

        def integrand(logE, eta):
            return (
                self._spline_log_interpolation(logE, eta)
                * np.power(10, 2 * logE)
                * np.log(10)
            )

        integral = quad(integrand, logElow, logEhigh, (eta), limit=100)
        val = integral[0]
        err = integral[1]
        ratio = err / val
        if ratio > 1e-3:
            logger.warning(f"Flux density integral has a large error of {100*ratio:.3f}%.")
        return (P * val * u.GeV / u.s / u.m**2).to(u.erg / u.s / u.cm**2)

    @classmethod
    def make_stan_sampling_func(cls, f_name):
        # Not needed? replaced by envelope sampling
        return

    @classmethod
    def make_stan_lpdf_func(cls, f_name):
        # Create stan specific class, copy from AtmosphericNuMuFlux
        return

    @classmethod
    def make_stan_flux_conv_func(cls, f_name):
        # Should be a lookup table
        return
