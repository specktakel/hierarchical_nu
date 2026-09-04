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
from pathlib import Path
import os
import sys

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

    # Replace by stuff from the simulation?
    _lower_energy = 1e2 * u.GeV
    _upper_energy = 1e7 * u.GeV
    _name = "SeyfertII"

    eta_grid = np.concatenate((np.arange(1., 30.,), np.arange(30., 101, 2)))

    def __init__(
        self,
        logLx: float,
        P: Parameter,
        eta: Parameter,
        z: float,
        energy_points: int = 80,
        eta_points: int = 100,
        source_name: str = "ps",
    ):
        """
        Implements source model of Seyfert II galaxies.
        Parameters:
            P: Parameter
                Pressure ratio of source, realistically capped at 0.5
            eta: Parameter
                Inverse magnetic turbulence strength, limited by simulation inputs to (2, 150)
            logLx: float
                log10(x-ray luminosity / (erg / s))
            z: float
                Source redshift
            energy_points: int
                Number of energy grid points used for interpolation in stan (less=faster)
            eta_points:
                Number of eta grid points used for interpolation in stan (less=faster)
            source_name: str
                Name to use as prefix in stan function names, needed for mutli ps fits
        """

        super().__init__(self)
        self._parameters["eta"] = eta
        self._parameters["P"] = P
        self._source_name = source_name
        self._logLx = logLx

        # Load appropriate file containing the energy density in the source environment
        # and, together with redshift, convert it into a number flux at the detector

        # TODO properly replace this
        sys.path.append(os.path.expanduser("~/icecube/seyfert_spectra"))
        from nu_pop_model.diffuse_flux import mu_nu_flux, photon_flux

        path_to_simulations = Path(
            #os.path.expanduser("~/icecube/seyfert_spectra/combined_files_eta_1")
            os.path.expanduser("~/icecube/seyfert_spectra/combined_nu_gamma")
        )
        self._filename = (
            #path_to_simulations / f"neutrino_density_logLx_{np.round(logLx, 2):.2f}.h5"
            path_to_simulations / f"neutrino_gamma_density_logLx_{np.round(logLx, 2):.2f}.h5"
        )

        with h5py.File(self._filename, "r") as f:
            # get information about the spectrum
            #Enu = f["energy"][()] * u.Unit(f["energy_unit"][()].decode("ascii"))
            #energy_density = f["energy_density"][()] * u.Unit(
            #    f["energy_density_unit"][()].decode("ascii")
            #)
            #gamma_density = None   # add gammas to combined files
            Enu = f["Enu"][()] * u.Unit(f["Enu_unit"][()].decode("ascii"))
            nu_density = f["nu_density"][()] * u.Unit(f["nu_density_unit"][()].decode("ascii"))
            gamma_density = f["gamma_density"][()] * u.Unit(f["gamma_density_unit"][()].decode("ascii"))
            Egamma = f["Egamma"][()] * u.Unit(f["Egamma_unit"][()].decode("ascii"))
            eta = f["eta"][()]

        self._eta = eta
        self._energy = Enu
        self._gamma_energy = Egamma
        self._gamma_density = gamma_density

        nu_flux_grid = np.zeros((eta.size, Enu.size))
        gamma_flux_grid = np.zeros((eta.size, Egamma.size))
        for c, e in enumerate(eta):
            nu_flux_grid[c] = (
                mu_nu_flux(   # query the differential flux
                    Enu.to_value(u.GeV),
                    Enu.to_value(u.GeV),
                    nu_density[c].to_value(1 / u.GeV / u.s),
                    z,
                )
                * 1e4
            )  # conversion from cm-2 to m-2

            gamma_flux_grid[c] = (
                photon_flux(
                    Egamma.to_value(u.GeV),
                    Egamma.to_value(u.GeV),
                    gamma_density[c].to_value(1 / u.GeV / u.s),
                    z,
                )
                * 1e4
            )  # conversion from cm-2 to m-2
        nu_flux_grid[nu_flux_grid == 0.0] = nu_flux_grid[nu_flux_grid != 0.0].min()   # mask out zeros because we log somewhere
        gamma_flux_grid[gamma_flux_grid == 0] = gamma_flux_grid[gamma_flux_grid != 0].min()
        self._nu_flux_grid = nu_flux_grid << 1 / u.GeV / u.s / u.m**2
        self._gamma_flux_grid = gamma_flux_grid << 1 / u.GeV / u.s / u.m**2
        self._energy_points = energy_points
        self._eta_points = eta_points

        self._log_flux_spline = RectBivariateSpline(
            np.log10(self._energy.to_value(u.GeV)),
            self._eta,
            np.log10(self._nu_flux_grid.to_value(1 / u.GeV / u.s / u.m**2)).T,
            kx=1,
            ky=3,
        )

        self._log_gamma_flux_spline = RectBivariateSpline(
            np.log10(self._gamma_energy.to_value(u.GeV)),
            self._eta,
            np.log10(self._gamma_flux_grid.to_value(1 / u.GeV / u.s / u.m**2)).T,
            kx=1,
            ky=3,
        )

        # properly normalise everything: stan needs pdf in energy on a grid of eta
        # thus divide all slices by their integral over energy -> pdf(E;eta)
        # pass to stan implementation
        pdf_grid = np.zeros(nu_flux_grid.shape)
        integral_grid = np.zeros(self._eta.shape)
        energy_flux_grid = np.zeros(self._eta.shape)
        eta = self._parameters["eta"]
        eta_init_val = eta.value
        fixed = eta.fixed
        eta.fixed = False

        # Fix pressure ratio to 1 for the following calculation
        # this is needed because we use integration methods of this class
        # which multiply results by the pressure ratio
        P = self._parameters["P"]
        par_range = P.par_range
        P.par_range = (0, 10)
        fixed_P = P.fixed
        P_init_val = P.value
        P.fixed = False
        P.value = 1.0
        
        for c, e in enumerate(self._eta):
            eta.value = e
            integral = self.total_flux_int.to_value(1 / u.m**2 / u.s)
            pdf_grid[c] = (
                self._nu_flux_grid[c].to_value(1 / u.GeV / u.m**2 / u.s) / integral
            )
            integral_grid[c] = integral
            e_flux_int = self.total_flux_density.to_value(u.GeV / u.m**2 / u.s)
            energy_flux_grid[c] = e_flux_int
        eta.value = eta_init_val
        eta.fixed = fixed
        P.value = P_init_val
        P.fixed = fixed_P
        P.par_range = par_range

        self._integral_grid = integral_grid
        self._energy_flux_grid = energy_flux_grid
        self._log_pdf_spline = RectBivariateSpline(
            np.log10(Enu.to_value(u.GeV)), self._eta, np.log(pdf_grid).T, kx=1, ky=3
        )

        log_energy_grid = np.linspace(
            np.log10(self._lower_energy.to_value(u.GeV)),
            np.log10(self._upper_energy.to_value(u.GeV)),
            self.energy_points,
        )
        # eta_grid = np.linspace(self._eta.min(), self._eta.max(), self.eta_points)

        self.log_pdf_grid = self._log_pdf_spline(log_energy_grid, self.eta_grid)
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
    def source_name(self):
        return self._source_name

    @property
    def name(self):
        return self._name

    @property
    def eta_points(self):
        return self._eta_points

    @property
    def energy_points(self):
        return self._energy_points

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
            f"{self.source_name}_SeyfertNuMu_logpdf",
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
            f"{self.source_name}_integrated_flux",
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
            f"{self.source_name}_flux_conv",
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
        return np.power(10, self._log_flux_spline(logE, eta)).squeeze()

    def _gamma_spline_log_interpolation(self, logE: Union[Iterable, float], eta: float):
        """
        Evaluate spline representation of photon flux model
        :param logE: log10(E/GeV), Iterable or float
        :param eta: eta
        """

        logE = np.atleast_1d(logE)
        return np.power(10, self._log_gamma_flux_spline(logE, eta)).squeeze()

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
    def gamma_flux(self, energy: u.GeV) -> 1 / (u.GeV * u.m**2 * u.s):
        eta = self._parameters["eta"].value
        P = self._parameters["P"].value
        eval = (
            self._gamma_spline_log_interpolation(
                np.log10(energy.to_value(u.GeV)),
                eta,
            )
            << 1 / u.GeV / u.s / u.m**2
        )
        flux = eval * P
        return flux

    @u.quantity_input
    def precompute_gamma_energy_flux(self, energy_bins: u.GeV):
        """
        Precomputes for all eta the expected gamma-ray energy flux over the provided binning.
        Multiply in the end by the pressure ratio to get properly scaled value to use in a likelihood.
        """

        eta_par = self._parameters["eta"]
        eta_init = eta_par.value
        eta_fixed = eta_par.fixed
        eta_par.fixed = False
        P = self._parameters["P"].value

        fluxes = []

        for e in self.eta_grid:
            eta_par.value = e
            # normalise to pressure ratio, use default stan units
            eflux = self.gamma_energy_flux_integral_per_bin(energy_bins).to_value(u.GeV / u.s / u.m**2) / P
            fluxes.append(eflux)



        # restore param
        eta_par.value = eta_init
        eta_par.fixed = eta_fixed

        return fluxes

    @u.quantity_input
    def gamma_energy_flux_integral_per_bin(self, energy_bins: u.GeV) -> u.GeV / (u.m**2 * u.s):
        eta = self._parameters["eta"].value
        P = self._parameters["P"].value

        Emin = energy_bins[:-1]
        Emax = energy_bins[1:]

        logElow = np.log10(Emin.to_value(u.GeV))
        logEhigh = np.log10(Emax.to_value(u.GeV))

        def integrand(logE, eta):
            return (
                self._gamma_spline_log_interpolation(logE, eta)
                * np.power(10, 2 * logE)
                * np.log(10)
            )
        integral = []
        for (lEl, lEh) in zip(logElow, logEhigh):
            integral.append(
                quad(integrand, lEl, lEh, (eta))[0]
            )
        integral = P * np.array(integral) << u.GeV / u.m**2 / u.s

        return integral

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

        integral = quad(integrand, logElow, logEhigh, (eta), limit=200)
        val = integral[0]
        err = integral[1]
        ratio = err / val
        if ratio > 1e-3:
            logger.warning(
                f"Flux density integral has a large error of {100*ratio:.3f}%."
            )
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
