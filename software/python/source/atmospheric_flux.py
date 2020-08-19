from functools import partial
import pickle

import astropy.units as u
import numpy as np
import scipy
from scipy.integrate import dblquad, quad

from MCEq.core import MCEqRun
import crflux.models as crf


from .flux_model import FluxModel
from ..cache import Cache
from ..backend import (
    UserDefinedFunction,
    StanArray,
    TruncatedParameterization,
    LogParameterization,
    StringExpression,
    FunctionCall,
    ReturnStatement,
    ForwardVariableDef
)

Cache.set_cache_dir(".cache")


class _AtmosphericNuMuFluxStan(UserDefinedFunction):
    def __init__(
            self,
            splined_flux,
            log_energy_grid,
            theta_points=50):

        UserDefinedFunction.__init__(
            self,
            "AtmopshericNumuFlux",
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real")

        self.theta_points = theta_points

        cos_theta_grid = np.linspace(0, 1, self.theta_points)
        cos_theta_centers = 0.5 * (cos_theta_grid[1:] + cos_theta_grid[:-1])

        spl_evals = np.empty((self.theta_points, len(log_energy_grid)))

        for i, cos_theta in enumerate(cos_theta_grid):
            spl_evals[i] = splined_flux(cos_theta, log_energy_grid).squeeze()

        with self:
            spl_evals_stan = StanArray(
                "AtmosphericFluxPolyCoeffs",
                "real",
                spl_evals)

            cos_theta_grid_stan = StanArray("cos_theta_grid", "real", cos_theta_grid)
            log_energy_grid_stan = StanArray("log_energy_grid", "real", log_energy_grid)

            truncated_e = TruncatedParameterization(
                "true_energy", 10**log_energy_grid[0], 10**log_energy_grid[-1])
            log_trunc_e = LogParameterization(truncated_e)

            # abs() since the flux is symmetric around the horizon
            cos_dir = StringExpression(["abs(cos(pi() - acos(true_dir[3])))"])
            cos_theta_bin_index = FunctionCall([cos_dir, cos_theta_grid_stan], "binary_search", 2)

            vect_spl_vals_low = FunctionCall([spl_evals_stan[cos_theta_bin_index]], "to_vector")
            vect_spl_vals_high = FunctionCall([spl_evals_stan[cos_theta_bin_index + 1]], "to_vector")
            vect_log_e_grid = FunctionCall([log_energy_grid_stan], "to_vector")

            interpolated_energy_low = FunctionCall([vect_log_e_grid, vect_spl_vals_low, log_trunc_e], "interpolate", 3)
            interpolated_energy_high = FunctionCall(
                [vect_log_e_grid, vect_spl_vals_high, log_trunc_e], "interpolate", 3)

            vector_log_trunc_e = ForwardVariableDef(
                "vector_interp_energies",
                "vector[2]")
            vector_coz_grid_points = ForwardVariableDef(
                "vector_coz_grid_points",
                "vector[2]")
            vector_log_trunc_e[1] << interpolated_energy_low
            vector_log_trunc_e[2] << interpolated_energy_high
            vector_coz_grid_points[1] << cos_theta_grid_stan[cos_theta_bin_index]
            vector_coz_grid_points[2] << cos_theta_grid_stan[cos_theta_bin_index + 1]

            interpolate_cosz = FunctionCall([vector_coz_grid_points, vector_log_trunc_e, cos_dir], "interpolate", 3)
            _ = ReturnStatement([interpolate_cosz])


class AtmosphericNuMuFlux(FluxModel):

    CACHE_FNAME = "mceq_flux.pickle"
    EMAX = 1E9 * u.GeV
    EMIN = 1 * u.GeV
    THETA_BINS = 100

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._setup()
        self._parameters = {}

    def _setup(self):
        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                self._flux_spline = pickle.load(fr)
        else:
            mceq = MCEqRun(
                # High-energy hadronic interaction model
                interaction_model='SIBYLL23C',

                # cosmic ray flux at the top of the atmosphere
                primary_model=(crf.HillasGaisser2012, 'H3a'),

                # zenith angle
                theta_deg=0.,
            )

            theta_grid = np.degrees(np.arccos(np.linspace(0, 1, self.THETA_BINS)))
            numu_fluxes = []
            for theta in theta_grid:
                mceq.set_theta_deg(theta)
                mceq.solve()
                numu_fluxes.append(
                    (mceq.get_solution('numu')
                     + mceq.get_solution('antinumu')))

            emask = (mceq.e_grid < self.EMAX) & (mceq.e_grid > self.EMIN)
            splined_flux = scipy.interpolate.RectBivariateSpline(
                np.cos(np.radians(theta_grid)),
                np.log10(mceq.e_grid[emask]),
                np.log10(numu_fluxes)[:, emask],
            )
            self._flux_spline = splined_flux
            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                pickle.dump(splined_flux, fr)

    def make_stan_function(self, energy_points=100, theta_points=50):
        log_energy_grid = np.linspace(np.log10(self.EMIN), np.log10(self.EMAX), energy_points)
        return _AtmosphericNuMuFluxStan(
            self._flux_spline,
            log_energy_grid,
            theta_points)

    @u.quantity_input
    def __call__(self, energy: u.GeV, dec: u.rad, ra: u.rad) -> 1 / (u.GeV * u.s * u.m**2 * u.sr):
        energy = np.atleast_1d(energy)
        if np.any((energy > self.EMAX) | (energy < self.EMIN)):
            raise ValueError("Energy needs to be in {} < E {}".format(self.EMIN, self.EMAX))

        cosz = np.atleast_1d(-np.sin(dec))

        try:
            result = np.power(10, self._flux_spline(
                cosz, np.log10(energy / u.GeV)))
        except ValueError as e:
            print("Error in spline evaluation. Are the evaluation points ordered?")
            raise e

        return np.squeeze(result) << (1 / (u.GeV * u.s * u.cm**2 * u.sr))

    def call_fast(self, energy, dec, ra):
        energy = np.atleast_1d(energy)
        if np.any((energy > self.EMAX.value) | (energy < self.EMIN.value)):
            raise ValueError("Energy needs to be in {} < E {}".format(self.EMIN, self.EMAX))

        cosz = np.atleast_1d(-np.sin(dec))

        try:
            result = np.power(10, self._flux_spline(
                cosz, np.log10(energy)))
        except ValueError as e:
            print("Error in spline evaluation. Are the evaluation points ordered?")
            raise e

        return np.squeeze(result)

    @u.quantity_input
    def total_flux(self, energy: u.GeV) -> 1 / (u.m**2 * u.s * u.GeV):

        energy = energy.value

        def wrap_call(dec):
            return self.call_fast(energy, dec, 0)

        integral = quad(wrap_call, -np.pi, np.pi)[0] / (u.cm**2 * u.s * u.rad * u.GeV)
        ra_int = 2 * np.pi * u.rad

        return integral * ra_int

    @property
    @u.quantity_input
    def total_flux_int(self) -> 1 / (u.m**2 * u.s):
        return self.integral(*self.energy_bounds, -np.pi * u.rad, np.pi * u.rad, 0 * u.rad, 2 * np.pi * u.rad)

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        raise NotImplementedError()

    def redshift_factor(self, z: float):
        return 1

    @property
    def energy_bounds(self):
        return (self.EMIN, self.EMAX)

    @u.quantity_input
    def integral(
            self,
            e_low: u.GeV,
            e_up: u.GeV,
            dec_low: u.rad,
            dec_up: u.rad,
            ra_low: u.rad,
            ra_up: u.rad) -> 1 / (u.m**2 * u.s):

        def wrap_call(log10_energy, dec):
            return self.call_fast(10**log10_energy, dec, 0) * 10**log10_energy

        e_low = (e_low / u.GeV).value
        e_up = (e_up / u.GeV).value
        dec_low = (dec_low / u.rad).value
        dec_up = (dec_up / u.rad).value

        ra_int = ra_up - ra_low
        ra_low = (ra_low / u.rad).value
        ra_up = (ra_up / u.rad).value

        # TODO: Make sure precision is good
        integral = dblquad(wrap_call, dec_low, dec_up, lambda _: np.log10(e_low), lambda _: np.log10(e_up))
        integral = integral[0] / (u.cm**2 * u.s * u.rad)

        return integral * ra_int
