import pickle
from functools import lru_cache
import astropy.units as u
import numpy as np
import scipy
from scipy.integrate import dblquad, quad

from MCEq.core import MCEqRun
import crflux.models as crf


from .flux_model import FluxModel
from ..utils.cache import Cache
from ..backend import (
    UserDefinedFunction,
    StanArray,
    TruncatedParameterization,
    LogParameterization,
    StringExpression,
    FunctionCall,
    ReturnStatement,
    ForwardVariableDef,
)

Cache.set_cache_dir(".cache")


class _AtmosphericNuMuFluxStan(UserDefinedFunction):
    def __init__(
        self,
        splined_flux,
        log_energy_grid,
        theta_points: int = 50,
    ):

        UserDefinedFunction.__init__(
            self,
            "AtmosphericNumuFlux",
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real",
        )

        self.theta_points = theta_points

        cos_theta_grid = np.linspace(-1, 1, self.theta_points)

        spl_evals = np.empty((self.theta_points, len(log_energy_grid)))

        for i, cos_theta in enumerate(cos_theta_grid):

            spl_evals[i] = splined_flux(cos_theta, log_energy_grid).squeeze()

        with self:

            cos_theta = ForwardVariableDef("cos_theta", "real")
            cos_theta_bin_index = ForwardVariableDef("cos_theta_bin_index", "int")
            vals_cos_theta_low = ForwardVariableDef(
                "vals_cos_theta_low", "vector[%i]" % len(log_energy_grid)
            )
            vals_cos_theta_high = ForwardVariableDef(
                "vals_cos_theta_high", "vector[%i]" % len(log_energy_grid)
            )

            spl_evals_stan = StanArray("AtmosphericNuMuFluxGrid", "real", spl_evals)

            cos_theta_grid_stan = StanArray("cos_theta_grid", "real", cos_theta_grid)
            log_energy_grid_stan = StanArray("log_energy_grid", "real", log_energy_grid)

            truncated_e = TruncatedParameterization(
                "true_energy", 10 ** log_energy_grid[0], 10 ** log_energy_grid[-1]
            )
            log_trunc_e = LogParameterization(truncated_e)

            # Use abs() since the flux is symmetric around the horizon
            cos_theta << StringExpression(["abs(cos(pi() - acos(true_dir[3])))"])
            cos_theta_bin_index << FunctionCall(
                [cos_theta, cos_theta_grid_stan],
                "binary_search",
            )

            # StringExpression(["print(\"cos_dir = \",",cos_dir,")"])
            # StringExpression(["print(\"cos_theta_bin_index = \",",cos_theta_bin_index,")"])

            vals_cos_theta_low << FunctionCall(
                [spl_evals_stan[cos_theta_bin_index]], "to_vector"
            )
            vals_cos_theta_high << FunctionCall(
                [spl_evals_stan[cos_theta_bin_index + 1]], "to_vector"
            )

            # vect_spl_vals_low = FunctionCall(
            #    [spl_evals_stan[cos_theta_bin_index]], "to_vector"
            # )
            # vect_spl_vals_high = FunctionCall(
            #    [spl_evals_stan[cos_theta_bin_index + 1]], "to_vector"
            # )
            vect_log_e_grid = FunctionCall([log_energy_grid_stan], "to_vector")

            interpolated_energy_low = FunctionCall(
                [vect_log_e_grid, vals_cos_theta_low, log_trunc_e], "interpolate"
            )
            interpolated_energy_high = FunctionCall(
                [vect_log_e_grid, vals_cos_theta_high, log_trunc_e], "interpolate"
            )

            vector_log_trunc_e = ForwardVariableDef(
                "vector_interp_energies", "vector[2]"
            )
            vector_cosz_grid_points = ForwardVariableDef(
                "vector_cosz_grid_points", "vector[2]"
            )
            vector_log_trunc_e[1] << interpolated_energy_low
            vector_log_trunc_e[2] << interpolated_energy_high
            vector_cosz_grid_points[1] << cos_theta_grid_stan[cos_theta_bin_index]
            vector_cosz_grid_points[2] << cos_theta_grid_stan[cos_theta_bin_index + 1]

            interpolate_cosz = FunctionCall(
                [vector_cosz_grid_points, vector_log_trunc_e, cos_theta], "interpolate"
            )

            # Units of GeV^-1 m^-2 s^-1
            _ = ReturnStatement([(10**interpolate_cosz) * 1e4])


class AtmosphericNuMuFlux(FluxModel):

    CACHE_FNAME = "mceq_flux.pickle"
    EMAX = 1e9 * u.GeV
    EMIN = 1 * u.GeV
    THETA_BINS = 100

    @u.quantity_input
    def __init__(self, lower_energy: u.GeV, upper_energy: u.GeV, *args, **kwargs):
        super().__init__()
        self._setup()
        self._parameters = {}
        if (lower_energy < self.EMIN) or (upper_energy > self.EMAX):
            raise ValueError("Invalid energy bounds")
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy

    def _setup(self):
        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                self._flux_spline = pickle.load(fr)
        else:
            mceq = MCEqRun(
                # High-energy hadronic interaction model
                interaction_model="SIBYLL23C",
                # cosmic ray flux at the top of the atmosphere
                primary_model=(crf.HillasGaisser2012, "H3a"),
                # zenith angle
                theta_deg=0.0,
            )

            theta_grid = np.degrees(np.arccos(np.linspace(0, 1, self.THETA_BINS)))
            numu_fluxes = []
            for theta in theta_grid:
                mceq.set_theta_deg(theta)
                mceq.solve()
                # TODO: Extend for cascade model?
                numu_fluxes.append(
                    (mceq.get_solution("numu") + mceq.get_solution("antinumu"))
                )

            theta_grid_2 = np.degrees(np.arccos(np.linspace(-1, 0, self.THETA_BINS)))[
                :-1
            ]
            numu_fluxes = numu_fluxes[::-1][:-1] + numu_fluxes

            emask = (mceq.e_grid < self.EMAX / u.GeV) & (
                mceq.e_grid > self.EMIN / u.GeV
            )
            splined_flux = scipy.interpolate.RectBivariateSpline(
                np.cos(np.radians(np.concatenate((theta_grid_2, theta_grid)))),
                np.log10(mceq.e_grid[emask]),
                np.log10(numu_fluxes)[:, emask],
            )
            self._flux_spline = splined_flux
            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                pickle.dump(splined_flux, fr)

    def make_stan_function(self, energy_points=100, theta_points=50):
        log_energy_grid = np.linspace(
            np.log10(self.EMIN / u.GeV).value,
            np.log10(self.EMAX / u.GeV).value,
            energy_points,
        )

        stan_func = _AtmosphericNuMuFluxStan(
            self._flux_spline, log_energy_grid, theta_points
        )
        return stan_func

    @u.quantity_input
    def __call__(
        self, energy: u.GeV, dec: u.rad, ra: u.rad
    ) -> 1 / (u.GeV * u.s * u.cm**2 * u.sr):
        energy = np.atleast_1d(energy)
        if np.any((energy > self.EMAX) | (energy < self.EMIN)):
            raise ValueError(
                "Energy needs to be in {} < E {}".format(self.EMIN, self.EMAX)
            )

        cosz = np.atleast_1d(-np.sin(dec))

        try:
            result = np.power(10, self._flux_spline(cosz, np.log10(energy / u.GeV)))
        except ValueError as e:
            print("Error in spline evaluation. Are the evaluation points ordered?")
            raise e

        return np.squeeze(result) << (1 / (u.GeV * u.s * u.cm**2 * u.sr))

    @lru_cache(maxsize=None)
    def call_fast(self, energy, dec, ra):
        energy = np.atleast_1d(energy)
        if np.any((energy > self.EMAX.value) | (energy < self.EMIN.value)):
            raise ValueError(
                "Energy needs to be in {} < E {}".format(self.EMIN, self.EMAX)
            )

        cosz = np.atleast_1d(-np.sin(dec))

        try:
            result = np.power(10, self._flux_spline(cosz, np.log10(energy)))
        except ValueError as e:
            print("Error in spline evaluation. Are the evaluation points ordered?")
            raise e

        return np.squeeze(result)

    @u.quantity_input
    def total_flux(self, energy: u.GeV) -> 1 / (u.m**2 * u.s * u.GeV):

        energy = (energy / u.GeV).value

        def _integral(energy):
            def wrap_call(sindec):
                return self.call_fast(energy, np.arcsin(sindec), 0)

            integral = quad(wrap_call, -1, 1)[0]
            ra_int = 2 * np.pi

            return integral * ra_int

        vect_int = np.vectorize(_integral)

        return vect_int(energy) << (1 / (u.cm**2 * u.s * u.GeV))

    @property
    @u.quantity_input
    def total_flux_int(self) -> 1 / (u.m**2 * u.s):
        return self.integral(
            *self.energy_bounds,
            (-np.pi / 2) * u.rad,
            (np.pi / 2) * u.rad,
            0 * u.rad,
            2 * np.pi * u.rad
        )

    @property
    @u.quantity_input
    def total_flux_density(self) -> u.erg / u.s / u.m**2:
        raise NotImplementedError()

    def redshift_factor(self, z: float):
        return 1

    @property
    def energy_bounds(self):
        return (self._lower_energy, self._upper_energy)

    @u.quantity_input
    def integral(
        self,
        e_low: u.GeV,
        e_up: u.GeV,
        dec_low: u.rad,
        dec_up: u.rad,
        ra_low: u.rad,
        ra_up: u.rad,
    ) -> 1 / (u.m**2 * u.s):
        def _integral(e_low, e_up, dec_low, dec_up, ra_low, ra_up):
            def wrap_call(log_energy, sindec):
                return self.call_fast(
                    np.exp(log_energy), np.arcsin(sindec), 0
                ) * np.exp(log_energy)

            ra_int = ra_up - ra_low

            integral = dblquad(
                wrap_call,
                np.sin(dec_low),
                np.sin(dec_up),
                lambda _: np.log(e_low),
                lambda _: np.log(e_up),
            )
            integral = integral[0] * ra_int

            return integral

        vect_int = np.vectorize(_integral)

        e_low = (e_low / u.GeV).value
        e_up = (e_up / u.GeV).value
        dec_low = (dec_low / u.rad).value
        dec_up = (dec_up / u.rad).value
        ra_low = (ra_low / u.rad).value
        ra_up = (ra_up / u.rad).value

        return vect_int(e_low, e_up, dec_low, dec_up, ra_low, ra_up) << (
            1 / (u.cm**2 * u.s)
        )

    def make_stan_sampling_func(cls, f_name):

        raise NotImplementedError()
