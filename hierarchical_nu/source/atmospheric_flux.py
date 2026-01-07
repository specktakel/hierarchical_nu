import pickle
from functools import lru_cache
import astropy.units as u
import numpy as np
import scipy
from scipy.integrate import dblquad, quad
import logging

from MCEq.core import MCEqRun
import crflux.models as crf
import mceq_config

from ..utils.roi import ROIList


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

from joblib import Parallel, delayed

mceq_config.debug_level = 0


months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class _AtmosphericNuMuFluxStan(UserDefinedFunction):
    """
    Stan interface of atmospheric muon neutrino spectrum
    """

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

        cos_theta_grid = np.linspace(-1, 1, theta_points)

        if ROIList.STACK:
            apply_roi = np.all([_.apply_roi for _ in ROIList.STACK])
        else:
            apply_roi = False

        if apply_roi:
            cosz_min = -np.sin(ROIList.DEC_max())
            cosz_max = -np.sin(ROIList.DEC_min())
            idx_min = np.digitize(cosz_min, cos_theta_grid) - 1
            idx_max = np.digitize(cosz_max, cos_theta_grid, right=True)
            self.cos_theta_grid = cos_theta_grid[idx_min : idx_max + 1]
            self.theta_points = self.cos_theta_grid.size
        else:
            self.theta_points = theta_points
            self.cos_theta_grid = cos_theta_grid

        self.log_energy_grid = log_energy_grid

        self._spl_evals = np.empty((self.theta_points, len(log_energy_grid)))

        for i, cos_theta in enumerate(self.cos_theta_grid):
            self._spl_evals[i] = splined_flux(cos_theta, log_energy_grid).squeeze()

        with self:
            cos_theta = ForwardVariableDef("cos_theta", "real")
            cos_theta_bin_index = ForwardVariableDef("cos_theta_bin_index", "int")
            vals_cos_theta_low = ForwardVariableDef(
                "vals_cos_theta_low", "vector[%i]" % len(log_energy_grid)
            )
            vals_cos_theta_high = ForwardVariableDef(
                "vals_cos_theta_high", "vector[%i]" % len(log_energy_grid)
            )

            self._spl_evals_stan = StanArray(
                "AtmosphericNuMuFluxGrid", "real", self._spl_evals
            )

            cos_theta_grid_stan = StanArray(
                "cos_theta_grid", "real", self.cos_theta_grid
            )
            log_energy_grid_stan = StanArray(
                "log_energy_grid", "real", self.log_energy_grid
            )

            truncated_e = TruncatedParameterization(
                "true_energy", 10 ** log_energy_grid[0], 10 ** log_energy_grid[-1]
            )
            log_trunc_e = LogParameterization(truncated_e)

            cos_theta << StringExpression(["cos(pi() - acos(true_dir[3]))"])
            cos_theta_bin_index << FunctionCall(
                [cos_theta, cos_theta_grid_stan],
                "binary_search",
            )

            # StringExpression(["print(\"cos_dir = \",",cos_dir,")"])
            # StringExpression(["print(\"cos_theta_bin_index = \",",cos_theta_bin_index,")"])

            vals_cos_theta_low << FunctionCall(
                [self._spl_evals_stan[cos_theta_bin_index]], "to_vector"
            )
            vals_cos_theta_high << FunctionCall(
                [self._spl_evals_stan[cos_theta_bin_index + 1]], "to_vector"
            )

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
    """
    Python interface of atmospheric muon neutrino spectrum
    """

    CACHE_FNAME = "mceq_flux.pickle"
    EMAX = 1e9 * u.GeV
    EMIN = 1 * u.GeV
    THETA_BINS = 100

    @u.quantity_input
    def __init__(self, lower_energy: u.GeV, upper_energy: u.GeV, **kwargs):
        """
        :param lower_energy: lower energy of flux model
        :param upper_energy: upper energy of flux model
        :param kwargs: Additional kwargs to modify cache dir
        """

        self.cache_dir = kwargs.pop("cache_dir", ".cache")
        super().__init__()
        # We can artificially add a powerlaw slope to the spectrum by multiplication
        # Only for debugging event rates used
        self._add_index = kwargs.pop("index", 0.0)
        self._setup()
        self._parameters = {}
        if (lower_energy < self.EMIN) or (upper_energy > self.EMAX):
            raise ValueError("Invalid energy bounds")
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy

    def _create_spline(self, flux, e_grid, theta_grid):

        def pl(E, index, E0):
            return np.power(E / E0, index)

        # Signature flux 2d array, egrid and theta/zenith
        flux = flux * pl(e_grid, self._add_index, 1e3)[np.newaxis, :]
        splined_flux = scipy.interpolate.RectBivariateSpline(
            np.cos(np.radians(theta_grid)),
            np.log10(e_grid),
            np.log10(flux),
        )
        return splined_flux

    def _load_from_file(self):
        with Cache.open(self.CACHE_FNAME, "rb") as fr:
            data = pickle.load(fr)
            if isinstance(data, scipy.interpolate.RectBivariateSpline):
                # For backwards compatibility
                self._flux_spline = data
            else:
                # is a tuple of flux, e_grid, theta_grid
                (e_grid, theta_grid), flux = data
                self._flux_spline = self._create_spline(flux, e_grid, theta_grid)

    def _load_from_monthly_files(self):
        for c, month in enumerate(months):
            if c == 0:
                with Cache.open(f"mceq_flux_{month}.pickle", "rb") as fr:
                    (e_grid, theta_grid), flux = pickle.load(fr)
            else:
                with Cache.open(f"mceq_flux_{month}.pickle", "rb") as fr:
                    (_e_grid, _theta_grid), _flux = pickle.load(fr)
                    if not np.all(np.isclose(e_grid, _e_grid)):
                        raise ValueError("Energy grids are incompatible.")
                    if not np.all(np.isclose(theta_grid, _theta_grid)):
                        raise ValueError("Theta grids are incompatible.")
                    flux += _flux

        # Take average over a year
        f_atmo = flux / 12

        self._flux_spline = self._create_spline(f_atmo, e_grid, theta_grid)
        # Save averaged flux to a file in Cache
        with Cache.open(self.CACHE_FNAME, "wb") as fr:
            pickle.dump(((e_grid, theta_grid), f_atmo), fr)

    def _setup(self):
        Cache.set_cache_dir(self.cache_dir)
        if self.CACHE_FNAME in Cache:
            logger.debug("Loading flux from Cache.")
            self._load_from_file()

        elif all([f"mceq_flux_{month}.pickle" in Cache for month in months]):
            logger.debug("Loading monthly fluxes from Cache.")
            self._load_from_monthly_files()

        else:

            def run(month):
                # Calculate the spectrum for each month's atmosphere
                Cache.set_cache_dir(self.cache_dir)
                mceq = MCEqRun(
                    # High-energy hadronic interaction model
                    interaction_model="SIBYLL23C",
                    # cosmic ray flux at the top of the atmosphere
                    primary_model=(crf.HillasGaisser2012, "H4a"),
                    # zenith angle
                    theta_deg=0.0,
                    density_model=("MSIS00_IC", ("SouthPole", month)),
                )

                theta_grid = np.degrees(np.arccos(np.linspace(-1, 1, self.THETA_BINS)))
                numu_fluxes = []
                for theta in theta_grid:
                    mceq.set_theta_deg(theta)
                    mceq.solve()
                    numu_fluxes.append(
                        (
                            mceq.get_solution("total_numu")
                            + mceq.get_solution("total_antinumu")
                        )
                    )
                numu_fluxes = np.array(numu_fluxes)

                with Cache.open(f"mceq_flux_{month}.pickle", "wb") as f:
                    pickle.dump(((mceq.e_grid, theta_grid), numu_fluxes), f)

            run_for = []
            for month in months:
                if f"mceq_flux_{month}.pickle" not in Cache:
                    run_for.append(month)

            logger.debug(f"Running simulation for {run_for}")
            Parallel(n_jobs=len(run_for), backend="loky")(
                delayed(run)(month) for month in run_for
            )

            self._load_from_monthly_files()
        Cache.set_cache_dir(".cache")

    def make_stan_function(self, energy_points=100, theta_points=50):
        """
        :param energy_points: number of points for interpolation over energy
        :param theta_points: number of points for interpolation over declination,
            defined over the entire sky if ROIs are not applied
        """
        log_energy_grid = np.linspace(
            np.log10(self._lower_energy.to_value(u.GeV)),
            np.log10(self._upper_energy.to_value(u.GeV)),
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
        """
        Returns differential flux
        """

        energy = np.atleast_1d(energy)
        if np.any((energy > self.EMAX) | (energy < self.EMIN)):
            raise ValueError(
                "Energy needs to be in {} < E {}".format(self.EMIN, self.EMAX)
            )

        cosz = np.atleast_1d(-np.sin(dec))

        try:
            result = np.power(
                10,
                self._flux_spline(cosz, np.log10(energy.to_value(u.GeV)), grid=False),
            )
        except ValueError as e:
            print("Error in spline evaluation. Are the evaluation points ordered?")
            raise e

        return np.squeeze(result) << (1 / (u.GeV * u.s * u.cm**2 * u.sr))

    @lru_cache(maxsize=None)
    def call_fast(self, energy, dec, ra):
        energy = np.atleast_1d(energy)
        if np.any(
            (energy > self.EMAX.to_value(u.GeV)) | (energy < self.EMIN.to_value(u.GeV))
        ):
            raise ValueError(
                "Energy needs to be in {} < E {}".format(self.EMIN, self.EMAX)
            )

        cosz = np.atleast_1d(-np.sin(dec))

        try:
            result = np.power(10, self._flux_spline(cosz, np.log10(energy), grid=False))
        except ValueError as e:
            print("Error in spline evaluation. Are the evaluation points ordered?")
            raise e

        # Correct if outside bounds
        result.T[energy < self._lower_energy.to_value(u.GeV)] = 0.0
        result.T[energy > self._upper_energy.to_value(u.GeV)] = 0.0

        return np.squeeze(result)

    @u.quantity_input
    def total_flux(self, energy: u.GeV) -> 1 / (u.m**2 * u.s * u.GeV):
        """
        Returns differential flux integrated over the sky
        """

        energy = energy.to_value(u.GeV)

        def _integral(energy):
            def wrap_call(sindec):
                return self.call_fast(energy, np.arcsin(sindec), 0)

            integral = quad(wrap_call, -1, 1)[0]
            ra_int = 2 * np.pi

            return integral * ra_int

        vect_int = np.vectorize(_integral)

        return vect_int(energy) << (1 / (u.cm**2 * u.s * u.GeV))

    @u.quantity_input
    def flux_per_dec_band(
        self, energy: u.GeV, dec_min: u.rad, dec_max: u.rad
    ) -> 1 / (u.m**2 * u.s * u.GeV):
        """
        Returns differential flux integrated over specificed declination range
        """

        energy = energy.to_value(u.GeV)

        def _integral(energy):
            def wrap_call(sindec):
                return self.call_fast(energy, np.arcsin(sindec), 0)

            integral = quad(
                wrap_call,
                np.sin(dec_min.to_value(u.rad)),
                np.sin(dec_max.to_value(u.rad)),
            )[0]
            ra_int = 2 * np.pi

            return integral * ra_int

        vect_int = np.vectorize(_integral)

        return vect_int(energy) << (1 / (u.cm**2 * u.s * u.GeV))

    @property
    @u.quantity_input
    def total_flux_int(self) -> 1 / (u.m**2 * u.s):
        """
        Returns number flux integrated over energy and the entire sky
        """

        return self.integral(
            *self.energy_bounds,
            (-np.pi / 2) * u.rad,
            (np.pi / 2) * u.rad,
            0 * u.rad,
            2 * np.pi * u.rad,
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
        """
        Returns flux integrated over arbitrary energy and rectangular RA x DEC range
        """

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

        # Check edge cases
        e_low[((e_low < self._lower_energy) & (e_up > self._lower_energy))] = (
            self._lower_energy
        )
        e_up[((e_low < self._upper_energy) & (e_up > self._upper_energy))] = (
            self._upper_energy
        )

        e_low = e_low.to_value(u.GeV)
        e_up = e_up.to_value(u.GeV)
        dec_low = dec_low.to_value(u.rad)
        dec_up = dec_up.to_value(u.rad)
        ra_low = ra_low.to_value(u.rad)
        ra_up = ra_up.to_value(u.rad)

        return vect_int(e_low, e_up, dec_low, dec_up, ra_low, ra_up) << (
            1 / (u.cm**2 * u.s)
        )

    def make_stan_sampling_func(cls, f_name):
        raise NotImplementedError()
