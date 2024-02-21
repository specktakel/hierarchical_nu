from typing import Callable
import sys
from argparse import ArgumentParser
import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
from astropy import units as u
import pickle
import os

from hierarchical_nu.events import Events
from hierarchical_nu.detector.icecube import (
    IC40,
    IC59,
    IC79,
    IC86_I,
    IC86_II,
    EventType,
)
from hierarchical_nu.utils.roi import RectangularROI, ROIList
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.atmospheric_flux import months
from hierarchical_nu.events import Events
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.utils.cache import Cache
from hierarchical_nu.detector.r2021 import R2021EffectiveArea


from skyllh.analyses.i3.publicdata_ps.utils import FctSpline1D, FctSpline2D
from icecube_tools.detector.r2021 import R2021IRF

from icecube_tools.utils.data import data_directory

from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed


class RateCalculator:
    @u.quantity_input
    def __init__(
        self,
        season: EventType,
        flux_spline: Callable,
        aeff_spline: Callable,
        dec_idx: int,
    ):
        irf = R2021IRF.from_period(season.P)

        hists = []
        bin_edges = []

        for c_d in range(irf.declination_bins.size - 1):
            hists.append([])
            bin_edges.append([])
            for c_e in range(irf.true_energy_values.size):
                bin_e = irf.reco_energy_bins[c_e, c_d]
                bin_c = bin_e[:-1] + np.diff(bin_e) / 2
                bin_edges[c_d].append(bin_e)
                if (c_e, c_d) not in irf.faulty:
                    pdf = irf.reco_energy[c_e, c_d].pdf(bin_c)
                    if season.P == "IC86_II":
                        # Manually correct for zero bins in the middle
                        # of the energy space
                        if c_e == 3 and c_d == 1:
                            pdf[-3] = pdf[-2]
                            dlogE = np.diff(bin_edges[c_d][c_e])
                            # re-normalise to one
                            norm = np.sum(pdf * dlogE)
                            pdf /= norm
                    hists[c_d].append(pdf)
                else:
                    hists[c_d].append(np.zeros_like(bin_c))
        self.irf = irf
        self.hists = np.array(hists)
        self.bin_edges = np.array(bin_edges)
        self.season = season

        dec_min, dec_max = self.irf.declination_bins[dec_idx : dec_idx + 2]
        dec_min = np.deg2rad(-5) if dec_min < np.deg2rad(-5) else dec_min
        roi = RectangularROI(DEC_min=dec_min * u.rad, DEC_max=dec_max * u.rad)

        Ebins = np.geomspace(1e2, 1e5, 31)
        logEbins = np.log10(Ebins)
        Ebins_c = np.power(10, logEbins[:-1] + np.diff(logEbins) / 2)
        events = Events.from_ev_file(season)
        exp_N = np.histogram(events.energies.to_value(u.GeV), Ebins)[0]
        self.exp_N = exp_N

    def make_shifted_ereco_pdfs(self, a1, a2, b1, b2, dec):
        trueE = sm._true_e_bin_edges
        trueE_c = trueE[:-1] + np.diff(trueE) / 2
        ereco_pdfs = []
        shifts = [lambda x: x * a1 + b1, lambda x: x * a2 + b2] + [lambda x: x] * 20
        for shift, tE in zip(shifts, trueE_c):
            ereco_pdfs.append(
                self.create_shifted_reco_pdf(np.power(10, tE) * u.GeV, dec, shift=shift)
            )

        return ereco_pdfs

    @u.quantity_input
    def create_shifted_reco_pdf(
        self,
        Etrue: u.GeV,
        dec: u.rad,
        hist: bool = False,
        shift: Callable = lambda x: x,
    ):
        """This functions creates a spline for the reco energy
        distribution given a true neutrino engery.
        Copied from skyllh and slightly adapted to be used
        with shifted reco energy bins.
        """

        # manually correct for some zero-bin which does weird stuff to the spline

        tE_idx = (
            np.digitize(np.log10(Etrue.to_value(u.GeV)), self.irf.true_energy_bins) - 1
        )
        dec_idx = np.digitize(dec.to_value(u.rad), self.irf.declination_bins) - 1
        log10_reco_e_binedges = shift(self.bin_edges[dec_idx, tE_idx])
        pdf = self.hists[dec_idx, tE_idx]

        spline = FctSpline1D(pdf, log10_reco_e_binedges, norm=True)

        if hist:
            return pdf, log10_reco_e_binedges
        else:
            return spline

    @u.quantity_input
    def calc_rates(
        self,
        a1: float = 1.0,
        a2: float = 1.0,
        b1: float = 0.0,
        b2: float = 0.0,
        dec_min: u.rad = np.deg2rad(-5) * u.rad,
        dec_max: u.rad = np.deg2rad(10) * u.rad,
        detailed: bool = False,
    ):

        dec = (dec_max + dec_min) / 2
        ereco_pdfs = self.make_shifted_ereco_pdfs(a1, a2, b1, b2, dec)

        def run(El, Eh):
            cdfs = []
            for c, logE in enumerate(np.arange(2.25, 8.76, 0.5)):
                etrue_idx = sm.get_log10_true_e_idx(logE)
                assert c == etrue_idx
                pdf = ereco_pdfs[etrue_idx]

                def _cdf(x):
                    return pdf(x) / pdf.norm

                cdf = quad(_cdf, np.log10(El), np.log10(Eh))[0]
                cdfs.append(cdf)

            def integrand(logE):
                etrue_idx = sm.get_log10_true_e_idx(logE)

                def sin_dec_int(sindec, logE):
                    return aeff2dspline(sindec, logE)[0] * f_atmo_2dspline(
                        np.arcsin(sindec), logE
                    )

                sin_dec_integrated = quad(
                    sin_dec_int, np.sin(dec_min), np.sin(dec_max), args=(logE)
                )[0]

                return (
                    cdfs[etrue_idx]
                    * sin_dec_integrated
                    * np.power(10, logE)
                    * np.log(10)
                )

            if detailed:
                integrals = [
                    quad(integrand, Elow, Ehigh)[0] * 2 * np.pi
                    for Elow, Ehigh in zip(Etrue_bins[:-1], Etrue_bins[1:])
                ]
            else:
                # integral is over true energy
                integrals = quad(integrand, 2, 8)[0] * 2 * np.pi
            return integrals

        # loop over reconstructed energies
        rate = Parallel(n_jobs=20, backend="loky", prefer="threads")(
            delayed(run)(El, Eh) for (El, Eh) in zip(Ebins[:-1], Ebins[1:])
        )

        return np.array(rate)

    def likelihood(self, a1, a2, b1, b2):
        rates = self.calc_rates(a1, a2, b1, b2)
        mask = rates > 0.0
        Nex = rates * self.lifetime
        llh = -2 * np.sum(Nex[mask] + self.exp_N[mask] * np.log(Nex[mask]))
        return llh

    def scan():
        pass


def load_flux():
    Cache.set_cache_dir("input/mceq")
    for c, month in enumerate(months):
        with Cache.open(f"mceq_flux_{month}.pickle", "rb") as fr:
            if c == 0:
                (e_grid, theta_grid), flux = pickle.load(fr)
            else:
                (_e_grid, _theta_grid), _flux = pickle.load(fr)

                if not np.all(np.isclose(e_grid, _e_grid)):
                    raise ValueError("Energy grids are incompatible.")
                if not np.all(np.isclose(theta_grid, _theta_grid)):
                    raise ValueError("Theta grids are incompatible.")
                flux += _flux
    f_atmo = flux / 12

    return e_grid, theta_grid, f_atmo


# define utility function used for the atmo spectrum
def broken_pl(E, g1, g2, E0):
    output = np.ones_like(E)
    output[E <= E0] = np.power(E[E <= E0] / E0, g1)
    output[E > E0] = np.power(E[E > E0] / E0, g2)
    return output


def pl(E, g, E0):
    return broken_pl(E, g, g, E0)


def calc_rates():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--season", type=str)
    parser.add_argument("-d", "--declination", type=int)
    parser.add_argument("-i", "--index", type=float, default=0.0)
    args = parser.parse_args()
    season = args.season
    dec_bin = parser.declination
    index = parser.index

    Parameter.clear_registry()
    Parameter(1e1 * u.GeV, "Emin_det", fixed=True)
    # Bins for the rate calculation

    log_E_edges, dec_nu_edges, f_atmo = load_flux()

    E_factor = pl(np.power(10, log_E_edges), index, 1e3)
    flux_spline = FctSpline2D(
        f_atmo * E_factor[np.newaxis, :], dec_nu_edges, log_E_edges
    )

    aeff = R2021EffectiveArea(season=season)
    aeff_spline = FctSpline2D(
        aeff.cosz_bin_edges, np.log10(aeff.tE_bin_edges), aeff.eff_area
    )

    calc = RateCalculator(season, flux_spline, aeff_spline)
    print(calc.season)
    sys.exit()
