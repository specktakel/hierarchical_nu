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
from hierarchical_nu.source.atmospheric_flux import AtmosphericNuMuFlux
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.atmospheric_flux import months
from hierarchical_nu.events import Events
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.utils.cache import Cache
from hierarchical_nu.detector.r2021 import R2021EffectiveArea
from hierarchical_nu.detector.detector_model import EffectiveArea
from hierarchical_nu.utils.lifetime import LifeTime


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
        flux: AtmosphericNuMuFlux,
        aeff: EffectiveArea,
        dec_idx: int,
    ):
        """
        :param season: EventType instance for which to calculate the rate. Must be of IC40 to IC86_II
        :param flux: AtmosphericNuMuFlux to use for the rate calculation
        :param aeff: EffectiveArea instance matching the season, needs to have a spline implementation
        :dec_idx: Declination index of the IRF to use. Is fixed for the instance, not to be changed at run time.
        """
        irf = R2021IRF.from_period(season.P)

        if not isinstance(flux, AtmosphericNuMuFlux):
            raise ValueError("'atmo' needs to be instance of `AtmosphericNuMuFlux`")
        self._atmo = flux
        if not isinstance(aeff, EffectiveArea) or not hasattr(aeff, "_eff_area_spline"):
            raise ValueError("'aeff' is not a proper effective area")
        self._aeff = aeff

        hists = []
        bin_edges = []

        self.tE_size = irf.true_energy_values.size
        self.tE_bin_edges = irf.true_energy_bins
        self.aeff_tE_bin_edges = aeff.tE_bin_edges
        self.dec_size = irf.declination_bins.size - 1
        self.dec_bin_edges = irf.declination_bins << u.rad

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
        self._season = season
        self.lifetime = LifeTime().lifetime_from_dm(season)[season].to_value(u.s)

        # Create ROI in appropriate dec range covered by the IRF dec bin selected
        self._dec_idx = dec_idx
        dec_min, dec_max = self.irf.declination_bins[dec_idx : dec_idx + 2]
        dec_min = np.deg2rad(-5) if dec_min < np.deg2rad(-5) else dec_min
        ROIList.clear_registry()
        roi = RectangularROI(DEC_min=dec_min * u.rad, DEC_max=dec_max * u.rad)
        Parameter.clear_registry()
        Parameter(1e1 * u.GeV, "Emin_det", fixed=True)
        self.Ebins = np.geomspace(1e2, 1e5, 31)
        logEbins = np.log10(self.Ebins)
        Ebins_c = np.power(10, logEbins[:-1] + np.diff(logEbins) / 2)
        events = Events.from_ev_file(season)
        exp_N = np.histogram(events.energies.to_value(u.GeV), self.Ebins)[0]
        self.exp_rate = exp_N / self.lifetime
        self.exp_N = exp_N
        # Clean up
        ROIList.clear_registry()
        Parameter.clear_registry()

    @property
    def season(self):
        return self._season

    @property
    def dec_idx(self):
        return self._dec_idx

    def make_shifted_ereco_pdfs(self, a1, a2, b1, b2):

        ereco_pdfs = []
        shifts = [lambda x: x * a1 + b1, lambda x: x * a2 + b2] + [lambda x: x] * 20
        for c in range(self.tE_size):
            ereco_pdfs.append(
                self.create_shifted_reco_pdf(c, self.dec_idx, shift=shifts[c])
            )

        return ereco_pdfs

    @u.quantity_input
    def create_shifted_reco_pdf(
        self,
        tE_idx: int,
        dec_idx: int,
        hist: bool = False,
        shift: Callable = lambda x: x,
    ):
        """This functions creates a spline for the reco energy
        distribution given a true neutrino engery.
        Copied from skyllh and slightly adapted to be used
        with shifted reco energy bins.
        """

        # manually correct for some zero-bin which does weird stuff to the spline

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
        detailed: bool = False,
    ):

        sindec_min = np.sin(self.dec_bin_edges[self.dec_idx].to_value(u.rad))
        sindec_max = np.sin(self.dec_bin_edges[self._dec_idx + 1].to_value(u.rad))
        ereco_pdfs = self.make_shifted_ereco_pdfs(a1, a2, b1, b2)

        def run(El, Eh):
            cdfs = []
            # Precompute the pdf integrals, i.e. cdf, between the Ereco edges
            # for all tE bins. When asked for the integrand, only look up value
            for c, logE in enumerate(np.arange(2.25, 8.76, 0.5)):
                pdf = ereco_pdfs[c]

                def _cdf(x):
                    return pdf(x) / pdf.norm

                cdf = quad(_cdf, np.log10(El), np.log10(Eh))[0]
                cdfs.append(cdf)

            def integrand(logE):
                etrue_idx = np.digitize(logE, self.tE_bin_edges) - 1
                """
                pdf = ereco_pdfs[etrue_idx]

                def _cdf(x):
                    return pdf(x) / pdf.norm

                cdf = quad(_cdf, np.log10(El), np.log10(Eh))[0]
                """

                def sin_dec_int(sindec, logE):
                    return self._aeff._eff_area_spline((logE, -sindec)) * np.power(
                        10, self._atmo._flux_spline(-sindec, logE).squeeze()
                    )

                sin_dec_integrated = quad(
                    sin_dec_int, sindec_min, sindec_max, args=(logE)
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
                    for Elow, Ehigh in zip(
                        np.log10(self.Ebins[:-1]), np.log10(self.Ebins[1:])
                    )
                ]
            else:
                # integral is over true energy
                integrals = quad(integrand, 2, 8)[0] * 2 * np.pi
            return integrals

        # loop over reconstructed energies
        rate = Parallel(n_jobs=self.Ebins.size - 1, backend="loky", prefer="threads")(
            delayed(run)(El, Eh) for (El, Eh) in zip(self.Ebins[:-1], self.Ebins[1:])
        )
        # 1e4 to convert atmo flux from /cm**2 to /m**2
        return np.array(rate) * 1e4

    def likelihood(self, a1, a2, b1, b2):
        rates = self.calc_rates(a1, a2, b1, b2)
        mask = rates > 0.0
        Nex = rates * self.lifetime
        llh = -2 * np.sum(Nex[mask] + self.exp_N[mask] * np.log(Nex[mask]))
        return llh

    def scan():
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

    calc = RateCalculator(season, flux_spline, aeff_spline)
    print(calc.season)
    sys.exit()
