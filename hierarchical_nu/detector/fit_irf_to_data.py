from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.integrate import quad
from astropy import units as u

from hierarchical_nu.events import Events
from hierarchical_nu.detector.icecube import (
    EventType,
)
from hierarchical_nu.utils.roi import RectangularROI, ROIList
from hierarchical_nu.source.atmospheric_flux import AtmosphericNuMuFlux
from hierarchical_nu.source.flux_model import IsotropicDiffuseBG
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.events import Events
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.detector.detector_model import EffectiveArea
from hierarchical_nu.detector.r2021 import R2021EffectiveArea, R2021EnergyResolution
from hierarchical_nu.utils.lifetime import LifeTime
from icecube_tools.detector.r2021 import R2021IRF

from hierarchical_nu.utils.fitting_tools import Spline1D

from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

"""
Defines class to compare experimental event rates with predictions.
Currently only implements diffuse source components, i.e. astro diffuse and atmospheric.
"""


class RateCalculator:
    @u.quantity_input
    def __init__(
        self,
        season: EventType,
        atmo_flux: AtmosphericNuMuFlux,
        diffuse_flux: IsotropicDiffuseBG,
        dec_idx: int,
    ):
        """
        :param season: EventType instance for which to calculate the rate. Must be of IC40 to IC86_II
        :param flux: AtmosphericNuMuFlux to use for the rate calculation
        :param aeff: EffectiveArea instance matching the season, needs to have a spline implementation
        :dec_idx: Declination index of the IRF to use. Is fixed for the instance, not to be changed at run time.
        """
        irf = R2021IRF.from_period(season.P)

        aeff = R2021EffectiveArea(season=season.P)
        eres = R2021EnergyResolution(season=season.P)

        if not isinstance(atmo_flux, AtmosphericNuMuFlux):
            raise ValueError(
                "'atmo_flux' needs to be instance of `AtmosphericNuMuFlux`"
            )
        self._atmo = atmo_flux
        if not isinstance(diffuse_flux, IsotropicDiffuseBG):
            raise ValueError(
                "'diffuse_flux' needs to be an instance of `IsotropicDiffuseBG`"
            )
        self._diff = diffuse_flux
        if not isinstance(aeff, EffectiveArea) or not hasattr(aeff, "_eff_area_spline"):
            raise ValueError("'aeff' is not a proper effective area")
        self._aeff = aeff
        self._eres = eres

        hists = []
        bin_edges = []

        self.tE_size = irf.true_energy_values.size
        self.tE_bin_edges = irf.true_energy_bins
        self.tE_binc = self.tE_bin_edges[:-1] + np.diff(self.tE_bin_edges) / 2
        self.aeff_tE_bin_edges = aeff.tE_bin_edges
        # self.dec_size = irf.declination_bins.size - 1
        self.dec_bin_edges = irf.declination_bins << u.rad

        for c_d in range(irf.declination_bins.size - 1):
            hists.append([])
            bin_edges.append([])
            for c_e in range(irf.true_energy_values.size):
                bin_e = irf.reco_energy_bins[c_e, c_d]
                if bin_e.size == 1:
                    bin_e = irf.reco_energy_bins[4, c_d]
                bin_c = bin_e[:-1] + np.diff(bin_e) / 2
                bin_edges[c_d].append(bin_e)
                if (c_e, c_d) not in irf.faulty:
                    pdf = irf.reco_energy[c_e, c_d].pdf(bin_c)
                    diff = np.diff(np.nonzero(pdf)[0])
                    if not np.all(np.isclose(diff, np.ones_like(diff))):
                        # Found some zeros inbetween non zero values,
                        # will mess up the interpolation of logs of evaluated splines
                        # fix these by taking the linear interpolation with the next
                        # non-zero value
                        beginning = True
                        for c in range(pdf.size):
                            # avoid changing the object that's iterated over by using range
                            val = pdf[c]
                            if val == 0.0 and beginning:
                                # If we are at the beginning, do noting
                                continue
                            if val != 0.0 and beginning:
                                # if we find the first non zero entry, set beginning to false
                                beginning = False
                            if val == 0.0 and not beginning:
                                # if value is zero and not beginning, check if we are at the end
                                if np.all(np.isclose(pdf[c:], np.zeros_like(pdf[c:]))):
                                    # nothing to see
                                    break
                                else:
                                    # now we found some weird stuff
                                    # find the next non-zero value and linearly interpolate between the encompassing non-zero values
                                    prev = pdf[c - 1]
                                    next_idx = c + np.min(np.nonzero(pdf[c:])[0])
                                    next_val = pdf[next_idx]
                                    # if the zero-entries are bunched up, fix all
                                    next_zero_idxs = (
                                        c + np.nonzero(pdf[c:next_idx] == 0)[0]
                                    )
                                    pdf[next_zero_idxs] = np.interp(
                                        bin_c[next_zero_idxs],
                                        [bin_c[c - 1], bin_c[next_idx]],
                                        [prev, next_val],
                                    )
                        # re-normalise the histogram
                        norm = np.sum(pdf * np.diff(bin_e))
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
        dec_min = self.irf.declination_bins[dec_idx] * u.rad
        dec_max = self.irf.declination_bins[dec_idx + 1] * u.rad
        dec_min = (
            np.deg2rad(-5) * u.rad
            if dec_min.to_value(u.rad) < np.deg2rad(-5)
            else dec_min
        )
        self._dec_min = dec_min
        self._dec_max = dec_max
        # needed w/o unit
        self._dec_middle = (dec_max + dec_min) / 2 * u.rad
        ROIList.clear_registry()
        roi = RectangularROI(DEC_min=dec_min, DEC_max=dec_max)
        Parameter.clear_registry()
        Parameter(1e1 * u.GeV, "Emin_det", fixed=True)
        # Reco energy bins in which rates are calculated
        self.Ebins = np.geomspace(1e2, 1e7, 51)
        # True energy bins of effective area
        self.all_Ebins = np.geomspace(1e2, 1e9, 71)
        logEbins = np.log10(self.Ebins)
        self.logE_binc = logEbins[:-1] + np.diff(logEbins) / 2
        self.Ebins_c = np.power(10, logEbins[:-1] + np.diff(logEbins) / 2)
        events = Events.from_ev_file(season)
        exp_N = np.histogram(events.energies.to_value(u.GeV), self.Ebins)[0]
        self.exp_rate = exp_N / self.lifetime
        self.exp_N = exp_N
        # Clean up
        ROIList.clear_registry()
        Parameter.clear_registry()

    @property
    def dec_min(self):
        return self._dec_min

    @property
    def dec_max(self):
        return self._dec_max

    @property
    def sindec_min(self):
        return np.sin(self.dec_min.to_value(u.rad))

    @property
    def sindec_max(self):
        return np.sin(self.dec_max.to_value(u.rad))

    @property
    def season(self):
        return self._season

    @property
    def dec_idx(self):
        return self._dec_idx

    def make_shifted_ereco_pdfs(self, a1, a2, b1, b2):
        """
        Create new energy resolution using linear transformations
        on the reconstructed energy bins of the two lowest neutrino energies,
        i.e. new bins = a * old bins + b
        """

        ereco_pdfs = []
        shifts = [lambda x: x * a1 + b1, lambda x: x * a2 + b2] + [lambda x: x] * 20
        for c in range(self.tE_size):
            ereco_pdfs.append(
                self.create_shifted_reco_pdf(c, self.dec_idx, shift=shifts[c])
            )

        return ereco_pdfs

    def create_shifted_reco_pdf(
        self,
        tE_idx: int,
        dec_idx: int,
        hist: bool = False,
        shift: Callable = lambda x: x,
    ):
        """
        This functions creates a histogram or a spline of the histogram for the reco energy
        distribution given a true neutrino engery.
        Copied from skyllh and slightly adapted to be used
        with shifted reco energy bins.
        """

        # manually correct for some zero-bin which does weird stuff to the spline

        log10_reco_e_binedges = shift(self.bin_edges[dec_idx, tE_idx])
        pdf = self.hists[dec_idx, tE_idx]

        spline = Spline1D(pdf, log10_reco_e_binedges, norm=True)

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
        detailed: int = 0,
        spectrum: str = "atmo",
    ):
        """
        Calculate rates using a linear transformation
        on the binning of the two lowest energy bins of the IRF
        new bins = old bins * a + b
        """

        sindec_min = self.sindec_min
        sindec_max = self.sindec_max
        ereco_pdfs = self.make_shifted_ereco_pdfs(a1, a2, b1, b2)

        def run(El, Eh):
            cdfs = []
            # Precompute the pdf integrals, i.e. cdf, between the Ereco edges
            # for all tE bins. When asked for the integrand, only look up value
            for c, logE in enumerate(np.arange(2.25, 8.76, 0.5)):
                pdf = ereco_pdfs[c]

                def _cdf(x):
                    return pdf(x) / pdf.norm

                cdf = quad(_cdf, np.log10(El), np.log10(Eh), limit=200, full_output=1)[
                    0
                ]
                cdfs.append(cdf)

            def atmo_integrand(logE):
                etrue_idx = np.digitize(logE, self.tE_bin_edges) - 1
                """
                pdf = ereco_pdfs[etrue_idx]

                def _cdf(x):
                    return pdf(x) / pdf.norm

                cdf = quad(_cdf, np.log10(El), np.log10(Eh))[0]
                """

                def sin_dec_int(sindec, logE):
                    return self._aeff.eff_area_spline((logE, -sindec)) * np.power(
                        10, self._atmo._flux_spline(-sindec, logE).squeeze()
                    )

                sin_dec_integrated = quad(
                    sin_dec_int, sindec_min, sindec_max, args=(logE), limit=200
                )[0]                
                return (
                    cdfs[etrue_idx]
                    * sin_dec_integrated
                    * np.power(10, logE)
                    * np.log(10)
                )

            def diff_integrand(logE):
                etrue_idx = np.digitize(logE, self.tE_bin_edges) - 1
                """
                pdf = ereco_pdfs[etrue_idx]

                def _cdf(x):
                    return pdf(x) / pdf.norm

                cdf = quad(_cdf, np.log10(El), np.log10(Eh))[0]
                """

                def sin_dec_int(sindec, logE):
                    return self._aeff.eff_area_spline((logE, -sindec)) * self._diff(
                        np.power(10, logE) * u.GeV, np.arcsin(sindec) * u.rad, 0 * u.rad
                    ).to_value(1 / u.GeV / u.cm**2 / u.s / u.sr)

                sin_dec_integrated = quad(
                    sin_dec_int, sindec_min, sindec_max, args=(logE), limit=200
                )[0]

                return (
                    cdfs[etrue_idx]
                    * sin_dec_integrated
                    * np.power(10, logE)
                    * np.log(10)
                )

            if spectrum == "atmo":
                integrand = atmo_integrand
            elif spectrum == "diffuse":
                integrand = diff_integrand
            else:
                raise ValueError("`spectrum` must be `atmo` or `diffuse`")

            if detailed == 2:
                # use effective area binning, 10 bins per decade
                integrals = [
                    quad(integrand, Elow, Ehigh - 1e-4, limit=200, full_output=0)[0]
                    * 2
                    * np.pi
                    for Elow, Ehigh in zip(
                        np.log10(self.all_Ebins[:-1]), np.log10(self.all_Ebins[1:])
                    )
                ]
            elif detailed == 1:
                # use IRF binning, 2 bins per decade
                integrals = [
                    quad(integrand, Elow, Ehigh - 1e-4, limit=200, full_output=0)[0]
                    * 2
                    * np.pi
                    for Elow, Ehigh in zip(
                        self.tE_bin_edges[:-1], self.tE_bin_edges[1:]
                    )
                ]
            elif detailed == 0:
                # integral is over true energy, no splitting of true energies
                integrals = (
                    quad(integrand, 2, 9, limit=10_000, full_output=0)[0] * 2 * np.pi
                )
            else:
                raise ValueError("`detailed` must be 0, 1 or 2.")
            return integrals

        # loop over reconstructed energies
        rate = Parallel(n_jobs=self.Ebins.size - 1, backend="loky", prefer="threads")(
            delayed(run)(El, Eh) for (El, Eh) in zip(self.Ebins[:-1], self.Ebins[1:])
        )
        # 1e4 to convert atmo flux from /cm**2 to /m**2
        return np.array(rate) * 1e4

    def calc_rates_from_2d_splines(self, detailed: int = 0, spectrum="atmo"):

        sindec_min = self.sindec_min
        sindec_max = self.sindec_max

        # Boundaries are reconstructed energies
        def run(El, Eh):
            # Precompute the pdf integrals, i.e. cdf, between the Ereco edges
            # for all tE bins. When asked for the integrand, only look up value

            def atmo_integrand(logE):

                # This part can be split of from the energy integral and be calculated outside
                # energies (true, in this case) is only a parameter but not an actual variable
                def sin_dec_int(sindec, logE):
                    return self._aeff.eff_area_spline((logE, -sindec)) * np.power(
                        10, self._atmo._flux_spline(-sindec, logE).squeeze()
                    )

                sin_dec_integrated = quad(
                    sin_dec_int, sindec_min, sindec_max, args=(logE), limit=200
                )[0]

                return (
                    self._eres.prob_Edet_above_threshold(
                        np.power(10, logE) * u.GeV,
                        El * u.GeV,
                        self._dec_middle,
                        Eh * u.GeV,
                        use_interpolation=True,
                    )
                    * sin_dec_integrated
                    * np.power(10, logE)
                    * np.log(10)
                )

            def diff_integrand(logE):
                etrue_idx = np.digitize(logE, self.tE_bin_edges) - 1

                def sin_dec_int(sindec, logE):
                    return self._aeff.eff_area_spline((logE, -sindec)) * self._diff(
                        np.power(10, logE) * u.GeV, np.arcsin(sindec) * u.rad, 0 * u.rad
                    ).to_value(1 / u.GeV / u.cm**2 / u.s / u.sr)

                sin_dec_integrated = quad(
                    sin_dec_int, sindec_min, sindec_max, args=(logE), limit=200
                )[0]

                return (
                    self._eres.prob_Edet_above_threshold(
                        np.power(10, logE) * u.GeV,
                        El * u.GeV,
                        self._dec_middle,
                        Eh * u.GeV,
                        use_interpolation=True,
                    )
                    * sin_dec_integrated
                    * np.power(10, logE)
                    * np.log(10)
                )

            if spectrum == "atmo":
                integrand = atmo_integrand
            elif spectrum == "diffuse":
                integrand = diff_integrand

            if detailed == 2:
                # use effective area binning, 10 bins per decade
                integrals = [
                    quad(integrand, Elow, Ehigh - 1e-4, limit=200, full_output=0)[0]
                    * 2
                    * np.pi
                    for Elow, Ehigh in zip(
                        np.log10(self.all_Ebins[:-1]), np.log10(self.all_Ebins[1:])
                    )
                ]
            elif detailed == 1:
                # use IRF binning, 2 bins per decade
                integrals = [
                    quad(integrand, Elow, Ehigh - 1e-4, limit=200, full_output=0)[0]
                    * 2
                    * np.pi
                    for Elow, Ehigh in zip(
                        self.tE_bin_edges[:-1], self.tE_bin_edges[1:]
                    )
                ]
            elif detailed == 0:
                # integral is over true energy, no splitting of true energies
                integrals = (
                    quad(integrand, 2, 9, limit=10_000, full_output=0)[0] * 2 * np.pi
                )
            else:
                raise ValueError("`detailed` must be 0, 1 or 2.")
            return integrals

        # loop over reconstructed energies
        rate = Parallel(n_jobs=self.Ebins.size - 1, backend="loky", prefer="threads")(
            delayed(run)(El, Eh) for (El, Eh) in zip(self.Ebins[:-1], self.Ebins[1:])
        )
        # 1e4 to convert atmo flux from /cm**2 to /m**2
        return np.array(rate) * 1e4

    def exp_value(self, rates):
        # Calculates the (log) exp value of reconstructed energies
        # only use detailed==1
        assert rates.shape == (self.Ebins_c.size, self.tE_binc.size)

        # only calculate for the lowest three Etrue bins, rest is not interesting for the fit
        av = np.zeros(3)
        for c in range(3):
            av[c] = np.average(self.logE_binc, weights=rates[:, c])
        return av

    def likelihood(self, a1, a2, b1, b2, detailed: int = 0):
        # TODO implement detailed kwarg

        r = self.calc_rates(a1, a2, b1, b2, detailed)
        if detailed:
            rates = r.sum(axis=1)
        else:
            rates = r
        mask = rates > 0.0
        Nex = rates * self.lifetime
        scale = self.exp_N.sum() / Nex.sum()
        llh = -2 * np.sum(
            scale * Nex[mask] + self.exp_N[mask] * np.log(scale * Nex[mask])
        )
        if detailed == 1:
            exp = self.exp_value(r)

        return llh, exp

    def scan(self, a1, a2, b1, b2, n_jobs: int = 20, detailed: int = 0):
        """
        Scan over the parameter space
        """
        if detailed > 1:
            raise ValueError("`detailed` must be 0 or 1 for a scan")
        aa1, aa2, bb1, bb2 = np.meshgrid(a1, a2, b1, b2)
        ll = np.zeros(aa1.shape).flatten()
        exp = np.zeros((aa1.size, 3))
        with tqdm_joblib(desc="likelihood", total=aa1.size) as progress_bar:
            ret = Parallel(n_jobs=n_jobs, backend="loky", prefer="threads")(
                delayed(self.likelihood)(a1, a2, b1, b2, detailed=detailed)
                for (a1, a2, b1, b2) in zip(
                    aa1.flatten(), aa2.flatten(), bb1.flatten(), bb2.flatten()
                )
            )

        for c, (llh, expect) in enumerate(ret):
            ll[c] = llh
            exp[c] = expect
        self.ll = ll.reshape(aa1.shape)
        self.exp = exp.reshape((*aa1.shape, 3))
        return ll

    def plot_detailed_rates(
        self, detailed_rates, figsize=(6.4, 4.8), grid: bool = False
    ):

        if len(detailed_rates.shape) == 1:
            detailed = 0
        elif detailed_rates.size == self.Ebins_c.size * self.tE_binc.size:
            detailed = 1
        elif detailed_rates.size == self.Ebins_c.size * (self.all_Ebins.size - 1):
            detailed = 2
        else:
            raise ValueError("Something is fishy")

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        linestyles = ["solid", "dashed", "dotted", "dashdot", (0, (3, 5, 1, 5, 1, 5))]

        fig, axs = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [5, 1], "hspace": 0},
            figsize=figsize,
        )
        ax = axs[0]
        ax.scatter(self.Ebins_c, self.exp_rate, marker="+", color="red", label="data")
        _bin = 0
        IRF_ebins = np.arange(2, 9.1, 0.5)

        if detailed == 0:
            ax.stairs(detailed_rates, self.Ebins, label="simulation", color="black")
            summed_rates = detailed_rates
        elif detailed == 1:
            default_cycler = cycler(color=colors)
            ax.set_prop_cycle(default_cycler)

            ax.stairs(
                detailed_rates.sum(axis=1),
                self.Ebins,
                color="black",
                label="simulation",
            )

            for c, rate in enumerate(detailed_rates.T):
                if c == 7:
                    break
                ax.stairs(
                    rate,
                    self.Ebins,
                    label=rf"$E=10^{{{IRF_ebins[_bin]:.1f}-{IRF_ebins[_bin+1]:.1f}}}\si{{\GeV}}$",
                )
                _bin += 1
            summed_rates = detailed_rates.sum(axis=1)

        else:
            default_cycler = cycler(color=colors) * cycler(linestyle=linestyles)
            ax.set_prop_cycle(default_cycler)
            ax.stairs(
                detailed_rates.sum(axis=1),
                self.Ebins,
                color="black",
                label="simulation",
            )
            for c, rate in enumerate(detailed_rates.T):
                if c == 5 * 7:
                    break
                if c % 5 == 0:
                    ax.stairs(
                        rate,
                        self.Ebins,
                        label=rf"$E=10^{{{(IRF_ebins[_bin]+IRF_ebins[_bin+1])/2:.2f}}}$ GeV",
                    )
                    _bin += 1
                else:
                    ax.stairs(
                        rate,
                        self.Ebins,
                    )
            summed_rates = detailed_rates.sum(axis=1)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_ylabel(r"event rate per bin [1/s]")

        ax.set_xlim(1e2, 1e5)
        _max = np.max(np.vstack((summed_rates, self.exp_rate))) * 2
        _min = np.min(self.exp_rate) / 2
        ax.set_ylim(_min, _max)
        ax.legend()
        if grid:
            ax.grid()
        ax = axs[1]
        ax.scatter(
            self.Ebins_c,
            self.exp_rate / summed_rates,
            color="black",
            marker="+",
        )
        ax.set_ylim(0.6, 1.6)
        if grid:
            ax.grid()
        ax.set_xlabel(r"$\hat{E}~[\si{\giga\electronvolt}]$")
        ax.set_ylabel(r"$\frac{\text{data}}{\text{sim}}$")

        return fig, axs

    def plot_hist_and_spline(self, tE_idx, dec_idx):
        spline = self.create_shifted_reco_pdf(tE_idx, dec_idx)
        n, bins = self.create_shifted_reco_pdf(tE_idx, dec_idx, hist=True)
        x = np.linspace(bins[0], bins[-1], 1_000)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, spline(x))
        binc = bins[:-1] + np.diff(bins) / 2
        ax.bar(binc, n, width=np.diff(bins), alpha=0.5, color="grey")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\hat{E}~[\si{\giga\electronvolt}]$")
        ax.set_ylabel("pdf")

        return fig, ax
