"""
Run posterior predictive checks for a fit
"""

from hierarchical_nu.fit import StanFit
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.utils.config_parser import ConfigParser
from hierarchical_nu.utils.roi import ROIList
from hierarchical_nu.events import Events

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from astropy import units as u
from astropy.coordinates import SkyCoord
import os
from pathlib import Path
from tqdm.autonotebook import tqdm
import logging

logger = logging.getLogger(__file__)


class PPC:
    """
    :param fit: StanFit used to sample parameters from, may be reloaded or alive
    :param parser: ConfigParser instance used to create the fit
    """

    def __init__(self, fit: StanFit, parser: ConfigParser):
        self._parser = parser
        self._config = parser._hnu_config
        self._fit = fit
        self._ran_setup = False
        self._events = []

    def load(self, path):
        i = 0
        while True:
            try:
                self._events.append(
                    Events.from_file(path, group_name=f"events_{i}", apply_cuts=False)
                )
                i += 1
            except:
                break

    def _plot_racial_ppc(self, bins):
        pass

    def _plot_energy_ppc(self, bins):
        pass

    def plot(
        self,
        bins_Ereco=np.geomspace(1e2, 1e7, 8 * 50),
        bins_ang_sep_sq=np.arange(0, 25.1, 1 / 3),
        quantiles=[30, 60, 90],
        figsize=(6, 3),
    ):
        """
        Plot diagnostic PPCs
        :param bins_Ereco: bins in GeV for reconstructed muon energy plot
        :param bins_ang_seq_eq: binning of angular distance to source in units of degrees squared
        :param quantiles: list of quantiles between 0 and 100 to plot colour bands of
        :param figsize: (width, height) of figure
        """

        quantiles = np.atleast_1d(quantiles)
        q_low = (50 - quantiles / 2) / 100
        q_high = (50 + quantiles / 2) / 100
        q_low, q_high

        xlabels = [
            r"distance to source squared [deg$^2$]",
            r"$\hat{E}~[\si{\GeV}]$",
        ]

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # Squred angular distance posterior predictive check
        # use squared distance for approximately flat background distribution
        ax = axs[0]

        coords = SkyCoord(
            ra=self._fit.sources.point_source[0].ra,
            dec=self._fit.sources.point_source[0].dec,
            frame="icrs",
        )

        hists = np.array(
            [
                np.histogram(
                    coords.separation(_.coords).deg ** 2,
                    bins_ang_sep_sq,
                )[0]
                for _ in self._events
            ]
        )

        ql = np.quantile(hists, q_low, axis=0)
        qh = np.quantile(hists, q_high, axis=0)

        ang_sep = coords.separation(self._fit.events.coords).deg
        obs = np.histogram(ang_sep**2, bins=bins_ang_sep_sq)[0]

        idxs = np.arange(obs.size + 1)
        idxs[-1] = idxs[-2]

        for (
            q,
            l,
            h,
        ) in zip(quantiles, ql, qh):
            for c, (bl, bh) in enumerate(
                zip(bins_ang_sep_sq[:-1], bins_ang_sep_sq[1:])
            ):
                ax.fill_between(
                    [bl, bh],
                    l[c],
                    h[c],
                    color=viridis(0.0),
                    alpha=0.2,
                    edgecolor="none",
                )

        ax.step(
            bins_ang_sep_sq,
            obs[idxs],
            c="black",
            label="Observed",
            where="post",
            lw=1,
        )

        ax.legend()
        ax.set_xlabel(xlabels[0])
        ax.set_ylabel("counts per bin")

        # Detected energy posterior predictive check
        ax = axs[1]

        hists = np.array(
            [
                np.histogram(_.energies.to_value(u.GeV), bins=bins_Ereco)[0]
                for _ in self._events
            ]
        )
        obs = np.histogram(self._fit.events.energies.to_value(u.GeV), bins=bins_Ereco)[
            0
        ]

        ql = np.quantile(hists, q_low, axis=0)
        qh = np.quantile(hists, q_high, axis=0)

        for (
            q,
            l,
            h,
        ) in zip(quantiles, ql, qh):
            for c, (bl, bh) in enumerate(zip(bins_Ereco[:-1], bins_Ereco[1:])):
                ax.fill_between(
                    [bl, bh],
                    l[c],
                    h[c],
                    color=viridis(0.0),
                    alpha=0.2,
                    edgecolor="none",
                )

        idxs = np.arange(obs.size + 1)
        idxs[-1] = idxs[-2]

        ax.step(bins_Ereco, obs[idxs], c="black", label="Observed", where="post", lw=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabels[1])
        ax.set_ylabel("counts per bin")
        ax.legend()

        return fig, axs

    def setup(self):
        """
        Setup all objects to run simulations
        """

        try:
            sources = self._parser.sources
            dm = self._parser.detector_model
            obs_time = self._parser.obs_time
            self._parser.ROI
            self._sources = sources
            self._sim = self._parser.create_simulation(sources, dm, obs_time)
            self._sim.precomputation()
            self._sim.generate_stan_code()
            self._sim.compile_stan_code()
            self._ran_setup = True
        except Exception as e:
            # Except any error
            self._ran_setup = False
            raise (e)

    def run(self, show_progress: bool = True, output_file: Path = Path("ppc.h5")):
        """
        Method to run all simulations and save output
        :param show_progres: Set to True if progress par shall be displayed, default
        :param output_file: Path to output file, defaults to ./ppc.h5
        """

        if not self._ran_setup:
            raise ValueError("Need to run setup before running")

        rng = np.random.default_rng(seed=self._config.ppc_config.seed)
        sources = self._sources
        sim = self._sim
        config = self._config
        point_sources = sources.point_source
        fit = self._fit

        # Peace and quiet
        cmdstanpy_logger = logging.getLogger("cmdstanpy")
        cmdstanpy_logger.disabled = True

        with tqdm(total=config.ppc_config.n_samples, disable=not show_progress) as pbar:
            for i in range(config.ppc_config.n_samples):
                pbar.set_description("Running simulations")

                rint = rng.integers(low=0, high=fit["Nex"].size)

                if sources.atmospheric:
                    F_atmo = Parameter.get_parameter("F_atmo")

                    F_atmo.fixed = False
                    F_atmo.value = fit["F_atmo"].flatten()[rint] * (1 / u.m**2 / u.s)
                    F_atmo.fixed = True

                if sources.diffuse:
                    diff_norm = Parameter.get_parameter("diffuse_norm")
                    diff_index = Parameter.get_parameter("diff_index")

                    diff_index.value = fit["diff_index"].flatten()[rint]
                    diff_norm.fixed = False
                    diff_norm.value = fit["diffuse_norm"].flatten()[rint] * (
                        1 / u.GeV / u.m**2 / u.s
                    )
                    diff_norm.fixed = True

                if point_sources:
                    try:
                        lumi = Parameter.get_parameter("luminosity")

                        lumi.fixed = False
                        lumi.value = fit["L"].flatten()[rint] * u.GeV / u.s
                        lumi.fixed = True
                    except:
                        for c, ps in enumerate(point_sources):
                            name = f"ps_{c}_luminosity"
                            lumi = Parameter.get_parameter(name)
                            lumi.fixed = False
                            lumi.value = fit["L"][..., c].flatten()[rint] * u.GeV / u.s
                            lumi.fixed = True

                    for param_name in ["src_index", "beta_index", "E0_src"]:
                        if param_name in config.parameter_config.fit_params:
                            if not config.parameter_config.share_src_index:
                                name = f"ps_{c}_{param_name}"
                                param = Parameter.get_parameter(name)
                                if param_name != "E0_src":
                                    param.value = fit[param_name][..., :].flatten()[
                                        rint
                                    ]
                                else:
                                    param.value = (
                                        fit[param_name][..., :].flatten()[rint] * u.GeV
                                    )
                            else:
                                param = Parameter.get_parameter(param_name)
                                if param_name != "E0_src":
                                    param.value = fit[param_name].flatten()[rint]
                                else:
                                    param.value = (
                                        fit[param_name].flatten()[rint] * u.GeV
                                    )

                sim.run(
                    seed=config.ppc_config.seed, show_progress=False, show_console=False
                )
                if i == 0:
                    output_file = sim.events.to_file(
                        output_file, append=False, group_name=f"events_{i}"
                    )
                else:
                    sim.events.to_file(
                        output_file, append=True, group_name=f"events_{i}"
                    )
                self._events.append(sim.events)
                pbar.update(1)
        return output_file
