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
from hierarchical_nu.utils.plotting import lighten_color

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from matplotlib.patches import Patch
from astropy import units as u
from astropy.coordinates import SkyCoord
import h5py
from pathlib import Path
from tqdm.autonotebook import tqdm
from typing import Union
import logging
from line_profiler import profile

logger = logging.getLogger(__file__)


class PPC:

    def __init__(self, fit: StanFit, parser: ConfigParser):
        """
        :param fit: StanFit used to sample parameters from, may be reloaded or alive
        :param parser: ConfigParser instance used to create the fit
        """
        self._parser = parser
        self._config = parser._hnu_config.ppc_config
        self._fit = fit
        self._ran_setup = False
        self._events = []
        self._N = []

    def load(self, path):
        """
        Load events of saved PPC
        :param path: Path to saved PPC
        """

        i = 0
        while True:
            try:
                self._events.append(
                    Events.from_file(
                        path,
                        group_name=f"events_{i}",
                    )
                )
                i += 1
            except Exception as e:
                break
        with h5py.File(path, "r") as f:
            self._N = f["meta_data/N"][()]
            self._Nex_et = f["meta_data/Nex"][()]
            self._N_comp = f["meta_data/N_comp"][()]

    def _plot_radial_ppc(self, bins):
        # TODO
        raise NotImplementedError()

    def _plot_energy_ppc(self, bins):
        # TODO
        raise NotImplementedError()

    def plot(
        self,
        bins_Ereco=np.geomspace(1e2, 1e7, 8 * 5),
        bins_ang_sep_sq=np.arange(0, 5.1, 1 / 3),
        quantiles=[90, 68, 50],
        figsize=(6, 3),
        colors=None,
        alpha=0.2,
        axs=None,
    ):
        """
        Plot diagnostic PPCs
        :param bins_Ereco: bins in GeV for reconstructed muon energy plot
        :param bins_ang_seq_eq: binning of angular distance to source in units of degrees squared
        :param quantiles: list of quantiles between 0 and 100 to plot colour bands of
        :param figsize: (width, height) of figure
        :param colors: Provide some colour for plotting the bands
        """

        quantiles = np.atleast_1d(quantiles)
        q_low = (50 - quantiles / 2) / 100
        q_high = (50 + quantiles / 2) / 100
        q_low, q_high
        
        col_all = "C1"
        col_source = "C3"
        col_bg = "C2"

        xlabels = [
            r"distance to source squared [deg$^2$]",
            r"$\hat{E}~[\si{\GeV}]$",
        ]

        try:
            fig = axs[0].gcf()
        except:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        

        # Squared angular distance posterior predictive check
        # use squared distance for approximately flat background distribution
        ax = axs[0]

        coords = SkyCoord(
            ra=self._fit.sources.point_source[0].ra,
            dec=self._fit.sources.point_source[0].dec,
            frame="icrs",
        )

        hists_all = np.array(
            [
                np.histogram(
                    coords.separation(_.coords).deg ** 2,
                    bins_ang_sep_sq,
                )[0]
                for _ in self._events
            ]
        )

        hists_source = np.array(
            [
                np.histogram(
                    coords.separation(_[:N].coords).deg ** 2,
                    bins_ang_sep_sq,
                )[0]
                for (N, _) in zip(self._N, self._events)
            ]
        )

        hists_bg = np.array(
            [
                np.histogram(
                    coords.separation(_[N:].coords).deg ** 2,
                    bins_ang_sep_sq,
                )[0]
                for (N, _) in zip(self._N, self._events)
            ]
        )

        ql_all = np.quantile(hists_all, q_low, axis=0)
        qh_all = np.quantile(hists_all, q_high, axis=0)
        ql_source = np.quantile(hists_source, q_low, axis=0)
        qh_source = np.quantile(hists_source, q_high, axis=0)
        ql_bg = np.quantile(hists_bg, q_low, axis=0)
        qh_bg = np.quantile(hists_bg, q_high, axis=0)

        ang_sep = coords.separation(self._fit.events.coords).deg
        obs = np.histogram(ang_sep**2, bins=bins_ang_sep_sq)[0]

        for (
            q,
            l,
            h,
        ) in zip(quantiles, ql_all, qh_all):
            for c, (bl, bh) in enumerate(
                zip(bins_ang_sep_sq[:-1], bins_ang_sep_sq[1:])
            ):
                ax.fill_between(
                    [bl, bh],
                    l[c],
                    h[c],
                    color=col_all,
                    alpha=alpha,
                    edgecolor="none",
                )

        for (
            q,
            l,
            h,
        ) in zip(quantiles, ql_source, qh_source):
            for c, (bl, bh) in enumerate(
                zip(bins_ang_sep_sq[:-1], bins_ang_sep_sq[1:])
            ):
                ax.fill_between(
                    [bl, bh],
                    l[c],
                    h[c],
                    color=col_source,
                    alpha=alpha,
                    edgecolor="none",
                )
        for (
            q,
            l,
            h,
        ) in zip(quantiles, ql_bg, qh_bg):
            for c, (bl, bh) in enumerate(
                zip(bins_ang_sep_sq[:-1], bins_ang_sep_sq[1:])
            ):
                ax.fill_between(
                    [bl, bh],
                    l[c],
                    h[c],
                    color=col_bg,
                    alpha=alpha,
                    edgecolor="none",
                )

        ax.stairs(
            obs,
            bins_ang_sep_sq,
            color="black",
            lw=1,
        )

        def transform(x):
            """
            Function for axis rescaling in pyplot
            """

            output = np.zeros_like(x)
            output[x <= 0.0] = 0.0
            output[x > 0.0] = np.sqrt(x[x > 0.0])
            return output

        def inverse(x):
            """
            Function for axis rescaling in pyplot
            """

            return np.power(x, 2)


        #ax.set_xticks(np.arange(0, 25.1, 5))
        ax.set_xlim(0, bins_ang_sep_sq.max())
        ax.set_xlabel(xlabels[0])
        ax.set_ylabel("counts per bin")

        secax = ax.secondary_xaxis("top", functions=(transform, inverse))
        secax.set_xlabel("linear distance~[deg]")
        secax.set_xticks(np.arange(5.1))

        # Detected energy posterior predictive check
        ax = axs[1]

        hists_all = np.array(
            [
                np.histogram(_.energies.to_value(u.GeV), bins=bins_Ereco)[0]
                for _ in self._events
            ]
        )
        
        hists_source = np.array(
            [
                np.histogram(_[:N].energies.to_value(u.GeV), bins=bins_Ereco)[0]
                for (N, _) in zip(self._N, self._events)
            ]
        )
        
        hists_bg = np.array(
            [
                np.histogram(_[N:].energies.to_value(u.GeV), bins=bins_Ereco)[0]
                for (N, _) in zip(self._N, self._events)
            ]
        )

        obs = np.histogram(self._fit.events.energies.to_value(u.GeV), bins=bins_Ereco)[
            0
        ]

        ql_all = np.quantile(hists_all, q_low, axis=0)
        qh_all = np.quantile(hists_all, q_high, axis=0)
        
        ql_source = np.quantile(hists_source, q_low, axis=0)
        qh_source = np.quantile(hists_source, q_high, axis=0)
        
        ql_bg = np.quantile(hists_bg, q_low, axis=0)
        qh_bg = np.quantile(hists_bg, q_high, axis=0)

        for (
            q,
            l,
            h,
        ) in zip(quantiles, ql_all, qh_all):
            for c, (bl, bh) in enumerate(zip(bins_Ereco[:-1], bins_Ereco[1:])):
                ax.fill_between(
                    [bl, bh],
                    l[c],
                    h[c],
                    color=col_all,
                    alpha=alpha,
                    edgecolor="none",
                )
                
        for (
            q,
            l,
            h,
        ) in zip(quantiles, ql_source, qh_source):
            for c, (bl, bh) in enumerate(zip(bins_Ereco[:-1], bins_Ereco[1:])):
                ax.fill_between(
                    [bl, bh],
                    l[c],
                    h[c],
                    color=col_source,
                    alpha=alpha,
                    edgecolor="none",
                )
                
        for (
            q,
            l,
            h,
        ) in zip(quantiles, ql_bg, qh_bg):
            for c, (bl, bh) in enumerate(zip(bins_Ereco[:-1], bins_Ereco[1:])):
                ax.fill_between(
                    [bl, bh],
                    l[c],
                    h[c],
                    color=col_bg,
                    alpha=alpha,
                    edgecolor="none",
                )

        ax.stairs(obs, bins_Ereco, color="black", label="Observed", lw=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabels[1])
        ax.set_ylabel("counts per bin")
        
        
        signal_patch = Patch(edgecolor=None, facecolor="C3", alpha=0.3)
        signal_label = "PS events"

        bg_patch = Patch(edgecolor=None, facecolor="C2", alpha=0.3)
        bg_label = "bkg events"

        all_patch = Patch(edgecolor=None, facecolor="C0", alpha=0.3)
        all_label = "all events"

        handles, labels = ax.get_legend_handles_labels()
        handles += [signal_patch, bg_patch, all_patch]
        labels += [signal_label, bg_label, all_label]
        ax.legend(handles, labels)
        
        fig.subplots_adjust(wspace=0.3)

        return fig, axs, secax

    def setup(self, use_data_as_bg: bool = False):
        """
        Setup all objects to run simulations
        :param use_data_as_bg: Bool, defaults to false. If true, use RA-scrambled data as background
            and only simulate the point source
        """

        try:
            if use_data_as_bg:
                # Disable both background components and instead use RA-scrambled data
                self._parser._hnu_config.parameter_config.diffuse = False
                self._parser._hnu_config.parameter_config.atmospheric = False
                self._use_data_as_bg = True
            else:
                self._use_data_as_bg = False
            sources = self._parser.sources
            dm = self._parser.detector_model
            obs_time = self._parser.obs_time
            self._parser.ROI
            self._sources = sources
            self._obs_time = obs_time
            self._dm = dm
            self._sources = sources
            self._sim = self._parser.create_simulation(sources, dm, obs_time)
            self._sim.precomputation()
            self._sim.generate_stan_code()
            self._sim.compile_stan_code()
            self._Nex_et = []
            self._N = []
            self._N_comp = []
            self._Lambda = []
            self._ran_setup = True
            self._events = []
        except Exception as e:
            # Except any error
            self._ran_setup = False
            raise (e)

    @profile
    def run(
        self,
        show_progress: bool = True,
        output_file: Union[str, Path] = Path("ppc.h5"),
        overwrite: bool = False,
    ):
        """
        Method to run all simulations and save output
        :param show_progres: Set to True if progress par shall be displayed, defaults to True
        :param output_file: str or Path to output file, defaults to ./ppc.h5
        """

        if not self._ran_setup:
            raise ValueError("Need to run setup before running")

        if not isinstance(output_file, Path):
            output_file = Path(output_file)

        self._events = []
        rng = np.random.default_rng(seed=self._config.seed)
        sources = self._sources
        sim = self._sim
        config = self._parser._hnu_config
        point_sources = sources.point_source
        fit = self._fit

        # Peace and quiet
        cmdstanpy_logger = logging.getLogger("cmdstanpy")
        cmdstanpy_logger.disabled = True

        seed = self._config.seed

        if self._use_data_as_bg:
            bg_events = Events.from_event_files(*self._parser.detector_model, apply_roi=False)
            # Mask out all events with declinations outside the specified ROIs
            # for speedup when scrambling RA and MJDs
            dec_mask = np.logical_and(bg_events.coords.dec <= ROIList.DEC_max(), bg_events.coords.dec >= ROIList.DEC_min())
            bg_events = bg_events[dec_mask]

        with tqdm(total=self._config.n_samples, disable=not show_progress) as pbar:
            for i in range(self._config.n_samples):
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
                    if "eta" in fit.keys() or "ps_0_eta" in fit.keys():
                        # use pressure ratio instead of luminosity
                        try:
                            P = Parameter.get_parameter("pressure_ratio")

                            P.fixed = False
                            P.par_range = (0.0, max(fit["pressure_ratio"].flatten()))
                            P.value = fit["pressure_ratio"].flatten()[rint]
                            P.fixed = True
                        except:
                            for c, ps in enumerate(point_sources):
                                name = f"ps_{c}_pressure_ratio"
                                P = Parameter.get_parameter(name)
                                P.fixed = False
                                P.par_range = (
                                    0.0,
                                    max(fit["pressure_ratio"].flatten()),
                                )
                                P.value = fit["pressure_ratio_ind"][..., c].flatten()[
                                    rint
                                ]
                                P.fixed = True
                    else:
                        try:
                            lumi = Parameter.get_parameter("luminosity")

                            lumi.fixed = False
                            try:
                                lumi.value = fit["L"].flatten()[rint] * u.GeV / u.s
                            except (KeyError, ValueError):
                                lumi.value = (
                                    fit["L_ind"][..., 0].flatten()[rint] * u.GeV / u.s
                                )
                            lumi.fixed = True
                        except:
                            # TODO: add some ParameterNotFound Exception
                            for c, ps in enumerate(point_sources):
                                name = f"ps_{c}_luminosity"
                                lumi = Parameter.get_parameter(name)
                                lumi.fixed = False
                                lumi.value = (
                                    fit["L"][..., c].flatten()[rint] * u.GeV / u.s
                                )
                                lumi.fixed = True

                    for param_name in ["src_index", "beta_index", "E0_src", "eta"]:
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
                sim.compute_c_values(inplace=True)

                seed += 1
                while True:
                    sim.run(seed=seed, show_progress=False, show_console=False)
                    if sim.events is not None or (
                        point_sources and self._use_data_as_bg
                    ):
                        # I want to break free, but only if at least one event is sampled,
                        # or we use data to estimate the background
                        break
                    # elif point_sources:
                    #    continue

                    seed += 1

                # Output some more data for debugging
                sim._get_expected_Nnu(sim._get_sim_inputs()).copy()
                self._Nex_et.append(sim._Nex_et.copy())
                # If no events are sampled and we merge with background,
                # the Lambda array should read N_ps + 1 for all events (take number from further down somehow)
                out = sim._sim_output.stan_variable
                self._N.append(out("N_").astype(int))
                try:
                    self._Lambda.append(out("Lambda").astype(int).squeeze() - 1)
                except ValueError:
                    self._Lambda.append([[np.nan] * sim._sources.N])
                self._N_comp.append(out("N_comp_").astype(int))

                if self._use_data_as_bg:
                    bg_events.scramble_mjd(seed=self._config.seed + i)
                    bg_events.scramble_ra(seed=self._config.seed + i)
                    mask = bg_events._apply_ROIS()
                    _bg_events = bg_events[mask]
                    try:
                        events = sim.events.merge(_bg_events)
                    except AttributeError as e:
                        events = _bg_events

                else:
                    events = sim.events
                if i == 0:
                    output_file = events.to_file(
                        output_file,
                        append=False,
                        group_name=f"events_{i}",
                        overwrite=overwrite,
                    )
                else:
                    events.to_file(output_file, append=True, group_name=f"events_{i}")
                self._events.append(events)
                pbar.update(1)

        N = np.concatenate(self._N)
        N_comp = np.concatenate(self._N_comp)
        Nex = np.concatenate(self._Nex_et)

        with h5py.File(output_file, "r+") as f:
            g = f.create_group("meta_data")
            g.create_dataset("N", data=N)
            g.create_dataset("N_comp", data=N_comp)
            g.create_dataset("Nex", data=Nex)
            l = g.create_group("Lambdas")
            for c, arr in enumerate(self._Lambda):
                l.create_dataset(str(c), data=arr)

        return output_file
