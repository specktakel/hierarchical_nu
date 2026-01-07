import numpy as np
import os
import h5py
import logging
import collections
from astropy import units as u
from astropy.coordinates import SkyCoord
from typing import List, Union, Dict, Callable, Iterable
import corner
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm

try:
    from roque_cmap import roque_chill

    CMAP = roque_chill().reversed()
except ModuleNotFoundError:
    CMAP = "viridis_r"

import ligo.skymap.plot
import arviz as av
from pathlib import Path

from math import ceil
from time import time as thyme

from cmdstanpy import CmdStanModel

from hierarchical_nu.source.source import Sources, PointSource, icrs_to_uv, uv_to_icrs
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import (
    IsotropicDiffuseBG,
    LogParabolaSpectrum,
    PGammaSpectrum,
    TwiceBrokenPowerLaw,
    PowerLawSpectrum,
)
from hierarchical_nu.source.seyfert_model import SeyfertNuMuSpectrum
from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.detector.icecube import EventType, CAS, Refrigerator
from hierarchical_nu.detector.r2021 import (
    R2021EnergyResolution,
)
from hierarchical_nu.precomputation import ExposureIntegral
from hierarchical_nu.events import Events
from hierarchical_nu.priors import Priors, UnitPrior, MultiSourcePrior, NoPriorSetError, AngularPrior
from hierarchical_nu.source.source import uv_to_icrs

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.utils.git import git_hash
from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.utils.config_parser import ConfigParser
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.utils.roi import ROIList
from .source.source_info import SourceInfo

from omegaconf import OmegaConf


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class StanFit(SourceInfo):
    """
    To set up and run fits in Stan.
    """

    @u.quantity_input
    def __init__(
        self,
        sources: Sources,
        event_types: Union[EventType, List[EventType]],
        events: Events,
        observation_time: Dict[str, u.quantity.Quantity[u.year]],
        priors: Priors = Priors(),
        atmo_flux_energy_points: int = 100,
        atmo_flux_theta_points: int = 30,
        n_grid_points: int = 50,
        nshards: int = 0,
        use_event_tag: bool = False,
        debug: bool = False,
        reload: bool = False,
    ):
        """
        To set up and run fits in Stan.
        :param sources: instance of Sources
        :param event_types: EventType or List thereof, to be included in the fit
        :param events: instance of Events
        :param observation_time: astropy.units time for single event type or dictionary thereof with event type as key
        :param priors: instance of Priors of parameters
        :param atmo_flux_energy_points: number of points for atmo spectrum energy interpolation
        :param atmo_flux_theta_points: number of points for atmo spectrum cos(theta) interpolation
        :param n_grid_points: number of grid points used per parameter in precomputation of exposure
        :param nshards: number of shards into which data is split. zero or one for single thread, larger will compile code for multithreading
        :param use_event_tag: for multiple ROI set to True to only consider closest point source for each event
        :param debug: set to True for unit testing purposes
        :param reload: set to True if no stan interface should be created, i.e. reloading results
        """

        super().__init__(sources)
        # self._detector_model_type = detector_model
        if not isinstance(event_types, list):
            event_types = [event_types]
        if isinstance(observation_time, u.quantity.Quantity):
            observation_time = {event_types[0]: observation_time}
        if not len(event_types) == len(observation_time):
            raise ValueError(
                "number of observation times must match number of event types"
            )
        self._event_types = event_types
        self._events = events
        self._observation_time = observation_time
        self._n_grid_points = n_grid_points
        self._nshards = nshards
        self._priors = priors
        self._use_event_tag = use_event_tag

        stan_file_name = os.path.join(STAN_GEN_PATH, "model_code")

        self._reload = reload
        if not self._reload:
            self._stan_interface = StanFitInterface(
                stan_file_name,
                self._sources,
                self._event_types,
                priors=priors,
                nshards=nshards,
                atmo_flux_energy_points=atmo_flux_energy_points,
                atmo_flux_theta_points=atmo_flux_theta_points,
                use_event_tag=use_event_tag,
                debug=debug,
                bg=self._bg,
            )
        else:
            logger.debug("Reloading previous results.")

        if sources.atmospheric and self._event_types == [CAS]:
            raise NotImplementedError(
                "AtmosphericNuMuFlux currently only implemented "
                + "for use with NorthernTracksDetectorModel or "
                + "IceCubeDetectorModel"
            )

        if sources.atmospheric and sources.N == 1 and CAS in self._event_types:
            raise NotImplementedError(
                "AtmosphericNuMuFlux as the only source component "
                + "for IceCubeDetectorModel is not implemented. Just use "
                + "NorthernTracksDetectorModel instead."
            )

        # Silence log output
        logger_code_gen = logging.getLogger("hierarchical_nu.backend.code_generator")
        logger_code_gen.propagate = False

        # For use with plot methods
        self._def_var_names = []

        if self._sources.point_source:
            self._def_var_names.append("L")
            if self._fit_index:
                self._def_var_names.append("src_index")
            if self._fit_beta:
                self._def_var_names.append("beta_index")
            if self._fit_Enorm:
                self._def_var_names.append("E0_src")
            if self._fit_eta:
                self._def_var_names.append("eta")
            if self._seyfert:
                self._def_var_names.append("P_R")
            if self._fit_ang_sys:
                self._def_var_names.append("ang_sys_deg")
            self._def_var_names.append("Nex_src")

        if self._sources.diffuse:
            self._def_var_names.append("diffuse_norm")
            self._def_var_names.append("diff_index")

        if self._sources.atmospheric:
            self._def_var_names.append("F_atmo")

        if self._sources.background:
            self._def_var_names.append("Nex_bg")

        self._exposure_integral = collections.OrderedDict()

        self._fit_output = None

    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, p):
        previous_prior = self._priors.to_dict()
        if isinstance(p, Priors):
            new_prior = p.to_dict()
            for k, v in new_prior.items():
                if previous_prior[k].name != v.name:
                    logger.warning(
                        f"Prior type of {k} changed, regenerate and recompile the stan code."
                    )
            self._priors = p
            self._stan_interface._priors = p
        else:
            raise ValueError("Priors must be instance of Priors.")

    @property
    def sources(self):
        return self._sources

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, events: Events):
        if isinstance(events, Events):
            self._events = events
        else:
            raise ValueError("events must be instance of Events")

    @property
    def fit_output(self):

        return self._fit_output

    def precomputation(
        self,
        exposure_integral: collections.OrderedDict = None,
        show_progress: bool = False,
    ):
        """
        Run the necessary precomputation
        :param exposure_integral: instance of ExposureIntegral if already available.
        :param show_progress: set to True if progress bars should be displayed.
        """

        if not exposure_integral:
            for event_type in self._event_types:
                if self._bg:
                    llh = self._sources.background._likelihoods[event_type]
                else:
                    llh = None
                self._exposure_integral[event_type] = ExposureIntegral(
                    self._sources,
                    event_type,
                    self._n_grid_points,
                    show_progress=show_progress,
                    bg_llh=llh,
                )

        else:
            self._exposure_integral = exposure_integral

    def generate_stan_code(self):
        """
        Generate stan code from scratch
        """

        self._fit_filename = self._stan_interface.generate()

    def set_stan_filename(self, fit_filename):
        """
        Set filename of existing stan code
        :param fit_filename: filename of stan code
        """

        self._fit_filename = fit_filename

    def compile_stan_code(self, include_paths=None):
        """
        Compile stan code
        :param include_paths: list of paths to include stan files from
        """

        if not include_paths:
            include_paths = [STAN_PATH, STAN_GEN_PATH]

        self._fit = CmdStanModel(
            stan_file=self._fit_filename,
            stanc_options={"include-paths": include_paths},
            cpp_options={"STAN_THREADS": True},
        )

    def setup_stan_fit(self, filename: Union[str, Path] = ".stan_files/model_code"):
        """
        Create stan model from already compiled file
        :param filename: Path to compiled model file
        """

        self._fit = CmdStanModel(exe_file=filename)

    def run(
        self,
        iterations: int = 1000,
        chains: int = 1,
        seed: int = None,
        show_progress: bool = False,
        threads_per_chain: Union[int, None] = None,
        **kwargs,
    ):
        """
        Run fit
        :param iterations: int, number of MCMC iterations
        :param chains: Number of chains to run in parallel
        :param seed: random seed
        :param show_progress: Set to True if progress par should be displayed
        :param threads_per_chain: When set up using nshards > 1, number of threads to run in parallel per chain
        :param **kwargs: Other kwargs to be passed to cmdstanpy's sampling method
        """
        # Use threads_per_chain = nshards as default
        if not threads_per_chain and self._nshards > 0:
            threads_per_chain = self._nshards

        self._fit_inputs = self._get_fit_inputs()

        self._fit_output = self._fit.sample(
            data=self._fit_inputs,
            iter_sampling=iterations,
            chains=chains,
            seed=seed,
            show_progress=show_progress,
            threads_per_chain=threads_per_chain,
            **kwargs,
        )

    def keys(self):
        if self._reload:
            return self._fit_output.keys()
        else:
            return self._fit_output.stan_variables()

    def __getitem__(self, key):
        """
        Return samples from chains
        :param key: Variable name
        """

        if self._reload:
            data = self._fit_output[key]
        else:
            data = self._fit_output.stan_variable(key)

        shape = (
            self.chains,
            self.iterations,
            int(data.size / self.chains / self.iterations),
        )

        if self._reload:
            return self._fit_output[key].reshape(shape)
        else:
            return self._fit_output.stan_variable(key).reshape(shape)

    def setup_and_run(
        self,
        iterations: int = 1000,
        chains: int = 1,
        seed: int = None,
        show_progress: bool = False,
        include_paths: List[str] = None,
        **kwargs,
    ):
        """
        Run setup and perform fit
        :param iterations: int, number of MCMC iterations
        :param chains: Number of chains to run in parallel
        :param seed: random seed
        :param show_progress: Set to True if progress par should be displayed
        :param threads_per_chain: When set up using nshards > 1, number of threads to run in parallel per chain
        :param **kwargs: Other kwargs to be passed to cmdstanpy's sampling method
        """

        self.precomputation()
        self.generate_stan_code()
        self.compile_stan_code(include_paths=include_paths)
        self.run(
            iterations=iterations,
            chains=chains,
            seed=seed,
            show_progress=show_progress,
            **kwargs,
        )

    def get_src_position(self, source_idx: int = 0):
        """
        Return source position
        :param source_idx: Point source index
        """

        try:
            if self._sources.N == 0:
                raise AttributeError
            try:
                ra = self._sources.point_source[source_idx].ra
                dec = self._sources.point_source[source_idx].dec
            except IndexError:
                raise IndexError(f"Point source index {source_idx} is out of range.")
        except AttributeError:
            try:
                ra, dec = uv_to_icrs(self._fit_inputs["varpi"][source_idx])
            except IndexError:
                raise IndexError(f"Point source index {source_idx} is out of range.")
        source_coords = SkyCoord(ra=ra, dec=dec, frame="icrs")

        return source_coords

    def plot_trace(self, var_names=None, transform: bool = False, **kwargs):
        """
        Trace plot using list of stan parameter keys.
        :param var_names: single parameter name or list of parameters
        :param transform: set to True if log10(x) transformation should be applied
        :param **kwargs: other kwargs passed to arviz.plot_trace
        """

        if not var_names:
            var_names = self._def_var_names
        if transform:
            transform = lambda x: np.log10(x)
            axs = av.plot_trace(
                {key: self[key] for key in var_names}, transform=transform, **kwargs
            )
        else:
            axs = av.plot_trace(
                {key: self[key] for key in var_names}, var_names=var_names, **kwargs
            )
        fig = axs.flatten()[0].get_figure()

        return fig, axs

    def plot_trace_and_priors(self, var_names=None, transform: bool = False, **kwargs):
        """
        Trace plot and overplot the used priors.
        :param var_names: single parameter name or list of parameters
        :param transform: set to True if log10(x) transformation should be applied
        :param **kwargs: other kwargs passed to arviz.plot_trace
        """

        fig, axs = self.plot_trace(
            var_names=var_names, show=False, transform=transform, **kwargs
        )

        if not var_names:
            var_names = self._def_var_names

        priors_dict = self._priors.to_dict()

        def draw_prior_transform(prior, ax, x):
            pdf = prior.pdf_logspace
            ax.plot(
                x,
                pdf(np.power(10, x) * prior.UNITS),
                color="black",
                alpha=0.4,
                zorder=0,
            )

        def draw_prior(prior, ax, x):
            pdf = prior.pdf
            plot = pdf(x * prior.UNITS)
            ax.plot(x, plot, color="black", alpha=0.4, zorder=0)

        for ax_double in axs:
            name = ax_double[0].get_title()
            # check if there is a prior available for the variable
            try:
                # If so, get it and plot it
                if "ang_sys" in name:
                    prior = priors_dict["ang_sys"]
                else:
                    prior = priors_dict[name]
                ax = ax_double[0]
                supp = ax.get_xlim()
                x = np.linspace(*supp, 1000)

                if transform:
                    # Assumes that the only sensible transformation is log10
                    if isinstance(prior, MultiSourcePrior):
                        for p in prior:
                            draw_prior_transform(p, ax, x)
                    else:
                        draw_prior_transform(prior, ax, x)

                else:
                    if isinstance(prior, MultiSourcePrior):
                        for p in prior:
                            draw_prior(p, ax, x)
                    else:
                        draw_prior(prior, ax, x)

                if isinstance(prior, UnitPrior):
                    try:
                        unit = prior.UNITS.unit
                    except AttributeError:
                        unit = prior.UNITS
                    if transform:
                        # yikes
                        title = f"[$\\log_{{10}}\\left (\\frac{{\mathrm{{{name}}}}}{{{unit.to_string('latex_inline').strip('$')}}}\\right )$]"
                    else:
                        title = f"{name} [{unit.to_string('latex_inline')}]"
                    ax.set_title(title)

            except (KeyError, NoPriorSetError):
                pass

        fig = axs.flatten()[0].get_figure()

        return fig, axs

    def _get_kde(
        self,
        var_name,
        index: Union[int, slice, None] = None,
        transform: Callable = lambda x: x,
    ):
        """
        Retrieve kde approximation of samples for given parameter
        :param var_name: parameter name
        :param index: for vector/array parameters, only use this index
        :param transform: Lambda function for transformation of variable
        """

        chain = self[var_name]
        if index is not None:
            data = chain.T[index]
        else:
            data = chain
        return av.kde(transform(data))

    def corner_plot(self, var_names=None, truths=None):
        """
        Corner plot using list of Stan parameter keys and optional
        true values if working with simulated data.
        :param var_names: Variable names for corner plot
        :param truths: If provided, overplot True parameters
        """

        if not var_names:
            var_names = self._def_var_names

        # var_names.pop("Nex")

        # Organise samples
        samples_list = []
        label_list = []

        for key in var_names:
            samples = self[key]
            if not self._reload:
                if len(np.shape(samples)) > 1:
                    # This is for array-like variables, e.g. multiple PS
                    # having their own entry in src_index
                    for samp, src in zip(samples.T, self._sources.point_source):
                        samples_list.append(samp)
                        if key == "L" or key == "src_index":
                            label = "%s_" % src.name + key
                        else:
                            label = key

                        label_list.append(label)

                else:
                    samples_list.append(samples)
                    label_list.append(key)

            else:
                # check for len(np.shape(chain[key]) > 2 because extra dim for chains
                # would be, e.g. for 3 sources, (chains, iter_sampling, 3)
                if len(np.shape(samples)) > 2:
                    for i, src in zip(
                        range(np.shape(samples)[-1]), self._sources.point_source
                    ):
                        if key == "L" or key == "src_index":
                            label = "%s_" % src.name + key
                        else:
                            label = key
                        samples_list.append(
                            samples[:, :, i].reshape(
                                (
                                    self.iter_sampling * self.chains,
                                    1,
                                )
                            )
                        )
                        label_list.append(label)

                else:
                    samples_list.append(
                        samples.reshape(
                            (
                                self.iterations * self.chains,
                                1,
                            )
                        )
                    )
                    label_list.append(key)

        # Organise truths
        if truths:
            truths_list = []

            for key in var_names:
                try:
                    for t in truths[key]:
                        truths_list.append(t)

                except TypeError:
                    truths_list.append(truths[key])

        else:
            truths_list = None

        samples = np.column_stack(samples_list)

        return corner.corner(samples, labels=label_list, truths=truths_list)

    def _plot_energy_posterior(
        self,
        ax,
        center,
        assoc_idx,
        radius,
        color_scale,
        highlight: Union[Iterable, None] = None,
        assoc_threshold: Union[float, None] = 0.2,
        source_name: str = "",
        lw: float = 1.0,
        plot_text: bool = True,
        textsize: float = 8,
    ):
        ev_class = np.array(self._get_event_classifications())
        if radius is not None and center is not None:
            events = self.events
            events.coords.representation_type = "spherical"

            sep = events.coords.separation(center).deg
            mask = sep <= radius.to_value(u.deg)
        else:
            mask = np.ones(ev_class.shape[0], dtype=bool)

        assoc_prob = ev_class[:, assoc_idx][mask]

        # Try to get the true associations from the events
        if highlight is not None:
            highlight = np.atleast_1d(highlight)
            if not highlight.size == mask.size:
                raise ValueError(
                    "highlight must have same length as events inside the selection of the plot"
                )

        if color_scale == "lin":
            norm = colors.Normalize(0.0, 1.0, clip=True)
        elif color_scale == "log":
            norm = colors.LogNorm(1e-8, 1.0, clip=True)
        else:
            raise ValueError("No other scale supported")
        mapper = cm.ScalarMappable(norm=norm, cmap=CMAP)
        color = mapper.to_rgba(assoc_prob)

        indices = np.arange(self._events.N, dtype=int)[mask]

        for c, i in enumerate(indices):
            # get the support (is then log10(E/GeV)) and the pdf values
            supp, pdf = self._get_kde("E", i, lambda x: np.log10(x))
            # exponentiate the support, because we rescale the axis in the end
            ax.plot(
                np.power(10, supp),
                pdf,
                color=color[c],
                zorder=assoc_prob[c] + 1,
                lw=lw,
            )
        _, yhigh = ax.get_ylim()
        ax.set_xscale("log")

        for c, i in enumerate(indices):
            ax.vlines(
                self.events.energies[mask][c].to_value(u.GeV),
                yhigh,
                1.05 * yhigh,
                color=color[c],
                lw=lw * 0.8,
                zorder=assoc_prob[c] + 1,
            )
            if highlight is not None and highlight[i]:
                x, y = self._get_kde("E", i, lambda x: np.log10(x))
                idx_posterior = np.argmax(y)
                ax.plot(
                    [
                        np.power(10, x[idx_posterior]),
                        self.events.energies[mask][c].to_value(u.GeV),
                    ],
                    [y[idx_posterior], yhigh],
                    lw=0.5,
                    color="black",
                    ls="--",
                )

            elif highlight is None and assoc_threshold is not None:
                if assoc_prob[c] >= assoc_threshold:
                    # if we have more than threshold prob, link both lines up
                    x, y = self._get_kde("E", i, lambda x: np.log10(x))
                    idx_posterior = np.argmax(y)
                    ax.plot(
                        [
                            np.power(10, x[idx_posterior]),
                            self.events.energies[mask][c].to_value(u.GeV),
                        ],
                        [y[idx_posterior], yhigh],
                        lw=0.5,
                        color="black",
                        ls="--",
                    )
        if plot_text:
            ax.text(
                1.3e2,
                yhigh * 1.025,
                "$\hat E$",
                fontsize=textsize,
                verticalalignment="center",
            )

        if source_name:
            ax.text(
                0.95,
                0.95,
                source_name,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=textsize,
            )

        ax.set_xlabel(r"$E~[\mathrm{GeV}]$")
        ax.set_ylabel("pdf")
        ax.set_xlim(8e1, 1.4e9)
        return ax, mapper

    def plot_energy_posterior(
        self,
        center: Union[SkyCoord, int, None] = None,
        assoc_idx: int = 0,
        radius: Union[u.Quantity[u.deg], None] = None,
        color_scale: str = "lin",
        highlight: Union[Iterable, None] = None,
        assoc_threshold: Union[float, None] = 0.2,
        source_name: str = "",
        lw: float = 1.0,
        plot_text: bool = True,
        textsize: float = 8,
    ):
        """
        Plot energy posteriors in log10-space.
        Color corresponds to association probability.
        :param center: SkyCoord, int identifying PS or None to center selection on
        :param assoc_idx: integer identifying the source component to calculate assoc prob
        :param radius: if center is not None, select only events within radius around center
        :param color_scale: color scale of assoc prob, either "lin" or "log"
        :param highlight: List of event indices to highlight in plot, defaults to
            all events with association probability larger than `assoc_threshold` to selected source component.
        :param assoc_threshold: If highlight==None, highlight above this association probability.
        """

        fig, ax = plt.subplots(dpi=150)
        if isinstance(center, int):
            center = self.get_src_position(center)
        ax, mapper = self._plot_energy_posterior(
            ax,
            center,
            assoc_idx,
            radius,
            color_scale,
            highlight=highlight,
            assoc_threshold=assoc_threshold,
            source_name=source_name,
            lw=lw,
            plot_text=plot_text,
            textsize=textsize,
        )
        fig.colorbar(mapper, ax=ax, label=f"association probability to {assoc_idx:n}")

        return fig, ax

    def _plot_roi(
        self,
        center,
        ax,
        radius,
        assoc_idx,
        color_scale,
        highlight: Union[Iterable, None] = None,
        source_name: str = "",
        s: float = 30.0,
        textsize: float = 8,
    ):
        ev_class = np.array(self._get_event_classifications())
        assoc_prob = ev_class[:, assoc_idx]

        if highlight is not None:
            highlight = np.atleast_1d(highlight)
            if not highlight.size == assoc_prob.size:
                raise ValueError("highlight must have same length as events")

        min = 0.0
        max = 1.0
        if color_scale == "lin":
            norm = colors.Normalize(min, max, clip=True)
        elif color_scale == "log":
            norm = colors.LogNorm(1e-8, max, clip=True)
        else:
            raise ValueError("No other scale supported")
        mapper = cm.ScalarMappable(norm=norm, cmap=CMAP)
        color = mapper.to_rgba(assoc_prob)

        events = self.events
        events.coords.representation_type = "spherical"
        coords = events.coords

        sep = events.coords.separation(center).deg
        mask = sep <= radius.to_value(u.deg)
        indices = np.arange(self._events.N, dtype=int)[mask]

        ax.scatter(
            center.ra.deg,
            center.dec.deg,
            marker="x",
            color="black",
            zorder=10,
            alpha=0.4,
            transform=ax.get_transform("icrs"),
        )

        for c, i in enumerate(indices):
            edgecolor = "none"
            if highlight is not None:
                if highlight[i]:
                    edgecolor = colors.colorConverter.to_rgba("magenta", alpha=0.5)
                    ax.scatter(
                        coords[i].ra.deg,
                        coords[i].dec.deg,
                        color=color[i],
                        zorder=2.0,
                        transform=ax.get_transform("icrs"),
                        edgecolor=edgecolor,
                        facecolor="none",
                        s=s,
                    )

            ax.scatter(
                coords[i].ra.deg,
                coords[i].dec.deg,
                color=color[i],
                zorder=assoc_prob[i] + 1,
                transform=ax.get_transform("icrs"),
                s=s,
            )

        if source_name:
            ax.text(
                0.05,
                0.95,
                source_name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=textsize,
            )

        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        ax.grid()

        return ax, mapper

    @u.quantity_input
    def plot_roi(
        self,
        center: Union[SkyCoord, int] = 0,
        radius: u.Quantity[u.deg] = 5.0 * u.deg,
        assoc_idx: int = 0,
        color_scale: str = "lin",
        highlight: Union[Iterable, None] = None,
        source_name: str = "",
        s: float = 30.0,
        textsize: float = 8,
    ):
        """
        Create plot of the ROI.
        Events are colour-coded dots, color corresponding
        to the association probability to the point source proposed.
        Assumes there is a point source in self._sources[0].
        Size of events are meaningless.
        :param center: either SkyCoord or PS index to center the plot on
        :param radius: Radius of sky plot
        :param assoc_idx: source idx to calculate the association probability
        :param color_scale: display association probability on "lin" or "log" scale
        :param highlight: Iterable of event indices to highlight in plot.
        """

        if isinstance(center, int):
            center = self.get_src_position(center)

        # we are working in degrees here
        fig, ax = plt.subplots(
            subplot_kw={
                "projection": "astro degrees zoom",
                "center": center,
                "radius": f"{radius.to_value(u.deg)} deg",
            },
            dpi=150,
        )

        ax, mapper = self._plot_roi(
            center,
            ax,
            radius,
            assoc_idx,
            color_scale,
            highlight=highlight,
            source_name=source_name,
            s=s,
            textsize=textsize,
        )
        fig.colorbar(mapper, ax=ax, label=f"association probability to {assoc_idx:n}")

        return fig, ax

    @u.quantity_input
    def plot_energy_and_roi(
        self,
        center: Union[SkyCoord, int] = 0,
        assoc_idx: int = 0,
        radius: u.Quantity[u.deg] = 5 * u.deg,
        color_scale: str = "lin",
        highlight: Union[Iterable, None] = None,
        assoc_threshold: float = 0.2,
        figsize=(8, 3),
        source_name: str = "",
        lw: float = 1,
        s: float = 20.0,
        plot_text: bool = True,
        textsize: float = 8,
    ):
        """
        Create plot of the ROI.
        Events are colour-coded dots, color corresponding
        to the association probability to the point source proposed.
        Assumes there is a point source in self._sources[0].
        Size of events are meaningless.
        :param center: either SkyCoord or PS index to center the plot on
        :param radius: Radius of sky plot
        :param assoc_idx: source idx to calculate the association probability
        :param color_scale: display association probability on "lin" or "log" scale
        :param highlight: Iterable of event indices to highlight in plot.
        :param assoc_threshold: If highlight==None, highlight above this association probability.
        :param figsize: Tuple passed to pyplot.
        """

        fig = plt.figure(dpi=150, figsize=figsize)
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=(0.8, 1.0),
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            wspace=0.13,
            hspace=0.05,
        )

        axs = []

        ax = fig.add_subplot(gs[0, 1])

        if isinstance(center, int):
            center = self.get_src_position(center)

        ax, mapper = self._plot_energy_posterior(
            ax,
            center,
            assoc_idx,
            radius,
            color_scale,
            highlight=highlight,
            assoc_threshold=assoc_threshold,
            lw=lw,
            plot_text=plot_text,
            textsize=textsize,
        )

        ax.set_xlabel(r"$E~[\mathrm{GeV}]$")
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.set_ylabel("posterior pdf")
        axs.append(ax)
        fig.colorbar(mapper, label="association probability", ax=ax)

        ax = fig.add_subplot(
            gs[0, 0],
            projection="astro degrees zoom",
            center=center,
            radius=f"{radius.to_value(u.deg)} deg",
        )

        ax, _ = self._plot_roi(
            center,
            ax,
            radius,
            assoc_idx,
            color_scale,
            highlight=highlight,
            source_name=source_name,
            s=s,
            textsize=textsize,
        )
        axs.insert(0, ax)

        return fig, axs

    def _calculate_flux_grid(self):

        E = np.geomspace(1e2, 1e9, 1_000) << u.GeV

        if not self._sources.point_source:
            raise ValueError("A valid source list is required")

        if not self._reload:
            inputs = self._get_fit_inputs()
            iterations = self._fit_output._iter_sampling
            chains = self._fit_output.chains

        else:
            # Need try-except blocks for case of pgamma
            inputs = self._fit_inputs
            chains = self._fit_meta["chains"]
            iterations = self._fit_meta["iter_sampling"]

        if self._fit_index:
            alpha = self["src_index"]
        elif self._power_law or self._logparabola:
            alpha = inputs["src_index"]
        if self._fit_beta:
            beta = ["beta_index"]
        elif self._logparabola:
            beta = inputs["beta_index"]
        if self._fit_Enorm:
            E0 = self["E0_src"]
        elif self._logparabola:
            E0 = inputs["E0_src"]
        if self._fit_eta:
            eta = self["eta"]
        elif self._seyfert:
            eta = inputs["eta"]
        F = self["F"]

        F = F.reshape((iterations * chains, F.size // (iterations * chains)))
        if self._fit_index:
            N = alpha.size // (iterations * chains)
        elif self._fit_beta:
            N = beta.size // (iterations * chains)
        elif self._fit_Enorm:
            N = E0.size // (iterations * chains)
        elif self._fit_eta:
            N = eta.size // (iterations * chains)

        share_index = N == 1
        N_samples = iterations * chains

        self._flux_grid = (
            np.zeros((len(self._sources.point_source), E.size, N_samples))
            << 1 / u.GeV / u.m**2 / u.s
        )

        for c_ps, ps in enumerate(self._sources.point_source):
            if share_index:
                if self._fit_index:
                    index_vals = alpha.flatten()
                if self._fit_beta:
                    beta_vals = beta.flatten()
                if self._fit_Enorm:
                    E0_vals = E0.flatten()
                if self._fit_eta:
                    eta_vals = eta.flatten()

            else:
                if self._fit_index:
                    index_vals = alpha[:, c_ps].flatten()
                if self._fit_beta:
                    beta_vals = beta[:, c_ps].flatten()
                if self._fit_Enorm:
                    E0_vals = E0[:, c_ps].flatten()
                if self._fit_eta:
                    eta_vals = eta[:, c_ps].flatten()

            if not self._fit_index and not self._pgamma and not self._seyfert:
                index_vals = alpha[c_ps]
                ps.flux_model.spectral_shape.set_parameter("index", index_vals)

            if not self._fit_beta and self._logparabola:
                beta_vals = beta[c_ps]
                ps.flux_model.spectral_shape.set_parameter("beta", beta_vals)

            if not self._fit_Enorm and self._logparabola:
                E0_vals = E0[c_ps]
                ps.flux_model.spectral_shape.set_parameter(
                    "norm_energy", E0_vals * u.GeV
                )
            if not self._fit_eta and self._seyfert:
                eta_vals = eta[c_ps]
                ps.flux_model.spectral_shape.set_parameter("eta", eta_vals)

            flux_int = F[:, c_ps].flatten() << 1 / u.m**2 / u.s

            flux_grid = np.zeros((E.size, N_samples)) << 1 / u.GeV / u.m**2 / u.s

            for c in range(N_samples):
                if self._fit_index:
                    ps.flux_model.spectral_shape.set_parameter("index", index_vals[c])

                if self._fit_beta:
                    ps.flux_model.spectral_shape.set_parameter("beta", beta_vals[c])

                if self._fit_Enorm:
                    ps.flux_model.spectral_shape.set_parameter(
                        "norm_energy", E0_vals[c] * u.GeV
                    )
                if self._fit_eta:
                    ps.flux_model.spectral_shape.set_parameter("eta", eta_vals[c])

                flux = ps.flux_model.spectral_shape(E)  # 1 / GeV / s / m2

                # Needs to be in units used by stan
                int_flux = ps.flux_model.total_flux_int  # 1 / m2 / s

                flux_grid[:, c] = flux / int_flux * flux_int[c]

            self._flux_grid[c_ps] = flux_grid

    def _calculate_quantiles(self, E_power, energy_unit, area_unit, source_idx, LL, UL):

        E = np.geomspace(1e2, 1e9, 1_000) << u.GeV

        lower = np.zeros(E.size)
        upper = np.zeros(E.size)

        flux_grid = (
            self._flux_grid.copy().to_value(1 / energy_unit / area_unit / u.s)
            * np.power(E.to_value(energy_unit), E_power)[:, np.newaxis]
        )

        if source_idx == -1:
            flux_grid = flux_grid.sum(axis=0)
        else:
            flux_grid = flux_grid[source_idx]

        for c in range(E.size):
            lower[c] = np.quantile(flux_grid[c], LL)
            upper[c] = np.quantile(flux_grid[c], UL)

        return lower, upper

    def plot_flux_band(
        self,
        E_power: float = 0.0,
        credible_interval: Union[float, List[float]] = 0.5,
        source_idx: int = 0,
        energy_unit=u.TeV,
        area_unit=u.cm**2,
        x_energy_unit=u.GeV,
        upper_limit: bool = False,
        figsize=(8, 3),
        ax=None,
        **kwargs,
    ):
        """
        Plot flux uncertainties.
        :param E_power: float, plots flux * E**E_power.
        :param credible_interval: set (equal-tailed) credible intervals to be plotted.
        :param source_idx: Choose which point source's flux to plot. -1 for sum over all PS.
        :param energy_unit: Choose your favourite flux energy unit.
        :param area_unit: Choose your favourite flux area unit.
        :param x_energy_unit: Choose your favourite abscissa energy unit
        :param upper_limit: Set to True if only upper limit should be displayed
        :param figsize: Figsize for new figure (requiring `ax=None`)
        :param ax: Reuse existing axis, defaults to creating a new figure with single axis
        :param kwargs: Remaining kwargs will be passed to `pyplot.axis.fill_between` or `pyplot.axis.plot`
        """

        # Have some defaults for plotting
        fill_kwargs = dict(
            alpha=0.3,
            color="C0",
            edgecolor="none",
        )
        limit_kwargs = dict(
            alpha=0.3,
            color="C0",
            marker=r"$\downarrow$",
            markevery=0.06,
            markersize=10,
        )

        fill_kwargs |= kwargs
        limit_kwargs |= kwargs

        # Save some time calculating if the previous calculation has already used the same E_power
        if not hasattr(self, "_flux_grid"):
            self._calculate_flux_grid()

        flux_unit = 1 / energy_unit / area_unit / u.s
        E = np.geomspace(1e2, 1e9, 1_000) << u.GeV

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Find the interval to be plotted
        credible_interval = np.atleast_1d(credible_interval)
        for CI in credible_interval:
            if upper_limit:
                UL = CI
                LL = 0.0  # dummy
            else:
                UL = 0.5 - CI / 2
                LL = 0.5 + CI / 2

            lower, upper = self._calculate_quantiles(
                E_power, energy_unit, area_unit, source_idx, LL, UL
            )

            if not upper_limit:
                ax.fill_between(
                    E.to_value(
                        x_energy_unit,
                        equivalencies=u.spectral(),
                    ),
                    lower,
                    upper,
                    **fill_kwargs,
                )
            else:
                ax.plot(
                    E.to_value(x_energy_unit, equivalencies=u.spectral()),
                    upper,
                    **limit_kwargs,
                    # TODO fix alignment of arrow base to the line
                )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel(f"$E$ [{x_energy_unit.to_string('latex_inline')}]")
        ax.set_ylabel(
            f"flux [{(energy_unit**E_power * flux_unit).unit.to_string('latex_inline')}]"
        )
        if upper_limit:
            ax.set_ylim(bottom=np.min(upper) * 0.8)

        return fig, ax

    def plot_flux_band_HDI(
        self,
        E_power: float = 0.0,
        credible_interval: Union[float, List[float]] = 0.5,
        source_idx: int = 0,
        energy_unit=u.TeV,
        area_unit=u.cm**2,
        x_energy_unit=u.GeV,
        upper_limit: bool = False,
        figsize=(8, 3),
        ax=None,
        **kwargs,
    ):
        """
        Plot flux uncertainties.
        :param E_power: float, plots flux * E**E_power.
        :param credible_interval: set (equal-tailed) credible intervals to be plotted.
        :param source_idx: Choose which point source's flux to plot. -1 for sum over all PS.
        :param energy_unit: Choose your favourite flux energy unit.
        :param area_unit: Choose your favourite flux area unit.
        :param x_energy_unit: Choose your favourite abscissa energy unit
        :param upper_limit: Set to True if only upper limit should be displayed
        :param figsize: Figsize for new figure (requiring `ax=None`)
        :param ax: Reuse existing axis, defaults to creating a new figure with single axis
        :param kwargs: Remaining kwargs will be passed to `pyplot.axis.fill_between` or `pyplot.axis.plot`
        """

        # Have some defaults for plotting
        fill_kwargs = dict(
            alpha=0.3,
            color="C0",
            edgecolor="none",
        )
        limit_kwargs = dict(
            alpha=0.3,
            color="C0",
            marker=r"$\downarrow$",
            markevery=0.06,
            markersize=10,
        )

        fill_kwargs |= kwargs
        limit_kwargs |= kwargs

        # Save some time calculating if the previous calculation has already used the same E_power
        if not hasattr(self, "_flux_grid"):
            self._calculate_flux_grid()

        flux_unit = 1 / energy_unit / area_unit / u.s
        E = np.geomspace(1e2, 1e9, 1_000) << u.GeV

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Find the interval to be plotted
        credible_interval = np.atleast_1d(credible_interval)
        for CI in credible_interval:
            if upper_limit:
                UL = CI
                LL = 0.0  # dummy
            else:
                UL = 0.5 - CI / 2
                LL = 0.5 + CI / 2

            lower, upper = av.hdi(self._flux_grid)
            # self._calculate_quantiles(
            #    E_power, energy_unit, area_unit, source_idx, LL, UL
            # )
            if not upper_limit:
                ax.fill_between(
                    E.to_value(
                        x_energy_unit,
                        equivalencies=u.spectral(),
                    ),
                    lower,
                    upper,
                    **fill_kwargs,
                )
            else:
                ax.plot(
                    E.to_value(x_energy_unit, equivalencies=u.spectral()),
                    upper,
                    **limit_kwargs,
                    # TODO fix alignment of arrow base to the line
                )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel(f"$E$ [{x_energy_unit.to_string('latex_inline')}]")
        ax.set_ylabel(
            f"flux [{(energy_unit**E_power * flux_unit).unit.to_string('latex_inline')}]"
        )
        if upper_limit:
            ax.set_ylim(bottom=np.min(upper) * 0.8)

        return fig, ax

    def plot_peak_energy_flux(
        self,
        ax,
        levels=[0.5, 0.683, 0.95],
        energy_unit=u.TeV,
        area_unit=u.cm**2,
        x_energy_unit=u.GeV,
        **kwargs,
    ):
        """
        Plot 2d kde contours of peak energy flux and energy at which peak lies
        :param ax: Axis in which to plot
        :param levels: HDI levels to plot
        :param energy_unit: flux energy unit, i.e. energy_unit / area_unit / s
        :param area_unit: flux area unit, i.e. energy_unit / area_unit / s
        :param x_energy_unit: energy unit of x-axis
        """

        from matplotlib.lines import Line2D
        import seaborn as sns

        handles, labels = ax.get_legend_handles_labels()

        levels = np.sort(1 - np.atleast_1d(levels))
        try:
            E_peak = self._fit_output["E_peak"].squeeze() << u.GeV
            peak_flux = self._fit_output["peak_energy_flux"].squeeze() << u.GeV / u.m**2
        except:  # TODO find proper exception
            E_peak = self._fit_output.stan_variable("E_peak").squeeze() << u.GeV
            peak_flux = (
                self._fit_output.stan_varable("peak_energy_flux") << u.GeV / u.m**2
            )
        mask = peak_flux.value > 0.0
        data = {
            "E": E_peak.to_value(x_energy_unit, equivalencies=u.spectral())[mask],
            "flux": peak_flux.to_value(energy_unit / area_unit)[mask],
        }

        sns.kdeplot(data, x="E", y="flux", ax=ax, levels=levels, cmap=CMAP, **kwargs)

        colours = ax.collections[-1]._mapped_colors

        tex = plt.rcParams["text.usetex"]
        for c, l in zip(colours, levels):
            handles.append(Line2D([0], [0], color=c))
            if tex:
                label = rf"{int((1-l)*100):d}\% CR"
            else:
                label = rf"{int((1-l)*100):d}% CR"
            labels.append(label)
        ax.legend(handles, labels)

        # this is an effing mess
        legend = ax.get_legend()
        renderer = plt.gcf().canvas.get_renderer()
        extends = [t.get_window_extent(renderer).width for t in legend.get_texts()]
        max_extend = max(extends)
        for t, e in zip(legend.get_texts(), extends):
            t.set_position((max_extend - e, 0))

    def save(
        self,
        path: Path,
        overwrite: bool = False,
        save_json: bool = False,
        use_timestamp: bool = False,
        save_warmup: bool = False,
    ):
        """
        Save fit to h5 file.
        :param path: Path to which fit is saved.
        :param overwrite: Set to `True` to overwrite existing file,
            else timestamp is appended to `path` to avoid overwriting.
        param save_json: Set to `True` if arviz json output should be saved.
            uses provided path with .json extension.
        """

        # Check if filename consists of a path to some directory as well as the filename
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        if dirname:
            if not os.path.exists(dirname):
                logger.warning(
                    f"{dirname} does not exist, saving instead to {os.getcwd()}"
                )
                dirname = os.getcwd()
        else:
            dirname = os.getcwd()
        path = Path(dirname) / Path(filename)

        if (os.path.exists(path) and not overwrite) or use_timestamp:
            if os.path.exists(path):
                logger.warning(f"File {filename} already exists.")
            file = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            file += f"_{int(thyme())}"
            filename = file + ext

        path = Path(dirname) / Path(filename)

        with h5py.File(path, "w") as f:

            fit_folder = f.create_group("fit")
            inputs_folder = fit_folder.create_group("inputs")
            outputs_folder = fit_folder.create_group("outputs")
            meta_folder = fit_folder.create_group("meta")
            source_folder = f.create_group("sources")

            # Create config from sources
            config = HierarchicalNuConfig.make_config(self.sources)
            config_string = OmegaConf.to_yaml(config)
            source_folder.create_dataset("config", data=config_string)

            for key, value in self._fit_inputs.items():
                inputs_folder.create_dataset(key, data=value)

            # All quantities with event_types dependency follow this list's order
            inputs_folder.create_dataset(
                "event_types", data=[_.S for _ in self._event_types]
            )
            try:
                Emin_det = Parameter.get_parameter("Emin_det")
                inputs_folder.create_dataset(
                    "Emin_det", data=Emin_det.value.to_value(u.GeV)
                )
            except ValueError:
                for dm in self._event_types:
                    Emin_det = Parameter.get_parameter(f"Emin_det_{dm.P}")
                    inputs_folder.create_dataset(
                        f"Emin_det_{dm.P}", data=Emin_det.value.to_value(u.GeV)
                    )

            for key, value in self._fit_output.stan_variables().items():
                if save_warmup:
                    value = self._fit_output.stan_variable(key, inc_warmup=True)
                if key == "irf_return":
                    n_entries = len(value[0])
                    for n in range(n_entries):
                        if len(value[0][n].shape) >= 1:
                            out = np.concatenate([_[n] for _ in value])
                        else:
                            out = np.vstack([_[n] for _ in value])
                        outputs_folder.create_dataset(key + f".{n}", data=out)
                    continue
                outputs_folder.create_dataset(key, data=value)

            # Save some metadata for debugging, easier loading from file
            if np.any(self._fit_output.divergences):
                divergences = self._fit_output.method_variables()["divergent__"]
                meta_folder.create_dataset(
                    "divergences", data=np.argwhere(divergences == 1.0)
                )
            meta_folder.create_dataset("chains", data=self._fit_output.chains)
            meta_folder.create_dataset(
                "iter_sampling", data=self._fit_output._iter_sampling
            )
            meta_folder.create_dataset(
                "iter_warmup", data=self._fit_output._iter_warmup
            )
            meta_folder.create_dataset("save_warmup", data=int(save_warmup))
            meta_folder.create_dataset("runset", data=str(self._fit_output.runset))
            meta_folder.create_dataset("diagnose", data=self._fit_output.diagnose())
            f.create_dataset("version", data=git_hash)

            summary = self._fit_output.summary()
            meta = self._fit_output.method_variables()

            method_keys = [
                "lp__",
                "stepsize__",
                "treedepth__",
                "n_leapfrog__",
            ]

            # List of keys for which we are looking in the entirety of stan parameters
            key_stubs = [
                "L",
                "_luminosity",
                "src_index",
                "_src_index",
                "E[",
                "Esrc[",
                "F_atmo",
                "diffuse_norm",
                "F_diff",
                "diff_index",
                "Nex",
                "f_det",
                "f_arr",
                "logF",
                "Ftot",
                "Fs",
            ]

            keys = []
            for k, v in summary["R_hat"].items():
                for key in key_stubs:
                    if key in k:
                        keys.append(k)
                        break

            R_hat = np.array([summary["R_hat"][k] for k in keys])
            if "ESS_bulk" in summary.keys():
                ESS_bulk = np.array([summary["ESS_bulk"][k] for k in keys])
                ESS_tail = np.array([summary["ESS_tail"][k] for k in keys])
                meta_folder.create_dataset("ESS_bulk", data=ESS_bulk)
                meta_folder.create_dataset("ESS_tail", data=ESS_tail)
            if "N_Eff" in summary.keys():
                N_Eff = np.array([summary["N_Eff"][k] for k in keys])
                meta_folder.create_dataset("N_Eff", data=N_Eff)

            meta_folder.create_dataset("R_hat", data=R_hat)
            meta_folder.create_dataset("parameters", data=np.array(keys, dtype="S"))

            for key in method_keys:
                meta_folder.create_dataset(key, data=meta[key])

        self.events.to_file(path, append=True)

        # Add priors separately
        self.priors.addto(path, "priors")

        if save_json:
            df = av.from_cmdstanpy(self._fit_output)
            json_path = Path(dirname) / Path(os.path.splitext(filename)[0] + ".json")
            df.to_json(json_path)

        return path  # noqa: F821

    def diagnose(self):
        """
        Print fit diagnosis
        """

        try:
            print(self._fit_output.diagnose())
        except:
            print(self._fit_meta["diagnose"].decode("ascii"))

    def save_csvfiles(self, directory):
        """
        Save cmdstanpy csv files
        :param directory: Directory to save csv files to.
        """

        self._fit_output.save_csvfiles(directory)

    @classmethod
    def from_file(cls, *filename, load_warmup: bool = False):
        """
        Load fit output from file. Allows to
        make plots and run classification check.
        :param filename: single or multiple filenames to be loaded.
        """

        if len(filename) == 1:

            (
                event_types,
                events,
                obs_time_dict,
                priors,
                fit_inputs,
                outputs,
                meta,
                config,
            ) = cls._from_file(filename[0], load_warmup=load_warmup)

            temp = OmegaConf.create(config)
            default = HierarchicalNuConfig.load_default()
            merged = OmegaConf.merge(default, temp)
            config_parser = ConfigParser(merged)
            sources = config_parser.sources
            fit = cls(sources, event_types, events, obs_time_dict, priors, reload=True)

        else:
            outputs = {}
            meta = {}
            fit_outputs = []
            fit_meta = []
            configs = []
            for file in filename:
                (
                    event_types,
                    events,
                    obs_time_dict,
                    priors,
                    fit_inputs,
                    outputs,
                    meta,
                    config,
                ) = cls._from_file(file, load_warmup=load_warmup)
                fit_outputs.append(outputs)
                fit_meta.append(meta)
                if configs:
                    # Check that all source configurations are the same
                    if not configs[-1] == config:
                        raise ValueError("Cannot stack fits of different configs")
                configs.append(config)

            keys = fit_outputs[0].keys()
            for key in keys:
                outputs[key] = np.vstack([_[key] for _ in fit_outputs])
                # Some assert statement for the correct stacking?

            keys = fit_meta[0].keys()
            for key in keys:
                if key == "parameters":
                    meta[key] = fit_meta[0][key]
                elif key == "iter_sampling":
                    if not np.unique(np.array([_[key] for _ in fit_meta])).size == 1:
                        raise ValueError(
                            "Cannot stack fits of different sampling length"
                        )
                    meta[key] = fit_meta[0][key]
                elif key == "chains":
                    if not np.unique(np.array([_[key] for _ in fit_meta])).size == 1:
                        raise ValueError(
                            "Cannot stack fits of different sampling length"
                        )
                    meta[key] = np.sum([_[key] for _ in fit_meta])
                else:
                    meta[key] = np.vstack([_[key] for _ in fit_meta])

            config_parser = ConfigParser(OmegaConf.create(configs[-1]))
            sources = config_parser.sources
            fit = cls(sources, event_types, events, obs_time_dict, priors, reload=True)

        fit._fit_output = outputs
        fit._fit_inputs = fit_inputs
        fit._fit_meta = meta

        fit._def_var_names = []

        if sources.point_source:
            if "L" in fit.keys():
                fit._def_var_names.append("L")
            else:
                fit._def_var_names.append("L_ind")
            fit._def_var_names.append("Nex_src")

        if "src_index_grid" in fit_inputs.keys():
            fit._def_var_names.append("src_index")

        if "beta_index_grid" in fit_inputs.keys():
            fit._def_var_names.append("beta_index")

        if "E0_src_grid" in fit_inputs.keys():
            fit._def_var_names.append("E0_src")

        if "eta_grid" in fit_inputs.keys():
            fit._def_var_names.append("eta")
            if "pressure_ratio" in fit.keys():
                fit._def_var_names.append("pressure_ratio")
            else:
                fit._def_var_names.append("pressure_ratio_ind")

        if sources.diffuse:
            fit._def_var_names.append("diffuse_norm")
            fit._def_var_names.append("diff_index")

        if sources.atmospheric:
            fit._def_var_names.append("F_atmo")

        if "ang_sys_deg" in outputs.keys():
            fit._def_var_names.append("ang_sys_deg")

        return fit

    @staticmethod
    def _from_file(filename, load_warmup: bool = False):

        fit_inputs = {}
        fit_outputs = {}
        fit_meta = {}

        with h5py.File(filename, "r") as f:
            if "fit" not in f.keys():
                raise ValueError("File is not a saved hierarchical_nu fit.")

            try:
                for k, v in f["fit/meta"].items():
                    fit_meta[k] = v[()]
            except KeyError:
                fit_meta["chains"] = 1
                fit_meta["iter_sampling"] = 1000
                fit_meta["iter_warmup"] = 1000

            if "save_warmup" not in fit_meta:
                # Backwards compatibility
                fit_meta["save_warmup"] = False
                save_warmup = False
            else:
                save_warmup = bool(fit_meta["save_warmup"])

            for k, v in f["fit/inputs"].items():

                fit_inputs[k] = v[()]

            for k, v in f["fit/outputs"].items():
                # Add extra dimension for number of chains
                if k == "local_pars" or k == "global_pars" or "irf_return" in k:
                    continue
                temp = v[()]
                if len(temp.shape) == 1:
                    # non-vector variable

                    if save_warmup:
                        shape = (
                            fit_meta["chains"],
                            fit_meta["iter_warmup"] + fit_meta["iter_sampling"],
                        )
                    else:
                        shape = (fit_meta["chains"], fit_meta["iter_sampling"])
                    fit_outputs[k] = temp.reshape(shape)
                    if save_warmup and not load_warmup:
                        fit_outputs[k] = fit_outputs[k][..., fit_meta["iter_warmup"] :]

                else:
                    # Reshape to chains x draws x dim
                    if save_warmup:
                        shape = (
                            fit_meta["chains"],
                            fit_meta["iter_warmup"] + fit_meta["iter_sampling"],
                            *temp.shape[1:],
                        )
                    else:
                        shape = (
                            fit_meta["chains"],
                            fit_meta["iter_sampling"],
                            *temp.shape[1:],
                        )
                    fit_outputs[k] = temp.reshape(shape)
                    if save_warmup and not load_warmup:
                        fit_outputs[k] = fit_outputs[k][
                            :, fit_meta["iter_warmup"] :, ...
                        ]

            # requires config parser, which in turn imports fit, which in turn imports config parser...
            config = f["sources/config"][()].decode("ascii")

        event_types = [
            Refrigerator.stan2dm(_) for _ in fit_inputs["event_types"].tolist()
        ]

        obs_time = fit_inputs["T"] * u.s

        obs_time_dict = {et: obs_time[k] for k, et in enumerate(event_types)}

        # try:
        priors = Priors.from_group(filename, "priors")
        #except KeyError:
        #    # lazy fix for backwards compatibility
        #    priors = Priors()

        events = Events.from_file(
            filename,
            apply_Emin_det=False,
            apply_spatial_cuts=False,
            apply_temporal_cuts=False,
        )

        try:
            Emin_det = fit_inputs["Emin_det"]
            mask = events.energies >= Emin_det * u.GeV
            events.select(mask)
        except KeyError:
            mask = np.full(events.N, True)
            for dm in event_types:
                try:
                    Emin_det = fit_inputs[f"Emin_det_{dm.P}"]
                    mask[events.types == dm.S] = (
                        events.energies[events.types == dm.S] >= Emin_det * u.GeV
                    )
                except KeyError:
                    # backwards compatibility
                    pass
            events.select(mask)

        return (
            event_types,
            events,
            obs_time_dict,
            priors,
            fit_inputs,
            fit_outputs,
            fit_meta,
            config,
        )

    def check_classification(self, sim_outputs):
        """
        For the case of simulated data, check if
        events are correctly classified into the
        different source categories.
        :param sim_outputs: True associations of events, using `Lambda` of simulation.
        """

        Ns = len([s for s in self._sources.sources if isinstance(s, PointSource)])

        event_labels = sim_outputs["Lambda"] - 1

        prob_each_src = self._get_event_classifications()

        source_labels = ["src%i" % src for src in range(Ns)]

        if self._sources.atmospheric and self._sources.diffuse:
            source_labels.append("diff")
            source_labels.append("atmo")

        elif self._sources.diffuse:
            source_labels.append("diff")

        elif self._sources.atmospheric:
            source_labels.append("atmo")

        wrong = []
        assumed = []
        correct = []

        for i in range(len(prob_each_src)):
            classified = np.where(prob_each_src[i] == np.max(prob_each_src[i]))[0][
                0
            ] == int(event_labels[i])

            if not classified:
                wrong.append(i)

                print("Event %i is misclassified" % i)

                for src in range(Ns):
                    print("P(src%i) = %.6f" % (src, prob_each_src[i][src]))

                if self._sources.atmospheric and self._sources.diffuse:
                    print("P(diff) = %.6f" % prob_each_src[i][Ns])
                    print("P(atmo) = %.6f" % prob_each_src[i][Ns + 1])

                elif self._sources.diffuse:
                    print("P(diff) = %.6f" % prob_each_src[i][Ns])

                elif self._sources.atmospheric:
                    print("P(atmo) = %.6f" % prob_each_src[i][Ns])

                print("The correct component is", source_labels[int(event_labels[i])])
                correct.append(source_labels[int(event_labels[i])])
                assumed.append(source_labels[np.argmax(prob_each_src[i])])

        if not wrong:
            print("All events are correctly classified")
        else:
            print(
                "A total of %i events out of %i are misclassified"
                % (len(wrong), len(event_labels))
            )

        return wrong, assumed, correct

    @property
    def chains(self):
        """
        Return number of chains
        """

        if self._reload:
            return self._fit_meta["chains"]
        else:
            return self._fit_output.chains

    @property
    def iterations(self):
        """
        Return number of iterations per chain
        """

        if self._reload:
            return self._fit_meta["iter_sampling"]
        else:
            return self._fit_output.num_draws_sampling

    def _get_event_association_dist(self):
        # logprob (lp) is a misnomer, this is actually the rate parameter of each source component
        if not self._reload:
            logprob = self._fit_output.stan_variable("lp").transpose(1, 2, 0)
        else:
            # We are in a reloaded state and _fit_output is a dictionary
            # also the shape is "wrong" for transpose()
            logprob = (
                self._fit_output["lp"]
                .reshape(
                    (
                        self._fit_meta["chains"] * self._fit_meta["iter_sampling"],
                        *self._fit_output["lp"].shape[2:],
                    )
                )
                .transpose(1, 2, 0)
            )

        # the sum normalises to all source components
        ratio = np.exp(logprob) / np.sum(np.exp(logprob), axis=1)[:, np.newaxis, :]
        # axes are now event, component, sample
        return ratio

    def _get_event_classifications(self):
        """
        Get list of event classifications
        """

        ratio = self._get_event_association_dist()
        # average over samples, hence axis=-1
        assoc_prob = np.average(ratio, axis=-1).tolist()
        return assoc_prob

    def _get_fit_inputs(self):
        """
        Return dictionary of fit inputs, passed to cmdstanpy
        """

        self._get_par_ranges()
        fit_inputs = {}
        if self._events.N == 0:
            raise ValueError("Cannot perform fits with zero events")
        fit_inputs["N"] = self._events.N
        # Number of shards and max. events per shards only used if multithreading is desired
        fit_inputs["N_shards"] = self._nshards if self._nshards > 0 else 1
        try:
            fit_inputs["J"] = ceil(fit_inputs["N"] / fit_inputs["N_shards"])
        except ZeroDivisionError:
            fit_inputs["J"] = fit_inputs["N"]
        fit_inputs["Ns_tot"] = len([s for s in self._sources.sources])
        fit_inputs["Edet"] = self._events.energies.to_value(u.GeV)
        fit_inputs["omega_det"] = self._events.unit_vectors
        fit_inputs["omega_det"] = [
            (_ / np.linalg.norm(_)).tolist() for _ in fit_inputs["omega_det"]
        ]
        fit_inputs["event_type"] = self._events.types
        fit_inputs["kappa"] = self._events.kappas
        if self._ang_sys and not self._fit_ang_sys:
            fit_inputs["ang_err"] = np.sqrt(
                Parameter.get_parameter("ang_sys_add").value.to_value(u.rad) ** 2
                + self._events.ang_errs.to_value(u.rad) ** 2
            )
        elif self._fit_ang_sys:
            fit_inputs["ang_sys_min"], fit_inputs["ang_sys_max"] = (
                self._ang_sys_par_range
            )
            fit_inputs["ang_err"] = self._events.ang_errs.to_value(u.rad)
        else:
            fit_inputs["ang_err"] = self._events.ang_errs.to_value(u.rad)

        fit_inputs["Ns"] = len(
            [s for s in self._sources.sources if isinstance(s, PointSource)]
        )

        redshift = [
            s.redshift
            for s in self._sources.sources
            if isinstance(s, PointSource)
            or isinstance(s.flux_model, IsotropicDiffuseBG)
        ]
        D = [
            luminosity_distance(s.redshift).value
            for s in self._sources.sources
            if isinstance(s, PointSource)
        ]
        src_pos = [
            icrs_to_uv(s.dec.to_value(u.rad), s.ra.to_value(u.rad))
            for s in self._sources.sources
            if isinstance(s, PointSource)
        ]

        fit_inputs["z"] = redshift
        fit_inputs["D"] = D
        fit_inputs["varpi"] = src_pos

        fit_inputs["Emin"] = Parameter.get_parameter("Emin").value.to_value(u.GeV)
        fit_inputs["Emax"] = Parameter.get_parameter("Emax").value.to_value(u.GeV)

        if fit_inputs["Emin"] < 1e2:
            raise ValueError("Emin is lower than detector minimum energy")
        if fit_inputs["Emax"] > 1e9:
            raise ValueError("Emax is higher than detector maximum energy")

        if self._sources.point_source:
            fit_inputs["Emin_src"] = [
                ps.frame.transform(
                    Parameter.get_parameter("Emin_src").value, ps.redshift
                ).to_value(u.GeV)
                for ps in self._sources.point_source
            ]
            fit_inputs["Emax_src"] = [
                ps.frame.transform(
                    Parameter.get_parameter("Emax_src").value, ps.redshift
                ).to_value(u.GeV)
                for ps in self._sources.point_source
            ]

            if np.min(fit_inputs["Emin_src"]) < fit_inputs["Emin"]:
                raise ValueError(
                    "Minimum source energy may not be lower than minimum energy overall"
                )
            if np.max(fit_inputs["Emax_src"]) > fit_inputs["Emax"]:
                raise ValueError(
                    "Maximum source energy may not be higher than maximum energy overall"
                )

        if self._sources.diffuse:
            fit_inputs["Emin_diff"] = self._sources.diffuse.frame.transform(
                Parameter.get_parameter("Emin_diff").value,
                self._sources.diffuse.redshift,
            ).to_value(u.GeV)
            fit_inputs["Emax_diff"] = self._sources.diffuse.frame.transform(
                Parameter.get_parameter("Emax_diff").value,
                self._sources.diffuse.redshift,
            ).to_value(u.GeV)
            fit_inputs["Enorm_diff"] = (
                self._sources.diffuse.flux_model.spectral_shape._normalisation_energy.to_value(
                    u.GeV
                )
            )

            if fit_inputs["Emin_diff"] < fit_inputs["Emin"]:
                raise ValueError(
                    "Minimum diffuse energy may not be lower than minimum energy overall"
                )
            if fit_inputs["Emax_diff"] > fit_inputs["Emax"]:
                raise ValueError(
                    "Maximum diffuse energy may not be higher than maximum energy overall"
                )

        integral_grid = []
        integral_grid_2d = []
        atmo_integ_val = []
        obs_time = []

        for c, event_type in enumerate(self._event_types):
            obs_time.append(self._observation_time[event_type].to_value(u.s))

        fit_inputs["Ngrid"] = self._exposure_integral[event_type]._n_grid_points

        if self._use_event_tag:
            fit_inputs["event_tag"] = (
                np.array(self._events.get_tags(self._sources)).astype(int) + 1
            )

        if self._sources.point_source:
            # Check for shared source index
            if self._shared_src_index:
                key = "src_index"
                key_beta = "beta_index"
                key_Enorm = "E0_src"
                key_eta = "eta"

            # Otherwise just use first source in the list
            # src_index_grid is identical for all point sources
            else:
                key = "%s_src_index" % self._sources.point_source[0].name
                key_beta = "%s_beta_index" % self._sources.point_source[0].name
                key_Enorm = "%s_E0_src" % self._sources.point_source[0].name
                key_eta = "%s_eta" % self._sources.point_source[0].name

            if self._fit_index:
                fit_inputs["src_index_grid"] = self._exposure_integral[
                    event_type
                ].par_grids[key]
                # PS parameter limits
                fit_inputs["src_index_min"] = self._src_index_par_range[0]
                fit_inputs["src_index_max"] = self._src_index_par_range[1]
            elif self._logparabola:
                fit_inputs["src_index"] = [
                    ps.flux_model.parameters["index"].value
                    for ps in self._sources.point_source
                ]

            if self._fit_nex:
                fit_inputs["Nex_src_min"] = self._nex_par_range[0]
                fit_inputs["Nex_src_max"] = self._nex_par_range[1]

            try:
                fit_inputs["Lmin"] = self._lumi_par_range[0]
                fit_inputs["Lmax"] = self._lumi_par_range[1]
            except AttributeError:
                pass

            if self._fit_beta:
                fit_inputs["beta_index_grid"] = self._exposure_integral[
                    event_type
                ].par_grids[key_beta]
                fit_inputs["beta_index_min"] = self._beta_index_par_range[0]
                fit_inputs["beta_index_max"] = self._beta_index_par_range[1]
                fit_inputs["beta_index_mu"] = self._priors.beta_index.mu
                fit_inputs["beta_index_sigma"] = self._priors.beta_index.sigma
            elif self._logparabola:
                fit_inputs["beta_index"] = [
                    ps.flux_model.parameters["beta"].value
                    for ps in self._sources.point_source
                ]

            if self._fit_Enorm:
                fit_inputs["E0_src_grid"] = self._exposure_integral[
                    event_type
                ].par_grids[key_Enorm]
                fit_inputs["E0_src_min"] = self._E0_src_par_range[0].to_value(u.GeV)
                fit_inputs["E0_src_max"] = self._E0_src_par_range[1].to_value(u.GeV)
                if self._priors.E0_src.name == "lognormal":
                    fit_inputs["E0_src_mu"] = self._priors.E0_src.mu
                    fit_inputs["E0_src_sigma"] = self._priors.E0_src.sigma
                elif self._priors.E0_src.name == "normal":
                    fit_inputs["E0_src_mu"] = self._priors.E0_src.mu.to_value(u.GeV)
                    fit_inputs["E0_src_sigma"] = self._priors.E0_src.sigma.to_value(
                        u.GeV
                    )
            elif self._logparabola:
                fit_inputs["E0"] = [
                    ps.flux_model.parameters["norm_energy"].value.to_value(u.GeV)
                    for ps in self._sources.point_source
                ]

            if self._fit_eta:
                fit_inputs["eta_grid"] = self._exposure_integral[event_type].par_grids[
                    key_eta
                ]
                fit_inputs["eta_min"], fit_inputs["eta_max"] = self._eta_par_range
                fit_inputs["P_min"], fit_inputs["P_max"] = self._P_par_range
            if self._fit_ang_sys:
                fit_inputs["ang_sys_mu"] = self._priors.ang_sys.mu.to_value(u.rad)
                fit_inputs["ang_sys_sigma"] = self._priors.ang_sys.sigma.to_value(u.rad)
                if self._priors.ang_sys.name == "exponnorm":
                    fit_inputs["ang_sys_lam"] = self._priors.ang_sys.lam.to_value(
                        1 / u.rad
                    )

        # Inputs for priors of point sources
        if self._priors.src_index.name in ["normal", "lognormal"]:
            fit_inputs["src_index_mu"] = self._priors.src_index.mu
            fit_inputs["src_index_sigma"] = self._priors.src_index.sigma

        if self._priors.luminosity.name == "lognormal":
            fit_inputs["lumi_mu"] = self._priors.luminosity.mu
            fit_inputs["lumi_sigma"] = self._priors.luminosity.sigma
        elif self._priors.luminosity.name == "normal":
            fit_inputs["lumi_mu"] = self._priors.luminosity.mu.to_value(
                self._priors.luminosity.UNITS
            )
            fit_inputs["lumi_sigma"] = self._priors.luminosity.sigma.to_value(
                self._priors.luminosity.UNITS
            )
        elif self._priors.luminosity.name == "pareto":
            fit_inputs["lumi_xmin"] = self._priors.luminosity.xmin.to_value(
                self._priors.luminosity.UNITS
            )
            fit_inputs["lumi_alpha"] = self._priors.luminosity.alpha
        else:
            raise ValueError("No other prior type for luminosity implemented.")

        if self._priors.pressure_ratio.name in ["normal", "lognormal"]:
            fit_inputs["P_mu"] = self._priors.pressure_ratio.mu
            fit_inputs["P_sigma"] = self._priors.pressure_ratio.sigma

        if self._priors.Nex_src.name in ["normal", "lognormal"]:
            fit_inputs["Nex_mu"] = self._priors.Nex_src.mu
            fit_inputs["Nex_sigma"] = self._priors.Nex_src.sigma

        if self._priors.eta.name in ["normal", "lognormal"]:
            fit_inputs["eta_mu"] = self._priors.eta.mu
            fit_inputs["eta_sigma"] = self._priors.eta.sigma

        if self._sources.diffuse:
            # Just take any for now, using default parameters it doesn't matter
            fit_inputs["diff_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids["diff_index"]

            fit_inputs["diff_index_min"] = self._diff_index_par_range[0]
            fit_inputs["diff_index_max"] = self._diff_index_par_range[1]
            fit_inputs["diffuse_norm_min"] = self._diffuse_norm_par_range[0]
            fit_inputs["diffuse_norm_max"] = self._diffuse_norm_par_range[1]

            # Priors for diffuse model
            if self._priors.diffuse_flux.name == "normal":
                fit_inputs["f_diff_mu"] = self._priors.diffuse_flux.mu.to_value(
                    self._priors.diffuse_flux.UNITS
                )
                fit_inputs["f_diff_sigma"] = self._priors.diffuse_flux.sigma.to_value(
                    self._priors.diffuse_flux.UNITS
                )
            elif self._priors.diffuse_flux.name == "lognormal":
                fit_inputs["f_diff_mu"] = self._priors.diffuse_flux.mu
                fit_inputs["f_diff_sigma"] = self._priors.diffuse_flux.sigma
            else:
                raise ValueError(
                    "No other type of prior for diffuse index implemented."
                )
            fit_inputs["diff_index_mu"] = self._priors.diff_index.mu
            fit_inputs["diff_index_sigma"] = self._priors.diff_index.sigma

        if self._sources.atmospheric:
            fit_inputs["atmo_integrated_flux"] = (
                self._sources.atmospheric.flux_model.total_flux_int.to_value(
                    1 / (u.m**2 * u.s)
                )
            )

            fit_inputs["F_atmo_min"] = self._F_atmo_par_range[0]
            fit_inputs["F_atmo_max"] = self._F_atmo_par_range[1]

            # Priors for atmo model
            if self._priors.atmospheric_flux.name == "lognormal":
                fit_inputs["f_atmo_mu"] = self._priors.atmospheric_flux.mu
                fit_inputs["f_atmo_sigma"] = self._priors.atmospheric_flux.sigma
            elif self._priors.atmospheric_flux.name == "normal":
                fit_inputs["f_atmo_mu"] = self._priors.atmospheric_flux.mu.to_value(
                    self._priors.atmospheric_flux.UNITS
                )
                fit_inputs["f_atmo_sigma"] = (
                    self._priors.atmospheric_flux.sigma.to_value(
                        self._priors.atmospheric_flux.UNITS
                    )
                )
            else:
                raise ValueError(
                    "No other prior type for atmospheric flux implemented."
                )
        if self._sources.background:
            fit_inputs["bg_llh"] = np.zeros(self.events.N)
            time = LifeTime()

            for dm in self._event_types:
                N_dm = Events.from_ev_file(
                    dm,
                    apply_Emin_det=False,
                    apply_spatial_cuts=False,
                    apply_temporal_cuts=False,
                ).N

                time_norm = time.lifetime_from_dm(dm)[dm].to_value(u.s)

                decs = self.events.coords[dm.S == self.events.types].dec.to_value(u.rad)
                sindecs = np.sin(decs)
                ereco = np.log10(
                    self.events.energies[dm.S == self.events.types].to_value(u.GeV)
                )
                prob_ereco_and_omega = self.sources.background._likelihoods[
                    dm
                ].prob_ereco_and_omega(ereco, sindecs)

                # normalisation of flat logx distribution
                # actual distribution values depend on E-parameter in stan, so only normalisation
                # is accounted for at this stage
                E_true_norm = 1 / (np.log(fit_inputs["Emax"] / fit_inputs["Emin"]))

                fit_inputs["bg_llh"][dm.S == self.events.types] = np.log(
                    prob_ereco_and_omega
                    * E_true_norm  # accounts for E_nu integral, with a flat log(E) distribution
                    * N_dm    # multiply pdf by N_dm / time_norm to get rate of event per time
                    / time_norm
                    / self.events.N   # divide by total number of selected events
                                      # because we multiply in stan by parameter Nex_bg
                )

        # use the Eres slices for each event as data input
        # evaluate the splines at eadch event's reco energy
        # make this a loop over the IRFs
        # fix the number of Etrue vals to 14 because we only use it for this one data release
        # first index is event, second IRF Etrue bin

        self._ereco_spline_evals = np.zeros(
            (self.events.N, R2021EnergyResolution._log_tE_grid.size)
        )
        # energy_resolution.ereco_splines has first index as declination of IRF, bin_edges=-[90, -10, 10, 90] in degrees
        _, dec = uv_to_icrs(self.events.unit_vectors)
        dec_idx = np.zeros(self.events.N, dtype=int)
        for et in self._event_types:
            dec_idx[et.S == self.events.types] = (
                np.digitize(
                    dec[et.S == self.events.types].to_value(u.rad),
                    self._exposure_integral[
                        et
                    ].energy_resolution.dec_bin_edges.to_value(u.rad),
                )
                - 1
            )
        log_energies = np.log10(self.events.energies.to_value(u.GeV))

        idxs = (
            np.digitize(
                np.log10(self.events.energies.to_value(u.GeV)),
                R2021EnergyResolution._logEreco_grid_edges,
            )
            - 1
        )
        # safeguard against index errors in stan
        idxs = np.where(idxs == -1, 0, idxs)
        idxs = np.where(
            idxs > R2021EnergyResolution._logEreco_grid.size - 1,
            R2021EnergyResolution._logEreco_grid.size - 1,
            idxs,
        )
        ereco_indexed = R2021EnergyResolution._logEreco_grid[idxs]

        for et in self._event_types:
            for c_d in range(
                self._exposure_integral[et].energy_resolution.dec_binc.size
            ):
                try:
                    self._ereco_spline_evals[
                        (et.S == self.events.types) & (dec_idx == c_d)
                    ] = np.array(
                        [
                            self._exposure_integral[et].energy_resolution._2dsplines[
                                c_d
                            ](
                                logE,
                                self._exposure_integral[
                                    et
                                ].energy_resolution._log_tE_grid,
                                grid=False,
                            )
                            # for logE in ereco_indexed[
                            #    (et.S == self.events.types) & (dec_idx == c_d)
                            # ]
                            for logE in log_energies[
                                (et.S == self.events.types) & (dec_idx == c_d)
                            ]
                        ]
                    )
                except ValueError as e:
                    # When there is no match, ValueError is raised
                    # Clearly this should be an IndexError...
                    pass

        # Cath possible issues with the indexing
        if np.any(np.isnan(self._ereco_spline_evals)) or np.any(
            np.isinf(self._ereco_spline_evals)
        ):
            raise ValueError("Something is wrong, please fix me")
        fit_inputs["ereco_grid"] = self._ereco_spline_evals

        """
        idxs = np.digitize(
            np.log10(self.events.energies.to_value(u.GeV)),
            R2021EnergyResolution._logEreco_grid_edges,
        )
        # safeguard against index errors in stan
        idxs = np.where(idxs == 0, 1, idxs)
        idxs = np.where(
            idxs > R2021EnergyResolution._logEreco_grid.size,
            R2021EnergyResolution._logEreco_grid.size,
            idxs,
        )
        fit_inputs["ereco_idx"] = idxs
        """

        for c, event_type in enumerate(self._event_types):
            integral_grid.append([])
            integral_grid_2d.append([])
            for grid in self._exposure_integral[event_type].integral_grid:

                if len(grid.shape) == 2:
                    integral_grid_2d[-1].append(np.log(grid.to_value(u.m**2)).tolist())

                else:
                    integral_grid[-1].append(np.log(grid.to_value(u.m**2)).tolist())
            """integral_grid.append(
                [
                    np.log(_.to_value(u.m**2)).tolist()
                    for _ in self._exposure_integral[event_type].integral_grid
                ]
            )"""

            if self._sources.atmospheric:
                atmo_integ_val.append(
                    self._exposure_integral[event_type]
                    .integral_fixed_vals[0]
                    .to_value(u.m**2)
                )
        """
        try:
            ang_sys = Parameter.get_parameter("ang_sys_add")

        except ValueError:
            pass
        """

        fit_inputs["integral_grid"] = integral_grid
        fit_inputs["integral_grid_2d"] = integral_grid_2d
        fit_inputs["atmo_integ_val"] = atmo_integ_val
        fit_inputs["T"] = obs_time
        # To work with cmdstanpy serialization
        fit_inputs = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in fit_inputs.items()
        }

        return fit_inputs

    def _get_par_ranges(self):
        """
        Extract the parameter ranges to use in Stan from the
        defined parameters.
        """

        if self._sources.point_source:
            if self._fit_nex:
                Nex = Parameter.get_parameter("Nex_src")
                self._nex_par_range = Nex.par_range
            # TODO make similar to spectral parameters, L is not appended to the parameter list of the source
            if self._shared_luminosity:
                key = "luminosity"
            else:
                key = "%s_luminosity" % self._sources.point_source[0].name

            try:
                self._lumi_par_range = Parameter.get_parameter(key).par_range
                self._lumi_par_range = self._lumi_par_range.to_value(u.GeV / u.s)
            except ValueError:
                pass

            if self._logparabola or self._power_law:
                self._src_index_par_range = (
                    self._sources.point_source[0].parameters["index"].par_range
                )
            if self._logparabola:
                self._beta_index_par_range = (
                    self._sources.point_source[0].parameters["beta"].par_range
                )
            if self._logparabola or self._pgamma:
                self._E0_src_par_range = (
                    self._sources.point_source[0].parameters["norm_energy"].par_range
                )
            if self._seyfert:
                self._eta_par_range = (
                    self._sources.point_source[0].parameters["eta"].par_range
                )
                self._P_par_range = (
                    self._sources.point_source[0].parameters["P"].par_range
                )

        if self._sources.diffuse:
            self._diff_index_par_range = Parameter.get_parameter("diff_index").par_range
            self._diffuse_norm_par_range = Parameter.get_parameter(
                "diffuse_norm"
            ).par_range.to_value(1 / u.GeV / u.m**2 / u.s)

        if self._sources.atmospheric:
            self._F_atmo_par_range = Parameter.get_parameter(
                "F_atmo"
            ).par_range.to_value(1 / u.m**2 / u.s)

        if self._fit_ang_sys:
            self._ang_sys_par_range = Parameter.get_parameter(
                "ang_sys_add"
            ).par_range.to_value(u.rad)
