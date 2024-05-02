import numpy as np
import os
import h5py
import logging
import collections
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from typing import List, Union, Dict, Callable, Iterable
import corner
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import ligo.skymap.plot
import arviz as av
from pathlib import Path

from math import ceil, floor
from time import time

from cmdstanpy import CmdStanModel

from hierarchical_nu.source.source import Sources, PointSource, icrs_to_uv, uv_to_icrs
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import IsotropicDiffuseBG
from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.detector.icecube import EventType, CAS, Refrigerator
from hierarchical_nu.detector.r2021 import (
    R2021EnergyResolution,
    R2021LogNormEnergyResolution,
)
from hierarchical_nu.precomputation import ExposureIntegral
from hierarchical_nu.events import Events
from hierarchical_nu.priors import Priors, NormalPrior, LogNormalPrior, UnitPrior
from hierarchical_nu.source.source import spherical_to_icrs, uv_to_icrs

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.utils.git import git_hash


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StanFit:
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
    ):
        """
        To set up and run fits in Stan.
        """

        self._sources = sources
        # self._detector_model_type = detector_model
        if not isinstance(event_types, list):
            event_types = [event_types]
        if isinstance(observation_time, u.quantity.Quantity):
            observation_time = {event_types[0]: observation_time}
        assert len(event_types) == len(observation_time)
        self._event_types = event_types
        self._events = events
        self._observation_time = observation_time
        self._n_grid_points = n_grid_points
        self._nshards = nshards
        self._priors = priors
        self._use_event_tag = use_event_tag

        self._sources.organise()

        stan_file_name = os.path.join(STAN_GEN_PATH, "model_code")

        if sources.N != 0:
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
            self._def_var_names.append("src_index")

        if self._sources.diffuse:
            self._def_var_names.append("F_diff")
            self._def_var_names.append("diff_index")

        if self._sources.atmospheric:
            self._def_var_names.append("F_atmo")

        if self._sources._point_source and (
            self._sources.atmospheric or self._sources.diffuse
        ):
            self._def_var_names.append("f_arr")
            self._def_var_names.append("f_det")

        self._exposure_integral = collections.OrderedDict()

        # Check for shared luminosity and src_index params
        try:
            Parameter.get_parameter("luminosity")
            self._shared_luminosity = True
        except ValueError:
            self._shared_luminosity = False

        try:
            Parameter.get_parameter("src_index")
            self._shared_src_index = True
        except ValueError:
            self._shared_src_index = False

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
    def events(self):
        return self._events

    @events.setter
    def events(self, events: Events):
        if isinstance(events, Events):
            self._events = events
        else:
            raise ValueError("events must be instance of Events")

    def precomputation(
        self,
        exposure_integral: collections.OrderedDict = None,
    ):
        if not exposure_integral:
            for event_type in self._event_types:
                self._exposure_integral[event_type] = ExposureIntegral(
                    self._sources, event_type, self._n_grid_points
                )

        else:
            self._exposure_integral = exposure_integral

    def generate_stan_code(self):
        self._fit_filename = self._stan_interface.generate()

    def set_stan_filename(self, fit_filename):
        self._fit_filename = fit_filename

    def compile_stan_code(self, include_paths=None):
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

    def setup_and_run(
        self,
        iterations: int = 1000,
        chains: int = 1,
        seed: int = None,
        show_progress: bool = False,
        include_paths: List[str] = None,
        **kwargs,
    ):
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

    def plot_trace(self, var_names=None, **kwargs):
        """
        Trace plot using list of stan parameter keys.
        """

        if not var_names:
            var_names = self._def_var_names

        axs = av.plot_trace(self._fit_output, var_names=var_names, **kwargs)
        fig = axs.flatten()[0].get_figure()

        return fig, axs

    def plot_trace_and_priors(self, var_names=None, **kwargs):
        """
        Trace plot and overplot the used priors.
        """

        fig, axs = self.plot_trace(var_names=var_names, show=False, **kwargs)

        if not var_names:
            var_names = self._def_var_names

        priors_dict = self._priors.to_dict()

        for ax_double in axs:
            name = ax_double[0].get_title()
            # check if there is a prior available for the variable
            try:
                # If so, get it and plot it
                prior = priors_dict[name]
                ax = ax_double[0]
                supp = ax.get_xlim()
                x = np.linspace(*supp, 1000)

                if "transform" in kwargs.keys():
                    # Assumes that the only sensible transformation is log10
                    pdf = prior.pdf_logspace
                    ax.plot(
                        x,
                        pdf(np.power(10, x) * prior.UNITS),
                        color="black",
                        alpha=0.4,
                        zorder=0,
                    )

                else:
                    pdf = prior.pdf
                    ax.plot(x, pdf(x * prior.UNITS), color="black", alpha=0.4, zorder=0)
                if isinstance(prior, UnitPrior):
                    try:
                        unit = prior.UNITS.unit
                    except AttributeError:
                        unit = prior.UNITS
                    if "transform" in kwargs.keys():
                        # yikes
                        title = f"[$\\log_{{10}}\\left (\\frac{{{name}}}{{{unit.to_string('latex_inline').strip('$')}}}\\right )$]"
                    else:
                        title = f"{name} [{unit.to_string('latex_inline')}]"
                    ax.set_title(title)

            except KeyError:
                pass

        fig = axs.flatten()[0].get_figure()

        return fig, axs

    def _get_kde(
        self,
        var_name,
        index: Union[int, slice, None] = None,
        transform: Callable = lambda x: x,
    ):
        try:
            chain = self._fit_output.stan_variable(var_name)
        except AttributeError:
            chain = self._fit_output[var_name]
        if index is not None:
            data = chain.T[index]
        else:
            data = chain
        return av.kde(transform(data))

    def corner_plot(self, var_names=None, truths=None):
        """
        Corner plot using list of Stan parameter keys and optional
        true values if working with simulated data.
        """

        logger.warning(
            "If you are in a reloaded state with multiple point sources, add the used source list through <StanFit._sources = sources>"
        )
        if not var_names:
            var_names = self._def_var_names

        try:
            chain = self._fit_output.stan_variables()
            stan = True
        except AttributeError:
            chain = self._fit_output
            stan = False

        # Organise samples
        samples_list = []
        label_list = []

        for key in var_names:
            if stan:
                if len(np.shape(chain[key])) > 1:
                    # This is for array-like variables, e.g. multiple PS
                    # having their own entry in src_index
                    for samp, src in zip(chain[key].T, self._sources.point_source):
                        samples_list.append(samp)
                        if key == "L" or key == "src_index":
                            label = "%s_" % src.name + key
                        else:
                            label = key

                        label_list.append(label)

                else:
                    samples_list.append(chain[key])
                    label_list.append(key)

            else:
                # check for len(np.shape(chain[key]) > 2 because extra dim for chains
                # would be, e.g. for 3 sources, (chains, iter_sampling, 3)
                if len(np.shape(chain[key])) > 2:
                    for i, src in zip(
                        range(np.shape(chain[key])[-1]), self._sources.point_source
                    ):
                        if key == "L" or key == "src_index":
                            label = "%s_" % src.name + key
                        else:
                            label = key
                        samples_list.append(
                            chain[key][:, :, i].reshape(
                                (
                                    self._fit_meta["iter_sampling"]
                                    * self._fit_meta["chains"],
                                    1,
                                )
                            )
                        )
                        label_list.append(label)

                else:
                    samples_list.append(
                        chain[key].reshape(
                            (
                                self._fit_meta["iter_sampling"]
                                * self._fit_meta["chains"],
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
        true_assoc: Union[Iterable, None] = None,
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
        if true_assoc is not None:
            true_assoc = np.atleast_1d(true_assoc)
            assert true_assoc.size == mask.size

        if color_scale == "lin":
            norm = colors.Normalize(0.0, 1.0, clip=True)
        elif color_scale == "log":
            norm = colors.LogNorm(1e-8, 1.0, clip=True)
        else:
            raise ValueError("No other scale supported")
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis_r)
        color = mapper.to_rgba(assoc_prob)

        indices = np.arange(self._events.N, dtype=int)[mask]

        for c, i in enumerate(indices):
            # get the support (is then log10(E/GeV) and the pdf values
            supp, pdf = self._get_kde("E", i, lambda x: np.log10(x))
            # exponentiate the support, because we rescale the axis in the end
            if true_assoc is not None:
                if assoc_idx == true_assoc[i]:
                    ax.plot(
                        np.power(10, supp),
                        pdf,
                        color="magenta",
                        zorder=assoc_prob[c] + 1 - 1e-4,
                        lw=3,
                        alpha=0.4,
                    )

            ax.plot(
                np.power(10, supp),
                pdf,
                color=color[c],
                zorder=assoc_prob[c] + 1,
            )
        _, yhigh = ax.get_ylim()
        ax.set_xscale("log")

        for c, i in enumerate(indices):
            ax.vlines(
                self.events.energies[mask][c].to_value(u.GeV),
                yhigh,
                1.05 * yhigh,
                color=color[c],
                lw=1,
                zorder=assoc_prob[c] + 1,
            )
            if true_assoc is not None and assoc_idx == true_assoc[i]:
                ax.vlines(
                    self.events.energies[mask][c].to_value(u.GeV),
                    yhigh,
                    1.05 * yhigh,
                    color="magenta",
                    lw=3,
                    alpha=0.4,
                    zorder=assoc_prob[c] + 1 - 1e-4,
                )
                

            if assoc_prob[c] > 0.2:
                # if we have more than 20% association prob, link both lines up
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

        ax.text(1e7, yhigh, "$\hat E$")

        ax.set_xlabel(r"$E~[\mathrm{GeV}]$")
        ax.set_ylabel("pdf")
        ax.set_xlim(8e1, 1.4e8)
        return ax, mapper

    def plot_energy_posterior(
        self,
        center: Union[SkyCoord, int, None] = None,
        assoc_idx: int = 0,
        radius: Union[u.Quantity[u.deg], None] = None,
        color_scale: str = "lin",
        true_assoc: Union[Iterable, None] = None,
    ):
        """
        Plot energy posteriors in log10-space.
        Color corresponds to association probability.
        :param center: SkyCoord, int identifying PS or None to center selection on
        :param assoc_idx: integer identifying the source component to calculate assoc prob
        :param radius: if center is not None, select only events within radius around center
        :param color_scale: color scale of assoc prob, either "lin" or "log"
        """

        fig, ax = plt.subplots(dpi=150)
        if isinstance(center, int):
            center = self.get_src_position(center)
        ax, mapper = self._plot_energy_posterior(
            ax, center, assoc_idx, radius, color_scale, true_assoc
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
        true_assoc: Union[Iterable, None] = None,
    ):
        ev_class = np.array(self._get_event_classifications())
        assoc_prob = ev_class[:, assoc_idx]

        if true_assoc is not None:
            true_assoc = np.atleast_1d(true_assoc)
            assert true_assoc.size == assoc_prob.size

        min = 0.0
        max = 1.0
        if color_scale == "lin":
            norm = colors.Normalize(min, max, clip=True)
        elif color_scale == "log":
            norm = colors.LogNorm(1e-8, max, clip=True)
        else:
            raise ValueError("No other scale supported")
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis_r)
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
            if true_assoc is not None:
                if true_assoc[i] == assoc_idx:
                    edgecolor = colors.colorConverter.to_rgba("magenta", alpha=0.5)

            ax.scatter(
                coords[i].ra.deg,
                coords[i].dec.deg,
                color=color[i],
                zorder=assoc_prob[i] + 1,
                transform=ax.get_transform("icrs"),
                edgecolor=edgecolor,
                s=30,
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
        true_assoc: Union[Iterable, None] = None,
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
            center, ax, radius, assoc_idx, color_scale, true_assoc
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
        true_assoc: Union[Iterable, None] = None,
    ):
        fig = plt.figure(dpi=150, figsize=(8, 3))
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
            ax, center, assoc_idx, radius, color_scale, true_assoc
        )

        ax.set_xlabel(r"$E~[\mathrm{GeV}]$")
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.set_ylabel("posterior pdf")
        axs.append(ax)
        fig.colorbar(mapper, label=f"association probability to {assoc_idx:n}", ax=ax)

        ax = fig.add_subplot(
            gs[0, 0],
            projection="astro degrees zoom",
            center=center,
            radius=f"{radius.to_value(u.deg)} deg",
        )

        ax, _ = self._plot_roi(center, ax, radius, assoc_idx, color_scale, true_assoc)
        axs.insert(0, ax)

        return fig, ax

    def save(self, path, overwrite: bool = False):
        
        # Check if filename consists of a path to some directory as well as the filename
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        if dirname:
            if not os.path.exists(dirname):
                logger.warning(f"{dirname} does not exist, saving instead to {os.getcwd()}")
                dirname = os.getcwd()
        else: 
            dirname = os.getcwd()
        path = Path(dirname) / Path(filename)
        
        if os.path.exists(path) and not overwrite:
            logger.warning(f"File {filename} already exists.")
            file = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            file += f"_{int(time())}"
            filename = file + ext

        with h5py.File(filename, "w") as f:
            fit_folder = f.create_group("fit")
            inputs_folder = fit_folder.create_group("inputs")
            outputs_folder = fit_folder.create_group("outputs")
            meta_folder = fit_folder.create_group("meta")

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
            meta_folder.create_dataset("runset", data=str(self._fit_output.runset))
            meta_folder.create_dataset("diagnose", data=self._fit_output.diagnose())
            f.create_dataset("version", data=git_hash)

            summary = self._fit_output.summary()

            # List of keys for which we are looking in the entirety of stan parameters
            key_stubs = [
                "lp__",
                "L",
                "_luminosity",
                "src_index",
                "_src_index",
                "E[",
                "Esrc[",
                "F_atmo",
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
            for k, v in summary["N_Eff"].items():
                for key in key_stubs:
                    if key in k:
                        keys.append(k)
                        break

            R_hat = np.array([summary["R_hat"][k] for k in keys])
            N_Eff = np.array([summary["N_Eff"][k] for k in keys])

            meta_folder.create_dataset("N_Eff", data=N_Eff)
            meta_folder.create_dataset("R_hat", data=R_hat)
            meta_folder.create_dataset("parameters", data=np.array(keys, dtype="S"))

        self.events.to_file(filename, append=True)

        # Add priors separately
        self.priors.addto(filename, "priors")

    def save_csvfiles(self, directory):
        """
        Save cmdstanpy csv files
        :param directory: Directory to save csf files to.
        """

        self._fit_output.save_csvfiles(directory)

    @classmethod
    def from_file(cls, filename):
        """
        Load fit output from file. Allows to
        make plots and run classification check.
        """

        # priors_dict = {}

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

            for k, v in f["fit/inputs"].items():
                # if "mu" in k or "sigma" in k:
                #    priors_dict[k] = v[()]

                fit_inputs[k] = v[()]

            for k, v in f["fit/outputs"].items():
                # Add extra dimension for number of chains
                if k == "local_pars" or k == "global_pars":
                    continue

                temp = v[()]
                if len(temp.shape) == 1:
                    # non-vector variable
                    fit_outputs[k] = temp.reshape(
                        (fit_meta["chains"], fit_meta["iter_sampling"])
                    )
                else:
                    # Reshape to chains x draws x dim
                    fit_outputs[k] = temp.reshape(
                        (
                            fit_meta["chains"],
                            fit_meta["iter_sampling"],
                            *temp.shape[1:],
                        )
                    )

        event_types = [
            Refrigerator.stan2dm(_) for _ in fit_inputs["event_types"].tolist()
        ]

        obs_time = fit_inputs["T"] * u.s

        obs_time_dict = {et: obs_time[k] for k, et in enumerate(event_types)}

        try:
            priors = Priors.from_group(filename, "priors")
        except KeyError:
            # lazy fix for backwards compatibility
            priors = Priors()

        events = Events.from_file(filename)

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

        fit = cls(Sources(), event_types, events, obs_time_dict, priors)

        fit._fit_output = fit_outputs
        fit._fit_inputs = fit_inputs
        fit._fit_meta = fit_meta

        if "src_index_grid" in fit_inputs.keys():
            fit._def_var_names.append("L")
            fit._def_var_names.append("src_index")

        if "diff_index_grid" in fit_inputs.keys():
            fit._def_var_names.append("F_diff")
            fit._def_var_names.append("diff_index")

        if "atmo_integ_val" in fit_inputs.keys():
            fit._def_var_names.append("F_atmo")

        if "src_index_grid" in fit_inputs.keys() and (
            "atmo_integ_val" in fit_inputs.keys()
            or "diff_index_grid" in fit_inputs.keys()
        ):
            fit._def_var_names.append("f_arr")
            fit._def_var_names.append("f_det")

        return fit

    def diagnose(self):
        try:
            print(self._fit_output.diagnose().decode("ascii"))
        except AttributeError:
            print(self._fit_meta["diagnose"].decode("ascii"))

    def check_classification(self, sim_outputs):
        """
        For the case of simulated data, check if
        events are correctly classified into the
        different source categories.
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

    def _get_event_classifications(self):
        # logprob is a misnomer, this is actually the rate parameter of each source component
        try:
            logprob = self._fit_output.stan_variable("lp").transpose(1, 2, 0)
        except AttributeError:
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
        # average over samples, hence axis=-1
        assoc_prob = np.average(ratio, axis=-1).tolist()
        return assoc_prob

    def _get_fit_inputs(self):

        self._get_par_ranges()
        fit_inputs = {}
        fit_inputs["N"] = self._events.N
        if self._nshards not in [0, 1]:
            # Number of shards and max. events per shards only used if multithreading is desired
            fit_inputs["N_shards"] = self._nshards
            fit_inputs["J"] = ceil(fit_inputs["N"] / fit_inputs["N_shards"])
        fit_inputs["Ns_tot"] = len([s for s in self._sources.sources])
        fit_inputs["Edet"] = self._events.energies.to(u.GeV).value
        fit_inputs["omega_det"] = self._events.unit_vectors
        fit_inputs["omega_det"] = [
            (_ / np.linalg.norm(_)).tolist() for _ in fit_inputs["omega_det"]
        ]
        fit_inputs["event_type"] = self._events.types
        fit_inputs["kappa"] = self._events.kappas
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

        if self._sources.point_source:
            fit_inputs["Emin_src"] = (
                Parameter.get_parameter("Emin_src").value.to(u.GeV).value
            )
            fit_inputs["Emax_src"] = (
                Parameter.get_parameter("Emax_src").value.to(u.GeV).value
            )

        fit_inputs["Emin"] = Parameter.get_parameter("Emin").value.to(u.GeV).value
        fit_inputs["Emax"] = Parameter.get_parameter("Emax").value.to(u.GeV).value

        if self._sources.diffuse:
            fit_inputs["Emin_diff"] = (
                Parameter.get_parameter("Emin_diff").value.to(u.GeV).value
            )
            fit_inputs["Emax_diff"] = (
                Parameter.get_parameter("Emax_diff").value.to(u.GeV).value
            )

        integral_grid = []
        atmo_integ_val = []
        obs_time = []

        for c, event_type in enumerate(self._event_types):
            obs_time.append(self._observation_time[event_type].to(u.s).value)

            # event_type = self._detector_model_type.event_types[0]

        fit_inputs["Ngrid"] = self._exposure_integral[event_type]._n_grid_points

        if self._use_event_tag:
            fit_inputs["event_tag"] = (
                np.array(self._events.get_tags(self._sources)).astype(int) + 1
            )

        if self._sources.point_source:
            # Check for shared source index
            if self._shared_src_index:
                key = "src_index"

            # Otherwise just use first source in the list
            # src_index_grid is identical for all point sources
            else:
                key = "%s_src_index" % self._sources.point_source[0].name

            fit_inputs["src_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids[key]

            # PS parameter limits
            fit_inputs["src_index_min"] = self._src_index_par_range[0]
            fit_inputs["src_index_max"] = self._src_index_par_range[1]

            fit_inputs["Lmin"] = self._lumi_par_range[0]
            fit_inputs["Lmax"] = self._lumi_par_range[1]

        # Inputs for priors of point sources
        if self._priors.src_index.name not in ["normal", "lognormal"]:
            raise ValueError("No other prior type for source index implemented.")
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

        if self._sources.diffuse:
            # Just take any for now, using default parameters it doesn't matter
            fit_inputs["diff_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids["diff_index"]

            fit_inputs["diff_index_min"] = self._diff_index_par_range[0]
            fit_inputs["diff_index_max"] = self._diff_index_par_range[1]
            fit_inputs["F_diff_min"] = self._F_diff_par_range[0]
            fit_inputs["F_diff_max"] = self._F_diff_par_range[1]

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
                self._sources.atmospheric.flux_model.total_flux_int.to(
                    1 / (u.m**2 * u.s)
                ).value
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
        # log_energies = np.log10(self.events.energies.to_value(u.GeV))

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
                            for logE in ereco_indexed[
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
            integral_grid.append(
                [
                    np.log(_.to(u.m**2).value).tolist()
                    for _ in self._exposure_integral[event_type].integral_grid
                ]
            )

            if self._sources.atmospheric:
                atmo_integ_val.append(
                    self._exposure_integral[event_type]
                    .integral_fixed_vals[0]
                    .to(u.m**2)
                    .value
                )

        fit_inputs["integral_grid"] = integral_grid
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
            if self._shared_luminosity:
                key = "luminosity"
            else:
                key = "%s_luminosity" % self._sources.point_source[0].name

            self._lumi_par_range = Parameter.get_parameter(key).par_range
            self._lumi_par_range = self._lumi_par_range.to_value(u.GeV / u.s)

            if self._shared_src_index:
                key = "src_index"
            else:
                key = "%s_src_index" % self._sources.point_source[0].name

            self._src_index_par_range = Parameter.get_parameter(key).par_range

        if self._sources.diffuse:
            self._diff_index_par_range = Parameter.get_parameter("diff_index").par_range
            self._F_diff_par_range = Parameter.get_parameter("F_diff").par_range.to_value(1 / u.m**2 / u.s)

        if self._sources.atmospheric:
            self._F_atmo_par_range = Parameter.get_parameter(
                "F_atmo"
            ).par_range.to_value(1 / u.m**2 / u.s)
