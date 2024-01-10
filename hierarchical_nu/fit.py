import numpy as np
import os
import h5py
import logging
import collections
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from typing import List, Union, Dict
import corner
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import ligo.skymap.plot
import arviz as av

from math import ceil, floor
from time import time

from cmdstanpy import CmdStanModel

from hierarchical_nu.source.source import Sources, PointSource, icrs_to_uv, uv_to_icrs
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import IsotropicDiffuseBG
from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.detector.icecube import EventType, CAS, Refrigerator
from hierarchical_nu.precomputation import ExposureIntegral
from hierarchical_nu.events import Events
from hierarchical_nu.priors import Priors, NormalPrior, LogNormalPrior, UnitPrior

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.stan.fit_interface import StanFitInterface


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

    def setup_stan_fit(self, filename):
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

    def get_src_position(self):
        try:
            ra = self._sources.point_source[0].ra
            dec = self._sources.point_source[0].dec
        except IndexError:
            ra, dec = uv_to_icrs(self._fit_inputs["varpi"][0])
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

    def corner_plot(self, var_names=None, truths=None):
        """
        Corner plot using list of Stan parameter keys and optional
        true values if working with simulated data.
        """

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
                    for i, s in enumerate(chain[key].T):
                        samples_list.append(s)
                        if key == "L" or key == "src_index":
                            label = "ps_%i_" % i + key
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
                    for i in range(np.shape(chain[key][-1])):
                        if key == "L" or key == "src_index":
                            label = "ps_%i_" % i + key
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
                if truths[key].size > 1:
                    for t in truths[key]:
                        truths_list.append(t)

                else:
                    truths_list.append(truths[key])

        else:
            truths_list = None

        samples = np.column_stack(samples_list)

        return corner.corner(samples, labels=label_list, truths=truths_list)

    def _plot_energy_posterior(self, input_axs, ax, source_idx, color_scale):
        ev_class = np.array(self._get_event_classifications())
        assoc_prob = ev_class[:, source_idx]

        if color_scale == "lin":
            norm = colors.Normalize(0.0, 1.0, clip=True)
        elif color_scale == "log":
            norm = colors.LogNorm(1e-8, 1.0, clip=True)
        else:
            raise ValueError("No other scale supported")
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis_r)
        color = mapper.to_rgba(assoc_prob)

        for c, line in enumerate(input_axs[0, 0].lines):
            ax.plot(
                np.power(10, line.get_data()[0]),
                line.get_data()[1],
                color=color[c],
                zorder=assoc_prob[c] + 1,
            )
        _, yhigh = ax.get_ylim()
        ax.set_xscale("log")

        for c, line in enumerate(input_axs[0, 0].lines):
            ax.vlines(
                self.events.energies[c].to_value(u.GeV),
                yhigh,
                1.05 * yhigh,
                color=color[c],
                lw=1,
                zorder=assoc_prob[c] + 1,
            )
            if assoc_prob[c] > 0.1:
                # if we have more than 20% association prob, link both lines up
                x, y = input_axs[0, 0].lines[c].get_data()
                idx_posterior = np.argmax(y)
                ax.plot(
                    [
                        np.power(10, x[idx_posterior]),
                        self.events.energies[c].to_value(u.GeV),
                    ],
                    [y[idx_posterior], yhigh],
                    lw=0.5,
                    color="black",
                    ls="--",
                )

        ax.text(1e7, yhigh, "$\hat E$")

        ax.set_xlabel(r"$E~[\mathrm{GeV}]$")
        ax.set_ylabel("pdf")

        return ax, mapper

    def plot_energy_posterior(self, source_idx: int = 0, color_scale: str = "lin"):
        """
        Plot energy posteriors in log10-space.
        Color corresponds to association to point source presumed
        to be in self.sources[0]
        TODO: make compatible with multiple PS
        """

        _, axs = self.plot_trace(
            var_names=["E"],
            transform=lambda x: np.log10(x),
            show=False,
            combined=True,
            divergences=None,
        )

        fig, ax = plt.subplots(dpi=150)

        ax, mapper = self._plot_energy_posterior(axs, ax, source_idx, color_scale)
        fig.colorbar(mapper, ax=ax, label=f"association probability to {source_idx:n}")

        return fig, ax

    def _plot_roi(self, source_coords, ax, radius, source_idx, color_scale):
        ev_class = np.array(self._get_event_classifications())
        assoc_prob = ev_class[:, source_idx]

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

        sep = events.coords.separation(source_coords).deg
        mask = sep < radius.to_value(u.deg) * 1.41
        # sloppy, don't know how to do that rn

        ax.scatter(
            source_coords.ra.deg,
            source_coords.dec.deg,
            marker="x",
            color="black",
            zorder=10,
            alpha=0.4,
            transform=ax.get_transform("icrs"),
        )

        for c, (colour, coord) in enumerate(zip(color[mask], events.coords[mask])):
            ax.scatter(
                coord.ra.deg,
                coord.dec.deg,
                color=colour,
                alpha=0.4,
                zorder=assoc_prob[c] + 1,
                transform=ax.get_transform("icrs"),
            )

        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        ax.grid()

        return ax, mapper

    @u.quantity_input
    def plot_roi(
        self, radius=5.0 * u.deg, source_idx: int = 0, color_scale: str = "lin"
    ):
        """
        Create plot of the ROI.
        Events are colour-coded dots, color corresponding
        to the association probability to the point source proposed.
        Assumes there is a point source in self.sources[0].
        Size of events are meaningless.
        """

        source_coords = self.get_src_position()

        # we are working in degrees here
        fig, ax = plt.subplots(
            subplot_kw={
                "projection": "astro degrees zoom",
                "center": source_coords,
                "radius": f"{radius.to_value(u.deg)} deg",
            },
            dpi=150,
        )

        ax, mapper = self._plot_roi(source_coords, ax, radius, source_idx, color_scale)
        fig.colorbar(mapper, ax=ax, label=f"association probability to {source_idx:n}")

        return fig, ax

    @u.quantity_input
    def plot_energy_and_roi(
        self, radius=5 * u.deg, source_idx: int = 0, color_scale: str = "lin"
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

        ax = fig.add_subplot(gs[0, 1])

        _, axs = self.plot_trace(
            var_names=["E"],
            transform=lambda x: np.log10(x),
            show=False,
            combined=True,
            divergences=None,
        )

        source_coords = self.get_src_position()

        ax, mapper = self._plot_energy_posterior(axs, ax, source_idx, color_scale)

        ax.set_xlabel(r"$E~[\mathrm{GeV}]$")
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.set_ylabel("posterior pdf")
        fig.colorbar(mapper, label=f"association probability to {source_idx:n}", ax=ax)

        ax = fig.add_subplot(
            gs[0, 0],
            projection="astro degrees zoom",
            center=source_coords,
            radius=f"{radius.to_value(u.deg)} deg",
        )

        ax, _ = self._plot_roi(source_coords, ax, radius, source_idx, color_scale)

        return fig, ax

    def save(self, filename, overwrite: bool = False):
        if os.path.exists(filename) and not overwrite:
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

            for key, value in self._fit_output.stan_variables().items():
                outputs_folder.create_dataset(key, data=value)

            # Save some metadata for debugging, easier loading from file
            meta_folder.create_dataset("divergences", data=self._fit_output.divergences)
            meta_folder.create_dataset("chains", data=self._fit_output.chains)
            meta_folder.create_dataset(
                "iter_sampling", data=self._fit_output._iter_sampling
            )
            meta_folder.create_dataset("runset", data=str(self._fit_output.runset))
            meta_folder.create_dataset("diagnose", data=self._fit_output.diagnose())

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

        priors = Priors.from_group(filename, "priors")

        events = Events.from_file(filename)

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
                assumed.append(event_labels[i])

        if not wrong:
            print("All events are correctly classified")
        else:
            print(
                "A total of %i events out of %i are misclassified"
                % (len(wrong), len(event_labels))
            )

        return wrong, assumed, correct

    def _get_event_classifications(self):
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

        n_comps = np.shape(logprob)[1]

        prob_each_src = []
        for lp in logprob:
            lps = []
            ps = []
            for src in range(n_comps):
                lps.append(np.mean(np.exp(lp[src])))
            norm = sum(lps)

            for src in range(n_comps):
                ps.append(lps[src] / norm)

            prob_each_src.append(ps)

        return prob_each_src

    def _get_fit_inputs(self):
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

        if self._sources.point_source:
            try:
                Parameter.get_parameter("src_index")
                key = "src_index"
            except ValueError:
                key = "ps_0_src_index"

            fit_inputs["src_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids[key]

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
            fit_inputs["lumi_xmin"] = self._priors.luminosity.xmin
            fit_inputs["lumi_alpha"] = self._priors.luminosity.alpha
        else:
            raise ValueError("No other prior type for luminosity implemented.")

        if self._sources.diffuse:
            # Just take any for now, using default parameters it doesn't matter
            fit_inputs["diff_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids["diff_index"]

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
            fit_inputs[
                "atmo_integrated_flux"
            ] = self._sources.atmospheric.flux_model.total_flux_int.to(
                1 / (u.m**2 * u.s)
            ).value

            # Priors for atmo model
            if self._priors.atmospheric_flux.name == "lognormal":
                fit_inputs["f_atmo_mu"] = self._priors.atmospheric_flux.mu
                fit_inputs["f_atmo_sigma"] = self._priors.atmospheric_flux.sigma
            elif self._priors.atmospheric_flux.name == "normal":
                fit_inputs["f_atmo_mu"] = self._priors.atmospheric_flux.mu.to_value(
                    self._priors.atmospheric_flux.UNITS
                )
                fit_inputs[
                    "f_atmo_sigma"
                ] = self._priors.atmospheric_flux.sigma.to_value(
                    self._priors.atmospheric_flux.UNITS
                )
            else:
                raise ValueError(
                    "No other prior type for atmospheric flux implemented."
                )

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
