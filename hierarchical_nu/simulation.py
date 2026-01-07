import numpy as np
import os
from typing import Union, List, Dict
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.interpolate import RegularGridInterpolator
import h5py
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel
import logging
import collections
import ligo.skymap.plot
from pathlib import Path

from time import time as thyme

from icecube_tools.utils.vMF import get_theta_p


from hierarchical_nu.utils.plotting import SphericalCircle

from hierarchical_nu.detector.icecube import EventType, NT, CAS
from hierarchical_nu.precomputation import ExposureIntegral
from hierarchical_nu.source.source import (
    Sources,
    PointSource,
    icrs_to_uv,
    BackgroundSource,
)
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import (
    IsotropicDiffuseBG,
    LogParabolaSpectrum,
    PowerLawSpectrum,
    TwiceBrokenPowerLaw,
    PGammaSpectrum,
)
from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.events import Events
from hierarchical_nu.utils.roi import ROI, CircularROI, ROIList

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.utils.git import git_hash
from .source.source_info import SourceInfo


sim_logger = logging.getLogger(__name__)
sim_logger.setLevel(logging.WARNING)


class Simulation(SourceInfo):
    """
    To set up and run simulations.
    """

    @u.quantity_input
    def __init__(
        self,
        sources: Sources,
        event_types: Union[EventType, List[EventType]],
        observation_time: Dict[EventType, u.quantity.Quantity[u.year]],
        atmo_flux_energy_points: int = 100,
        atmo_flux_theta_points: int = 30,
        n_grid_points: int = 50,
        N: dict = {},
        asimov: bool = False,
    ):
        """
        To set up and run simulations.
        :param sources: Sources instance
        :param event_types: EventType or List thereof, to be included in the fit
        :param observation_time: astropy.units time for single event type or dictionary thereof with event type as key
        :param atmo_flux_energy_points: number of points for atmo spectrum energy interpolation
        :param atmo_flux_theta_points: number of points for atmo spectrum cos(theta) interpolation
        :param n_grid_points: number of grid points used per parameter in precomputation of exposure
        :param N: dict with EventType as key and list as entry, to force simulation of specific event numbers,
            e.g. {IC86_II: [1, 2, 3]} for a single season and 3 source components.
        :param asimov: set to True to simulate closest integer of expected number of events.
        """

        super().__init__(sources)
        if not isinstance(event_types, list):
            event_types = [event_types]
        if not isinstance(observation_time, dict):
            observation_time = {event_types[0]: observation_time}
        if not len(event_types) == len(observation_time):
            raise ValueError(
                "number of observation times must match number of event types"
            )
        self._event_types = event_types
        self._observation_time = observation_time
        self._n_grid_points = n_grid_points
        self._asimov = asimov

        self._exposure_integral = collections.OrderedDict()

        if asimov:
            N = {}
            for event_type in self._event_types:
                # dummy values
                N[event_type] = [1] * self._sources.N

        if N:
            for event_type in self._event_types:
                if not len(N[event_type]) == len(sources):
                    raise ValueError(
                        "Provided event numbers must match number of sources"
                    )

                if CAS in self._event_types and self._sources.atmospheric:
                    if N[CAS][-1] != 0:
                        sim_logger.warning("Setting atmospheric cascade events to zero")
                        N[CAS][-1] = 0

            self._force_N = True
            self._N = N

        else:
            self._force_N = False

        stan_file_name = os.path.join(STAN_GEN_PATH, "sim_code")

        self._stan_interface = StanSimInterface(
            stan_file_name,
            self._sources,
            self._event_types,
            atmo_flux_energy_points=atmo_flux_energy_points,
            atmo_flux_theta_points=atmo_flux_theta_points,
            force_N=self._force_N,
        )

        # Silence log output
        logger = logging.getLogger("hierarchical_nu.backend.code_generator")
        logger.propagate = False

        # Check for unsupported combinations
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

        self.events = None

    @property
    def expected_Nnu_per_comp(self):

        sim_inputs = self._get_sim_inputs()
        self._get_expected_Nnu(sim_inputs)

        return self._expected_Nnu_per_comp

    @property
    def Nex_et(self):

        sim_inputs = self._get_sim_inputs()
        self._get_expected_Nnu(sim_inputs)

        return self._Nex_et

    @property
    def sources(self):
        return self._sources

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

    def compute_c_values(self, inplace: bool = False):
        """
        Method to re-compute all envelopes for rejection sampling,
        necessary for PPCs when spectral parameters are changed.
        """

        for eps in self._exposure_integral.values():
            eps._compute_c_values(inplace=inplace)

    def generate_stan_code(self):
        """
        Generate stan code from scratch
        """

        self._main_sim_filename = self._stan_interface.generate()

    def set_stan_filename(self, sim_filename):
        """
        Set stan file name for existing simulation code
        :param sim_filename: filename of stan code
        """

        self._main_sim_filename = sim_filename

    def compile_stan_code(self, include_paths=None):
        """
        Compile stan simulation code
        :param include_paths: list of directories to include stan files from
        """

        if not include_paths:
            include_paths = [STAN_PATH, STAN_GEN_PATH]

        stanc_options = {"include-paths": include_paths}

        self._main_sim = CmdStanModel(
            stan_file=self._main_sim_filename,
            stanc_options=stanc_options,
        )

    def setup_stan_sim(self, exe_file: Union[str, Path] = ".stan_files/sim_code"):
        """
        Reuse previously compiled model
        :param exe_file: Path to compiled stan file
        """

        self._main_sim = CmdStanModel(exe_file=exe_file)

    def run(self, seed=None, verbose=False, **kwargs):
        """
        Run the simulation.
        :param seed: random seed
        :param verbose: if True, print debug messages
        :param kwargs: kwargs passed to `cmdstanpy.CmdStanModel.sample()`
        """

        self._sim_inputs = self._get_sim_inputs(seed)
        self._expected_Nnu = self._get_expected_Nnu(self._sim_inputs)

        if self._asimov:
            # Override sim inputs with asimov-forced_N
            self._sim_inputs = self._get_sim_inputs(seed, asimov=True)

        # Create data field in sim inputs to handle number of expected events for each source component
        # self._Nex_et has all necessary information, dimension (detector models, source components)
        # In case of data being used as background, cut out the corresponding zero-entries
        if self.sources.background:
            Nex_et = self.Nex_et[:, :-1]
        else:
            Nex_et = self.Nex_et
        self._sim_inputs["Nex_et"] = Nex_et.tolist()

        skip = True if (self._expected_Nnu < 0.5 and self._asimov) else False
        if verbose:
            if skip:
                sim_logger.info("Asimov simulation with 0 events, skipping simulation.")
            else:
                sim_logger.info(
                    "Running a simulation with expected Nnu = %.2f events"
                    % self._expected_Nnu
                )

        if not skip:

            sim_output = self._main_sim.sample(
                data=self._sim_inputs,
                iter_sampling=1,
                chains=1,
                fixed_param=True,
                seed=seed,
                **kwargs,
            )

            self._sim_output = sim_output

            energies, coords, event_types, ang_errs = self._extract_sim_output()

            # Create filler MJD values, we are only doing time-averaged simulations
            mjd = Time([99.0] * len(energies), format="mjd")

            # Check for detected events
            if len(energies) != 0:
                self.events = Events(energies, coords, event_types, ang_errs, mjd)
            else:
                self.events = None
        else:
            self.events = None

    def _extract_sim_output(self):
        try:
            energies = self._sim_output.stan_variable("Edet")[0] * u.GeV
            dirs = self._sim_output.stan_variable("event")[0]
            coords = SkyCoord(
                dirs.T[0],
                dirs.T[1],
                dirs.T[2],
                representation_type="cartesian",
                frame="icrs",
            )
            event_types = self._sim_output.stan_variable("event_type")[0]
            event_types = [int(_) for _ in event_types]

            # Kappa parameter of VMF distribution
            kappa = self._sim_output.stan_variable("kappa")[0]
            # Equivalent 1 sigma errors in deg
            ang_errs = get_theta_p(kappa, p=0.683) * u.deg

        except ValueError:
            # No detected events
            energies = [] * u.GeV
            coords = []
            event_types = []
            ang_errs = [] * u.deg

        return energies, coords, event_types, ang_errs

    def save(self, path, overwrite: bool = False):
        """
        Save simulation
        :param path: filename of simulation, should have extension `.h5`
        :param overwrite: if True, overwrite files with identical name
        """

        # Check if filename consists of a path to some directory as well as the filename
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        if dirname:
            if not os.path.exists(dirname):
                sim_logger.warning(
                    f"{dirname} does not exist, saving instead to {os.getcwd()}"
                )
                dirname = os.getcwd()
        else:
            dirname = os.getcwd()
        path = Path(dirname) / Path(filename)

        if os.path.exists(path) and not overwrite:
            sim_logger.warning(f"File {filename} already exists.")
            file = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            file += f"_{int(thyme())}"
            filename = file + ext

        path = Path(dirname) / Path(filename)

        with h5py.File(path, "w") as f:
            sim_folder = f.create_group("sim")

            inputs_folder = sim_folder.create_group("inputs")
            for key, value in self._sim_inputs.items():
                inputs_folder.create_dataset(key, data=value)

            outputs_folder = sim_folder.create_group("outputs")
            N = len(self._sim_output.stan_variable("Edet")[0])
            for key, value in self._sim_output.stan_variables().items():
                if key == "event":
                    reshaped_events = value[0].reshape((3, N)).T
                    outputs_folder.create_dataset(key, data=reshaped_events)

                else:
                    outputs_folder.create_dataset(key, data=value[0])

            source_folder = sim_folder.create_group("source")
            flux_unit = 1 / (u.m**2 * u.s)
            source_folder.create_dataset(
                "total_flux_int",
                data=self._sources.total_flux_int().to(flux_unit).value,
            )

            outputs_folder.create_dataset(
                "expected_Nnu_per_comp", data=self._expected_Nnu_per_comp
            )
            f.create_dataset("version", data=git_hash)

        self.events.to_file(path, append=True)

        return path  # noqa: F821

    def show_spectrum(
        self, *components: str, scale: str = "linear", population: bool = False
    ):
        """
        Show binned spectrum of simulated data
        :param components: not used? what is this? # TODO fix
        :param scale: either `linear` or `log` to change y-axis scale of histograms
        :param population: if True, display all point sources as one entry
        """

        hatch_cycle = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
        Esrc = self._sim_output.stan_variable("Esrc")[0]
        E = self._sim_output.stan_variable("E")[0]
        lam = self._sim_output.stan_variable("Lambda")[0] - 1
        if not np.all(Esrc >= E):
            sim_logger.critical(
                "Some event has lower energy in its source than in the detector frame"
            )
        Edet = self.events.energies.value
        Emin_det = self._get_min_det_energy().to(u.GeV).value

        N = len(self._sources)
        N_ps = len(self._sources.point_source)

        if not population:
            Esrc_plot = [Esrc[np.nonzero(lam == float(i))] for i in range(N)]
            E_plot = [E[np.nonzero(lam == float(i))] for i in range(N)]
            Edet_plot = [Edet[np.nonzero(lam == float(i))] for i in range(N)]
        else:
            # compress all PSs into one entry
            Esrc_plot = [
                np.concatenate(
                    [Esrc[np.nonzero(lam == float(i))] for i in range(N_ps)]
                ),
                Esrc[np.nonzero(lam == float(N_ps))],
                Esrc[np.nonzero(lam == float(N_ps + 1))],
            ]
            E_plot = [
                np.concatenate([E[np.nonzero(lam == float(i))] for i in range(N_ps)]),
                E[np.nonzero(lam == float(N_ps))],
                E[np.nonzero(lam == float(N_ps + 1))],
            ]
            Edet_plot = [
                np.concatenate(
                    [Edet[np.nonzero(lam == float(i))] for i in range(N_ps)]
                ),
                Edet[np.nonzero(lam == float(N_ps))],
                Edet[np.nonzero(lam == float(N_ps + 1))],
            ]

        bins = np.logspace(
            np.log10(Emin_det),
            np.log10(Parameter.get_parameter("Emax").value.to(u.GeV).value),
            20,
            base=10,
        )

        if not population:
            sources = self._sources
        else:
            sources = [self._sources[0]]
            if self._sources.diffuse:
                sources.append(self._sources.diffuse)
            if self._sources.atmospheric:
                sources.append(self._sources.atmospheric)

        fig, ax = plt.subplots(3, 1)

        for c, (source, hatch, _Esrc, _E, _Edet) in enumerate(
            zip(sources, hatch_cycle, Esrc_plot, E_plot, Edet_plot)
        ):
            if c == 0:
                _bsrc = np.zeros(bins[:-1].shape)
                if not population:
                    label = source.name + " at source"
                else:
                    label = "population at source"
                # This is only needed s.t. flake does not complain
                _nEsrc = 0
            else:
                _bsrc += _nEsrc
                label = source.name

            _nEsrc, _, _ = ax[0].hist(
                _Esrc, bins=bins, label=label, bottom=_bsrc, alpha=0.5, hatch=hatch
            )

            if c == 0:
                _b = np.zeros(bins[:-1].shape)
                if not population:
                    label = source.name + " at detector"
                else:
                    label = "population at detector"
                _nE = 0
            else:
                _b += _nE
                label = source.name
            _nE, _, _ = ax[1].hist(
                _E, bins=bins, label=label, bottom=_b, alpha=0.5, hatch=hatch
            )

            if c == 0:
                _bdet = np.zeros(bins[:-1].shape)
                if not population:
                    label = source.name + ", detected"
                else:
                    label = "population, detected"
                _nEdet = 0
            else:
                _bdet += _nEdet
                label = source.name
            _nEdet, _, _ = ax[2].hist(
                _Edet, bins=bins, label=label, bottom=_bdet, alpha=0.5, hatch=hatch
            )

        for a in ax:
            a.set_xscale("log")
            a.set_yscale(scale)
            a.set_xlabel("E")
            a.legend()

        return fig, ax

    def show_skymap(
        self,
        track_zoom: float = 1.0,
        subplot_kw: dict = {"projection": "astro degrees mollweide"},
        population: bool = False,
    ):
        """
        :param track_zoom: Increase radius of track events by this factor for visibility
        :param subplot_kw: Customise projection style and boundaries with ligo.skymap
        :param population: if True, display all point sources as one entry
        """

        try:
            lam = list(
                self._sim_output.stan_variable("Lambda")[0] - 1
            )  # avoid Stan-style indexing
        except AttributeError:
            lam = list(self._sim_output.stan_variable["Lambda"][0] - 1)
        Ns = self._sim_inputs["Ns"]

        # Reduce the amount of labels in the legend and various colors for large populations
        # plot all point sources in one colour if so desired
        if population:
            label_cmap = plt.cm.Set1(list(range(self._sources.N - Ns + 1)))
        else:
            label_cmap = plt.cm.Set1(list(range(self._sources.N)))

        N_src_ev = sum([lam.count(_) for _ in range(Ns)])

        if self._sources.atmospheric and not self._sources.diffuse:
            N_bg_ev = 0
            N_atmo_ev = lam.count(Ns)

        else:
            N_bg_ev = lam.count(Ns)
            N_atmo_ev = lam.count(Ns + 1)

        fig, ax = plt.subplots(subplot_kw=subplot_kw)
        fig.set_size_inches((7, 5))

        self.events.coords.representation_type = "spherical"
        for r, d, l, e, t in zip(
            self.events.coords.icrs.ra,
            self.events.coords.icrs.dec,
            lam,
            self.events.ang_errs,
            self.events.types,
        ):
            if not population:
                color = label_cmap[int(l)]
            elif l == Ns:
                color = label_cmap[1]
            elif l == Ns + 1:
                color = label_cmap[2]
            else:
                color = label_cmap[0]

            if t == NT.S:
                e = e * track_zoom  # to make tracks visible

            circle = SphericalCircle(
                (r, d),
                e,
                color=color,
                alpha=0.5,
                transform=ax.get_transform("icrs"),
            )

            ax.add_patch(circle)

        fig.suptitle(
            "N_src_events = %i, N_bg_events = %i, N_atmo_events = %i"
            % (N_src_ev, N_bg_ev, N_atmo_ev),
            y=0.85,
        )
        # fig.tight_layout()

        return fig, ax

    def setup_and_run(self, include_paths=None):
        """
        Wrapper around setup functions for convenience.
        """

        self.precomputation()
        self.generate_stan_code()
        self.compile_stan_code(include_paths=include_paths)
        self.run()

    def _get_sim_inputs(self, seed=None, asimov=False):
        sim_inputs = {}

        redshift = [
            s.redshift
            for s in self._sources.sources
            if isinstance(s, PointSource)
            or isinstance(s.flux_model, IsotropicDiffuseBG)
        ]
        if self._sources.point_source:
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
            sim_inputs["D"] = D
            sim_inputs["varpi"] = src_pos
            sim_inputs["Ns"] = len(
                [s for s in self._sources.sources if isinstance(s, PointSource)]
            )
        else:
            sim_inputs["Ns"] = 0

        sim_inputs["z"] = redshift

        integral_grid = []
        integral_grid_2d = []
        atmo_integ_val = []
        rs_breaks = []
        rs_slopes = []
        rs_weights = []
        rs_N = []
        rs_norms = []
        Emin_det = []
        forced_N = []
        obs_time = []

        sim_inputs["Ngrid"] = self._n_grid_points

        sim_inputs["Emin"] = Parameter.get_parameter("Emin").value.to_value(u.GeV)
        sim_inputs["Emax"] = Parameter.get_parameter("Emax").value.to_value(u.GeV)

        if sim_inputs["Emin"] < 1e2:
            raise ValueError("Emin is lower than detector minimum energy")
        if sim_inputs["Emax"] > 1e9:
            raise ValueError("Emax is higher than detector maximum energy")

        if asimov:
            # Round expected number of events to nearest integer per source
            # distribute this number weighted with the Nex per event type over the event types
            N = np.rint(self._Nex_et.sum(axis=0)).astype(int)   # total number of events to be observed
            if not self.sources.background:
                self._N = np.zeros_like(self._Nex_et)
            else:
                self._N = np.zeros_like(self._Nex_et[:, :-1])
            for c, source in enumerate(self._sources):
                if isinstance(source, BackgroundSource):
                    break
                weights = self._Nex_et[:, c] / self._Nex_et[:, c].sum()

                if np.any(np.isnan(weights)):
                    N[:, c] = 0
                    continue

                # Sample et_idx for each source,
                # takes asimov-number of events separately for each source
                et_idx = np.random.choice(
                    range(len(self._event_types)),
                    p=weights,
                    size=np.rint(self._Nex_et[:, c].sum()).astype(int),
                )

                for i, et in enumerate(self._event_types):
                    # Count number at each idx
                    self._N[i, c] = np.count_nonzero(et_idx == i)
            N = {}
            for i, et in enumerate(self._event_types):
                N[et] = self._N[i].astype(int).tolist()

            self._N = N

        if self._sources.point_source:
            # This is copied from fit.py. While there is nothing being fit in a simulation,
            # for internal consistency the same approach of calculating Nex inside stan is used.
            # That means we need to check over which parameters is interpolated in the 1D or 2D grid,
            # hence we check which parameters would be free in a fit.

            lumi_units = u.GeV / u.s

            # Change this to check for pressure ratio first if we are using a Seyfert source
            if self._seyfert:
                try:
                    sim_inputs["pressure_ratio"] = [
                        Parameter.get_parameter("pressure_ratio").value
                    ] * len(self._sources.point_source)
                except ValueError:
                    sim_inputs["pressure_ratio"] = [
                        Parameter.get_parameter("%s_pressure_ratio" % s.name)
                        for s in self._sources.point_source
                    ]
                sim_inputs["L"] = [np.nan] * len(self._sources.point_source)
            else:
                try:
                    sim_inputs["L"] = [
                        Parameter.get_parameter(
                            "%s_luminosity" % s.name
                        ).value.to_value(lumi_units)
                        for s in self._sources.point_source
                    ]
                # If the individual parameters are not found we have a global luminosity
                except ValueError:
                    try:
                        sim_inputs["L"] = [
                            Parameter.get_parameter("luminosity").value.to_value(
                                lumi_units
                            )
                        ] * len(self._sources.point_source)
                    except ValueError:
                        sim_inputs["L"] = [np.nan] * len(self._sources.point_source)

            # Check for shared source index
            if self._shared_src_index:
                key = "src_index"
                key_beta = "beta_index"
                key_Enorm = "E0_src"
                key_eta = "eta"

            # Otherwise just use first source in the list
            # grids is identical for all point sources
            else:
                key = "%s_src_index" % self._sources.point_source[0].name
                key_beta = "%s_beta_index" % self._sources.point_source[0].name
                key_Enorm = "%s_E0_src" % self._sources.point_source[0].name
                key_eta = "%s_eta" % self.sources.point_source[0].name

            event_type = self._event_types[0]
            # This is a weird construct
            if self._fit_index:
                sim_inputs["src_index_grid"] = self._exposure_integral[
                    event_type
                ].par_grids[key]
            if self._logparabola or self._fit_index:
                sim_inputs["src_index"] = [
                    ps.flux_model.parameters["index"].value
                    for ps in self._sources.point_source
                ]
            if self._fit_beta:
                sim_inputs["beta_index_grid"] = self._exposure_integral[
                    event_type
                ].par_grids[key_beta]
            if self._logparabola:
                sim_inputs["beta_index"] = [
                    ps.flux_model.parameters["beta"].value
                    for ps in self._sources.point_source
                ]
            if self._fit_Enorm:
                sim_inputs["E0_src_grid"] = self._exposure_integral[
                    event_type
                ].par_grids[key_Enorm]
            if self._fit_Enorm or self._logparabola:
                sim_inputs["E0_src"] = [
                    ps.flux_model.parameters["norm_energy"].value.to_value(u.GeV)
                    for ps in self._sources.point_source
                ]
            if self._fit_eta:
                sim_inputs["eta_grid"] = self._exposure_integral[event_type].par_grids[
                    key_eta
                ]
            if self._pgamma:
                sim_inputs["src_index"] = [
                    ps.flux_model.spectral_shape._src_index
                    for ps in self._sources.point_source
                ]
            if self._seyfert:
                sim_inputs["eta"] = [
                    ps.flux_model.parameters["eta"].value
                    for ps in self._sources.point_source
                ]

            sim_inputs["Emin_src"] = [
                ps.frame.transform(
                    Parameter.get_parameter("Emin_src").value, ps.redshift
                ).to_value(u.GeV)
                for ps in self._sources.point_source
            ]
            sim_inputs["Emax_src"] = [
                ps.frame.transform(
                    Parameter.get_parameter("Emax_src").value, ps.redshift
                ).to_value(u.GeV)
                for ps in self._sources.point_source
            ]

            if np.min(sim_inputs["Emin_src"]) < sim_inputs["Emin"]:
                raise ValueError(
                    "Minimum source energy may not be lower than minimum energy overall"
                )
            if np.max(sim_inputs["Emax_src"]) > sim_inputs["Emax"]:
                raise ValueError(
                    "Maximum source energy may not be higher than maximum energy overall"
                )

        if self._sources.diffuse:
            # Same as for point sources
            sim_inputs["Ngrid"] = len(
                self._exposure_integral[self._event_types[0]].par_grids["diff_index"]
            )

            sim_inputs["diff_index_grid"] = self._exposure_integral[
                self._event_types[0]
            ].par_grids["diff_index"]

            sim_inputs["diff_index"] = Parameter.get_parameter("diff_index").value

            sim_inputs["Emin_diff"] = self._sources.diffuse.frame.transform(
                Parameter.get_parameter("Emin_diff").value,
                self._sources.diffuse.redshift,
            ).to_value(u.GeV)
            sim_inputs["Emax_diff"] = self._sources.diffuse.frame.transform(
                Parameter.get_parameter("Emax_diff").value,
                self._sources.diffuse.redshift,
            ).to_value(u.GeV)

            if sim_inputs["Emin_diff"] < sim_inputs["Emin"]:
                raise ValueError(
                    "Minimum diffuse energy may not be lower than minimum energy overall"
                )
            if sim_inputs["Emax_diff"] > sim_inputs["Emax"]:
                raise ValueError(
                    "Maximum diffuse energy may not be higher than maximum energy overall"
                )

        for c, event_type in enumerate(self._event_types):
            obs_time.append(self._observation_time[event_type].to(u.s).value)

            try:
                Emin_det.append(
                    Parameter.get_parameter("Emin_det").value.to_value(u.GeV)
                )

            except ValueError:
                Emin_det.append(
                    Parameter.get_parameter(f"Emin_det_{event_type.P}").value.to_value(
                        u.GeV
                    )
                )

            # Rejection sampling
            # Loop over detector models
            # loop over source components
            container = self._exposure_integral[event_type]._envelope_container
            rs_norms.append([])
            rs_breaks.append([])
            rs_slopes.append([])
            rs_weights.append([])
            rs_N.append([])
            for c_s in range(self._sources.N):
                # Do not normalise, the target is not normalised either and
                # that is what we want to approximate
                if isinstance(self._sources[c_s], BackgroundSource):
                    break
                norms = container[c_s].low_values
                rs_norms[-1].append(norms.tolist())
                rs_breaks[-1].append(container[c_s].bins.tolist())
                rs_slopes[-1].append(container[c_s].slopes.tolist())
                rs_weights[-1].append(container[c_s].weights.tolist())
                rs_N[-1].append(container[c_s].N)

            if self._force_N:
                forced_N.append(self._N[event_type])

            integral_grid.append([])
            integral_grid_2d.append([])

            for grid in self._exposure_integral[event_type].integral_grid:

                if len(grid.shape) == 2:
                    integral_grid_2d[-1].append(np.log(grid.to_value(u.m**2)).tolist())

                else:
                    integral_grid[-1].append(np.log(grid.to_value(u.m**2)).tolist())

            if self._sources.atmospheric:
                atmo_integ_val.append(
                    self._exposure_integral[event_type]
                    .integral_fixed_vals[0]
                    .to(u.m**2)
                    .value
                )

        # Fill the various rs arrays with zero entries in the end
        # because stan doesn't support ragged structures
        # that might pop up if we use different bin sizes,
        # energy ranges etc for different spectra
        rs_N_max = np.max(rs_N)
        for c, et in enumerate(self._event_types):
            for c_s in range(self._sources.N):
                if isinstance(self._sources[c_s], BackgroundSource):
                    break
                # breaks has length rs_N_max + 1
                while len(rs_breaks[c][c_s]) < rs_N_max + 1:
                    rs_breaks[c][c_s].append(0)
                while len(rs_norms[c][c_s]) < rs_N_max:
                    rs_norms[c][c_s].append(0)
                while len(rs_slopes[c][c_s]) < rs_N_max:
                    rs_slopes[c][c_s].append(0)
                while len(rs_weights[c][c_s]) < rs_N_max:
                    rs_weights[c][c_s].append(0)

        try:
            ROIList.STACK[0]
        except IndexError:
            raise ValueError("An ROI is needed at this point.")

        v_lim_low = (np.cos(-ROIList.DEC_min().to_value(u.rad) + np.pi / 2) + 1.0) / 2
        v_lim_high = (np.cos(-ROIList.DEC_max().to_value(u.rad) + np.pi / 2) + 1.0) / 2

        if NT in self._event_types:
            # acos(2 * v - 1) = theta -> v = cos(theta) + 1 / 2
            # v from 0 to 1
            # Only sample from Northern hemisphere

            # theta from 0 (north) to pi (south), dec from pi/2 (north) to -pi/2 (south)
            # theta = -dec + pi/2
            cz_max = max(self._exposure_integral[NT].effective_area._cosz_bin_edges)
            v_lim_low_detector = ((np.cos(np.pi - np.arccos(cz_max)) + 1) / 2) + 1e-2

            if v_lim_low_detector > v_lim_low:
                v_lim_low = v_lim_low_detector

            if not v_lim_high > v_lim_low:
                raise ValueError("")

        sim_inputs["v_low"] = v_lim_low
        sim_inputs["v_high"] = v_lim_high
        if len(ROIList.STACK) > 1:
            sim_inputs["u_low"] = 0.0
            sim_inputs["u_high"] = 1.0
        else:
            # Finding the most efficient RA range for multiple ROIs is not implemented yet
            sim_inputs["u_low"] = ROIList.STACK[0].RA_min.to_value(u.rad) / (
                2.0 * np.pi
            )
            sim_inputs["u_high"] = ROIList.STACK[0].RA_max.to_value(u.rad) / (
                2.0 * np.pi
            )

        # For circular ROI the center point and radius are needed
        if isinstance(ROIList.STACK[0], CircularROI):
            radii = [_.radius.to_value(u.rad) for _ in ROIList.STACK]
            sim_inputs["roi_radius"] = radii
            centers = []
            for roi in ROIList.STACK:
                roi.center.representation_type = "cartesian"
                centers.append(np.array([roi.center.x, roi.center.y, roi.center.z]))
            sim_inputs["roi_center"] = centers

        flux_units = 1 / (u.m**2 * u.s)

        if self._sources.diffuse:
            diffuse_bg = self._sources.diffuse
            sim_inputs["F_diff"] = diffuse_bg.flux_model.total_flux_int.to_value(
                flux_units
            )

        if self._sources.atmospheric:
            # Parameter F_atmo is created when adding atmo to the source list
            sim_inputs["F_atmo"] = Parameter.get_parameter("F_atmo").value.to_value(
                flux_units
            )

        sim_inputs["integral_grid"] = integral_grid
        sim_inputs["integral_grid_2d"] = integral_grid_2d
        if self._sources.atmospheric:
            sim_inputs["atmo_integ_val"] = atmo_integ_val
        sim_inputs["Emin_det"] = Emin_det
        sim_inputs["rs_N"] = rs_N
        sim_inputs["rs_breaks"] = rs_breaks
        sim_inputs["rs_weights"] = rs_weights
        sim_inputs["rs_slopes"] = rs_slopes
        sim_inputs["rs_norms"] = rs_norms
        sim_inputs["rs_N_max"] = rs_N_max
        sim_inputs["T"] = obs_time
        if self._force_N:
            sim_inputs["forced_N"] = forced_N

        sim_inputs["event_types"] = [_.S for _ in self._event_types]

        # Remove np.ndarrays for use with cmdstanpy
        sim_inputs = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in sim_inputs.items()
        }

        return sim_inputs

    @property
    def expected_Nnu(self):
        return self._get_expected_Nnu(self._get_sim_inputs())

    @property
    def Nex_et(self):
        self._get_expected_Nnu(self._get_sim_inputs())
        return self._Nex_et

    def _get_expected_Nnu(self, sim_inputs):
        """
        Calculates expected number of neutrinos to be simulated.
        Uses same approach as in the Stan code for cross-checks.
        :param sim_inputs: simulation input dictionary
        """

        sim_inputs_ = sim_inputs.copy()

        if self._logparabola:
            spectrum = "logparabola"
        elif self._pgamma:
            spectrum = "pgamma"
        elif self._power_law:
            spectrum = "power_law"
        elif self._seyfert:
            spectrum = "seyfert"
        else:
            spectrum = "none"

        fit_params = []
        if self._fit_index:
            fit_params.append("index")
        if self._fit_beta:
            fit_params.append("beta_index")
        if self._fit_Enorm:
            fit_params.append("E0_src")
        if self._fit_eta:
            fit_params.append("eta")

        Nex_et = np.zeros((len(self._event_types), self._sources.N))
        for c, event_type in enumerate(self._event_types):
            integral_grid = sim_inputs_["integral_grid"][c]
            integral_grid_2d = sim_inputs_["integral_grid_2d"][c]
            try:
                flux_conv_ = self._sources.point_source_spectrum.flux_conv_
            except (ValueError, AttributeError):
                # In this case flux_conv_ is not used
                flux_conv_ = lambda x: x
            Nex_et[c] = _get_expected_Nnu_(
                c,
                sim_inputs_,
                flux_conv_,
                integral_grid,
                integral_grid_2d,
                spectrum,
                fit_params,
                bool(self._sources.point_source),
                self._sources.diffuse,
                self._sources.atmospheric,
                self._shared_luminosity,
                self._sources,
                self._bg,
            )

        self._Nex_et = Nex_et
        # Nnu per source source component
        self._expected_Nnu_per_comp = np.sum(self._Nex_et, axis=0)

        return np.sum(self._Nex_et)

    @classmethod
    def from_file(cls, filename):
        raise NotImplementedError()

    def _get_min_det_energy(self):
        """
        Check for different definitions of minimum detected
        energy in parameter settings and return relevant
        value.

        This is necessary as it is possible to specify Emin_det
        or (Emin_det_tracks, Emin_det_cascades).
        """

        try:
            Emin_det = Parameter.get_parameter("Emin_det").value

        except ValueError:
            Edets = []
            for et in self._event_types:
                Edets.append(Parameter.get_parameter(f"Emin_det_{et.P}").value)

            Emin_det = min(Edets)

        return Emin_det


class SimInfo:
    def __init__(self, truths, inputs, outputs):
        """
        To store and reference simulation inputs/info.

        TODO: instead work on Simualtion.from_file() method
        to fully load simulation from output file.
        """

        self.truths = truths

        self.inputs = inputs

        self.outputs = outputs

    @classmethod
    def from_file(cls, filename):
        inputs = {}
        outputs = {}
        with h5py.File(filename, "r") as f:
            inputs_folder = f["sim/inputs"]
            source_folder = f["sim/source"]
            outputs_folder = f["sim/outputs"]

            input_keys = list(f["sim/inputs"].keys())

            atmo = False
            diff = False
            ps = False
            pgamma = False
            powerlaw = False
            logparabola = False
            seyfert = False
            for key in inputs_folder:
                inputs[key] = inputs_folder[key][()]

                if key == "F_atmo":
                    atmo = True

                if key == "F_diff":
                    diff = True

                if key == "L":
                    ps = True

            if "E0_src" and "beta_index" in input_keys:
                logparabola = True
            elif "E0_src" in input_keys:
                pgamma = True
            elif "eta" in input_keys:
                seyfert = True
            else:
                powerlaw = True

            for key in source_folder:
                inputs[key] = source_folder[key][()]

            for key in outputs_folder:
                outputs[key] = outputs_folder[key][()]

        truths = {}

        if ps:
            try:
                truths["L"] = inputs["L"]
            except KeyError:
                truths["P"] = inputs["P"]
            if powerlaw or logparabola:
                truths["src_index"] = inputs["src_index"]
            if logparabola:
                truths["beta_index"] = inputs["beta_index"]
            if pgamma or logparabola:
                truths["E0_src"] = inputs["E0_src"]
            if seyfert:
                truths["eta"] = inputs["eta"]

        if diff:
            truths["F_diff"] = inputs["F_diff"]
            truths["diff_index"] = inputs["diff_index"]

        if atmo:
            truths["F_atmo"] = inputs["F_atmo"]
        try:
            truths["Ftot"] = inputs["total_flux_int"]
            truths["f_arr"] = outputs["f_arr"]
            truths["f_arr_astro"] = outputs["f_arr_astro"]
            truths["f_det"] = outputs["f_det"]
            truths["f_det_astro"] = outputs["f_det_astro"]
        except KeyError:
            pass

        return cls(truths, inputs, outputs)

    @classmethod
    def merge(cls, background, point_source, output):
        """
        Check if two simulations are compatible to merge and do so
        Intended to merge pure background and pure point source simulations
        Does not recalculate all derived quantities, e.g. fractional fluxes
        :param background: path to background simulation
        :param point_source: path to point source simulation
        :param output: path at which merged simulation will be saved
        """

        # Meow
        from astropy.coordinates import concatenate as cat
        from astropy.time import Time

        bg = cls.from_file(background)
        ps = cls.from_file(point_source)

        bg_events = Events.from_file(
            background,
            apply_Emin_det=False,
            apply_spatial_cuts=False,
            apply_temporal_cuts=False,
        )
        ps_events = Events.from_file(
            point_source,
            apply_Emin_det=False,
            apply_spatial_cuts=False,
            apply_temporal_cuts=False,
        )

        energies = np.hstack((ps_events.energies, bg_events.energies))
        coords = cat((ps_events.coords, bg_events.coords))
        types = np.hstack((ps_events.types, bg_events.types))
        mjd = np.hstack((ps_events.mjd.to_value("mjd"), bg_events.mjd.to_value("mjd")))
        mjd = Time(mjd, format="mjd")
        ang_errs = np.hstack((ps_events.ang_errs, bg_events.ang_errs))

        # Create common event objects
        events = Events(energies, coords, types, ang_errs, mjd)

        # Truths keys
        ps_keys = ["L", "src_index", "beta_index", "E0_src", "eta", "pressure_ratio"]
        bg_keys = ["F_atmo", "F_diff", "diff_index"]

        truths = {}
        for key in ps_keys:
            try:
                truths[key] = ps.truths[key]
            except KeyError:
                continue
        for key in bg_keys:
            truths[key] = bg.truths[key]

        # Need to merge Lambda values for true associations
        # Keep to usual order: first point sources, diffuse astro, diffuse atmo
        Lambdas_ps = ps.outputs["Lambda"]
        N_ps = ps.inputs["D"].size
        Lambdas_bg = bg.outputs["Lambda"]
        Lambdas_bg_for_merging = np.zeros_like(Lambdas_bg)
        bg_components = np.unique(bg.outputs["Lambda"]).size
        for i in range(1, bg_components + 1):
            Lambdas_bg_for_merging[Lambdas_bg == i] = N_ps + i

        Lambdas = np.hstack((Lambdas_ps, Lambdas_bg_for_merging))

        try:
            ps.inputs["forced_N"]
            ps_forced_N = True
        except KeyError:
            ps_forced_N = False

        try:
            bg.inputs["forced_N"]
            bg_forced_N = True
        except KeyError:
            bg_forced_N = False

        with h5py.File(output, "w") as f:
            # Dummy folders
            sim_folder = f.create_group("sim")
            source_folder = sim_folder.create_group("source")
            inputs_folder = sim_folder.create_group("inputs")
            for key in ps_keys:
                try:
                    inputs_folder.create_dataset(key, data=ps.inputs[key])
                except KeyError:
                    continue
            for key in bg_keys:
                inputs_folder.create_dataset(key, data=bg.inputs[key])
            if ps_forced_N and bg_forced_N:
                inputs_folder.create_dataset(
                    "forced_N",
                    data=np.hstack((ps.inputs["forced_N"], bg.inputs["forced_N"])),
                )
            elif ps_forced_N:
                inputs_folder.create_dataset("forced_N", data=ps.inputs["forced_N"])
            outputs_folder = sim_folder.create_group("outputs")
            for key, value in truths.items():
                outputs_folder.create_dataset(key, data=value)
            outputs_folder.create_dataset("Lambda", data=Lambdas)

        events.to_file(output, append=True)


def _get_expected_Nnu_(
    c,
    sim_inputs,
    flux_conv_,
    integral_grid,
    integral_grid_2d,
    spectrum,
    fit_params,
    point_source=False,
    diffuse=False,
    atmospheric=False,
    shared_luminosity=True,
    sources=None,
    data_bg=False,
):
    """
    Helper function for calculating expected Nnu
    using stan sim_inputs.
    """

    n_params = 0
    if point_source:
        if spectrum == "logparabola":
            beta_index = sim_inputs["beta_index"]
            E0_src = sim_inputs["E0_src"]
            src_index = sim_inputs["src_index"]
        elif spectrum == "power_law":
            src_index = sim_inputs["src_index"]
        elif spectrum == "pgamma":
            E0_src = sim_inputs["E0_src"]
            # use len(D) for number of point sources
            src_index = [0.0] * len(sim_inputs["D"])
        elif spectrum == "seyfert":
            eta = sim_inputs["eta"]

        if "index" in fit_params:
            src_index_grid = sim_inputs["src_index_grid"]
            n_params += 1
            fit_index = True
        else:
            fit_index = False
        if "beta_index" in fit_params:
            beta_index_grid = sim_inputs["beta_index_grid"]
            n_params += 1
            fit_beta = True
        else:
            fit_beta = False
        if "E0_src" in fit_params:
            E0_src_grid = sim_inputs["E0_src_grid"]
            n_params += 1
            fit_Enorm = True
        else:
            fit_Enorm = False
        if "eta" in fit_params:
            n_params += 1
            fit_eta = True
        else:
            fit_eta = False
    if diffuse:
        diff_index = sim_inputs["diff_index"]
        diff_index_grid = sim_inputs["diff_index_grid"]

    Ns = sim_inputs["Ns"]

    F = []
    eps = []

    if point_source:
        for i, (d, Emin_src, Emax_src) in enumerate(
            zip(sim_inputs["D"], sim_inputs["Emin_src"], sim_inputs["Emax_src"])
        ):

            if n_params == 2:
                first = True
                if fit_index:
                    first_grid = src_index_grid
                    first_param = src_index[i]
                    first = False
                if fit_beta:
                    if first:
                        first_grid = beta_index_grid
                        first_param = beta_index[i]
                        first = False
                    else:
                        second_grid = beta_index_grid
                        second_param = beta_index[i]
                if fit_Enorm:
                    second_grid = E0_src_grid
                    second_param = E0_src[i]
                interp = RegularGridInterpolator(
                    (first_grid, second_grid), integral_grid_2d[i]
                )
                # E0 = sim_inputs["E0_src"][i]
                # beta = sim_inputs["beta_index"][i]
                eps.append(np.exp(interp(np.array([first_param, second_param])))[0])
            elif fit_index:
                grid = sim_inputs["src_index_grid"]
                param = src_index[i]
            elif fit_beta:
                grid = sim_inputs["beta_index_grid"]
                param = beta_index[i]
            elif fit_Enorm:
                grid = np.log(sim_inputs["E0_src_grid"])
                param = np.log(E0_src[i])
            elif fit_eta:
                grid = sim_inputs["eta_grid"]
                param = eta[i]

            # Create dummy values for all other parameters
            if spectrum == "logparabola":
                kwargs = {
                    "alpha": src_index[i],
                    "e_0": E0_src[i],
                    "beta": beta_index[i],
                }
            elif spectrum == "pgamma":
                kwargs = {"e_0": E0_src[i]}
            elif spectrum == "seyfert":
                kwargs = {"eta": eta[i]}
            else:
                kwargs = {"alpha": src_index[i]}
            if n_params == 1:
                eps.append(np.exp(np.interp(param, grid, integral_grid[i])))

            l = sim_inputs["L"][i]

            kwargs["e_low"] = Emin_src
            kwargs["e_up"] = Emax_src
            if np.isnan(l) or spectrum == "seyfert":
                flux = sources.point_source[i].flux_model.total_flux_int.to_value(
                    1 / u.s / u.m**2
                )
            else:
                flux = l / (4 * np.pi * np.power(d * 3.086e22, 2))
                flux = flux * flux_conv_(**kwargs)
            F.append(flux)

    if diffuse:
        eps.append(np.exp(np.interp(diff_index, diff_index_grid, integral_grid[-1])))

    if atmospheric:
        eps.append(sim_inputs["atmo_integ_val"][c])

    if data_bg:
        eps.append(0.0)

    eps = np.array(eps) * sim_inputs["T"][c]

    if diffuse:
        F.append(sim_inputs["F_diff"])

    if atmospheric:
        F.append(sim_inputs["F_atmo"])

    if data_bg:
        F.append(0.0)

    return eps * np.array(F)
