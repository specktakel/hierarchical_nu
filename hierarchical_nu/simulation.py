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
from hierarchical_nu.source.source import Sources, PointSource, icrs_to_uv
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import (
    IsotropicDiffuseBG, flux_conv_, LogParabolaSpectrum
)
from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.events import Events
from hierarchical_nu.utils.roi import ROI, CircularROI, ROIList

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.utils.git import git_hash


sim_logger = logging.getLogger(__name__)
sim_logger.setLevel(logging.DEBUG)


class Simulation:
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
        :param sources: Instance of `Sources` containing all to-be-simulated sources
        :param event_types: `EventType` or list thereof, NB that not all event types should be combined
        :param observation_time: single lifetime or dict of EventType: lifetime
        :param atmo_flux_energy_points: number of atmo model grid points in energy space
        :param atmo_flux_theta_points: number of atmo model grid points in theta/dec space
        :param n_grid_poins: number of grid points for interpolations in exposure integral
        :param N: if provided, simulate fixed number of events, e.g. {IC86_II: [1, 2, 3]}
        :param asimov: if True, simulate np.rint(sim._Nex_et) events, overrides N
        """

        self._sources = sources
        if not isinstance(event_types, list):
            event_types = [event_types]
        if isinstance(observation_time, u.quantity.Quantity):
            observation_time = {event_types[0]: observation_time}
        assert len(event_types) == len(observation_time)
        self._event_types = event_types
        self._observation_time = observation_time
        self._n_grid_points = n_grid_points
        self._asimov = asimov

        self._sources.organise()

        self._exposure_integral = collections.OrderedDict()

        if asimov:
            N = {}
            for event_type in self._event_types:
                # dummy values
                N[event_type] = [1] * self._sources.N

        if N:
            for event_type in self._event_types:
                assert len(N[event_type]) == len(sources)

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

        self.events = None

    def precomputation(
        self,
        exposure_integral: collections.OrderedDict = None,
    ):
        """
        Run the necessary precomputation
        """

        if not exposure_integral:
            for event_type in self._event_types:
                self._exposure_integral[event_type] = ExposureIntegral(
                    self._sources, event_type, self._n_grid_points
                )

        else:
            self._exposure_integral = exposure_integral

    def generate_stan_code(self):
        self._main_sim_filename = self._stan_interface.generate()

    def set_stan_filename(self, sim_filename):
        self._main_sim_filename = sim_filename

    def compile_stan_code(self, include_paths=None):
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

        if verbose:
            print(
                "Running a simulation with expected Nnu = %.2f events"
                % self._expected_Nnu
            )

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
                sim_logger.warning(f"{dirname} does not exist, saving instead to {os.getcwd()}")
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

    def show_spectrum(
        self, *components: str, scale: str = "linear", population: bool = False
    ):
        """
        Show binned spectrum of simulated data
        :param components: not used? what is this?
        :param scale: either `linear` or `log` to change y-axis scale of histograms
        :param population: if True, display all point sources as one entry
        """

        hatch_cycle = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
        Esrc = self._sim_output.stan_variable("Esrc")[0]
        E = self._sim_output.stan_variable("E")[0]
        lam = self._sim_output.stan_variable("Lambda")[0] - 1
        assert np.all(Esrc >= E)
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
        Emin_det = []
        rs_cvals = []
        rs_bbpl_Eth = []
        rs_bbpl_gamma1 = []
        rs_bbpl_gamma2_scale = []
        forced_N = []
        obs_time = []

        if asimov:
            # Round expected number of events to nearest integer per source
            # distribute this number weighted with the Nex per event type over the event types
            N = np.rint(self._Nex_et.sum(axis=0)).astype(int)
            self._N = np.zeros_like(self._Nex_et)
            for c, _ in enumerate(self._sources):

                weights = self._Nex_et[:, c] / self._Nex_et[:, c].sum()

                # Sample et_idx for each source
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

<<<<<<< HEAD
        for c, event_type in enumerate(self._event_types):
=======
        for event_type in self._event_types:
            if self._force_N:
                forced_N.append(self._N[event_type])

            integral_grid.append(
                [
                    np.log(_.to_value(u.m**2)).tolist()
                    for _ in self._exposure_integral[event_type].integral_grid
                ]
            )

            if self._sources.atmospheric:
                atmo_integ_val.append(
                    self._exposure_integral[event_type]
                    .integral_fixed_vals[0]
                    .to_value(u.m**2)
                )

>>>>>>> stan-frames
            obs_time.append(self._observation_time[event_type].to(u.s).value)

        if self._sources.point_source:
            # Check for shared source index
            if self._shared_src_index:
                key = "src_index"
                if isinstance(
                    self._sources.point_source[0].flux_model.spectral_shape,
                    LogParabolaSpectrum,
                ):
                    key_beta = "beta_index"

            # Otherwise just use first source in the list
            # src_index_grid is identical for all point sources
            else:
                key = "%s_src_index" % self._sources.point_source[0].name
                if isinstance(
                    self._sources.point_source[0].flux_model.spectral_shape,
                    LogParabolaSpectrum,
                ):
                    key_beta = "%s_beta_index" % self._sources.point_source[0].name

            sim_inputs["src_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids[key]

            if isinstance(
                self._sources.point_source[0].flux_model.spectral_shape,
                LogParabolaSpectrum,
            ):
                sim_inputs["beta_index_grid"] = self._exposure_integral[
                    event_type
                ].par_grids[key_beta]
                sim_inputs["E0"] = self._sources.point_source[0].flux_model.spectral_shape._normalisation_energy.to_value(u.GeV)

            # Check for shared src_index parameter
            if self._shared_src_index:
                sim_inputs["src_index"] = Parameter.get_parameter("src_index").value
                if isinstance(
                    self._sources.point_source[0].flux_model.spectral_shape,
                    LogParabolaSpectrum,
                ):
                    sim_inputs["beta_index"] = Parameter.get_parameter("beta_index").value
                

            # Otherwise look for individual src_index parameters
            else:
                sim_inputs["src_index"] = [
                    Parameter.get_parameter("%s_src_index" % s.name).value
                    for s in self._sources.point_source
                ]
<<<<<<< HEAD

                if isinstance(
                    self._sources.point_source[0].flux_model.spectral_shape,
                    LogParabolaSpectrum,
                ):
                    sim_inputs["beta_index"] = [
                        Parameter.get_parameter("%s_beta_index" % s.name).value
                        for s in self._sources.point_source
                    ]

            sim_inputs["Emin_src"] = (
                Parameter.get_parameter("Emin_src").value.to(u.GeV).value
            )
            sim_inputs["Emax_src"] = (
                Parameter.get_parameter("Emax_src").value.to(u.GeV).value
            )
=======
            sim_inputs["Emin_src"] = [
                ps.frame.transform(
                    Parameter.get_parameter("Emin_src").value,
                    ps.redshift
                ).to_value(u.GeV) for ps in self._sources.point_source
            ]
            sim_inputs["Emax_src"] = [
                ps.frame.transform(
                    Parameter.get_parameter("Emax_src").value,
                    ps.redshift
                ).to_value(u.GeV) for ps in self._sources.point_source
            ]
>>>>>>> stan-frames

        if self._sources.diffuse:
            # Same as for point sources
            sim_inputs["Ngrid"] = len(
                self._exposure_integral[event_type].par_grids["diff_index"]
            )

            sim_inputs["diff_index_grid"] = self._exposure_integral[
                event_type
            ].par_grids["diff_index"]

            sim_inputs["diff_index"] = Parameter.get_parameter("diff_index").value

            sim_inputs["Emin_diff"] = self._sources.diffuse.frame.transform(
                Parameter.get_parameter("Emin_diff").value,
                self._sources.diffuse.redshift
            ).to_value(u.GeV)
            sim_inputs["Emax_diff"] = self._sources.diffuse.frame.transform(
                Parameter.get_parameter("Emax_diff").value,
                self._sources.diffuse.redshift
            ).to_value(u.GeV)

        sim_inputs["Emin"] = Parameter.get_parameter("Emin").value.to_value(u.GeV)
        sim_inputs["Emax"] = Parameter.get_parameter("Emax").value.to_value(u.GeV)

        for c, event_type in enumerate(self._event_types):
            effective_area = self._exposure_integral[event_type].effective_area

            try:
                Emin_det.append(
                    Parameter.get_parameter("Emin_det").value.to_value(u.GeV)
                )

            except ValueError:
                Emin_det.append(
                    Parameter.get_parameter(f"Emin_det_{event_type.P}")
                    .value.to_value(u.GeV)
                )

            # Rejection sampling
            rs_bbpl_Eth.append(effective_area.rs_bbpl_params["threshold_energy"])
            rs_bbpl_gamma1.append(effective_area.rs_bbpl_params["gamma1"])
            rs_bbpl_gamma2_scale.append(effective_area.rs_bbpl_params["gamma2_scale"])
            rs_cvals.append(self._exposure_integral[event_type].c_values)

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

            assert v_lim_high > v_lim_low

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
            sim_inputs["F_atmo"] = (
                Parameter.get_parameter("F_atmo").value.to_value(flux_units)
            )
        lumi_units = u.GeV / u.s

        if self._sources.point_source:
            # Check for shared luminosity parameter
            if self._shared_luminosity:
                sim_inputs["L"] = (
                    Parameter.get_parameter("luminosity").value.to_value(lumi_units)
                )

            # Otherwise, look for individual luminosity parameters
            else:
                sim_inputs["L"] = [
                    Parameter.get_parameter("%s_luminosity" % s.name)
                    .value.to_value(lumi_units)
                    for s in self._sources.point_source
                ]

        sim_inputs["integral_grid"] = integral_grid
        sim_inputs["integral_grid_2d"] = integral_grid_2d
        if self._sources.atmospheric:
            sim_inputs["atmo_integ_val"] = atmo_integ_val
        sim_inputs["Emin_det"] = Emin_det
        sim_inputs["rs_cvals"] = rs_cvals
        sim_inputs["rs_bbpl_Eth"] = rs_bbpl_Eth
        sim_inputs["rs_bbpl_gamma1"] = rs_bbpl_gamma1
        sim_inputs["rs_bbpl_gamma2_scale"] = rs_bbpl_gamma2_scale
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

    def _get_expected_Nnu(self, sim_inputs):
        """
        Calculates expected number of neutrinos to be simulated.
        Uses same approach as in the Stan code for cross-checks.
        """

        sim_inputs_ = sim_inputs.copy()

        Nex_et = np.zeros((len(self._event_types), self._sources.N))
        for c, event_type in enumerate(self._event_types):
            integral_grid = sim_inputs_["integral_grid"][c]
            integral_grid_2d = sim_inputs_["integral_grid_2d"][c]
            Nex_et[c] = _get_expected_Nnu_(
                c,
                sim_inputs_,
                integral_grid,
                integral_grid_2d,
                self._sources.point_source,
                self._sources.diffuse,
                self._sources.atmospheric,
                self._shared_luminosity,
                self._shared_src_index,
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

            atmo = False
            diff = False
            ps = False
            for key in inputs_folder:
                inputs[key] = inputs_folder[key][()]

                if key == "F_atmo":
                    atmo = True

                if key == "F_diff":
                    diff = True

                if key == "L":
                    ps = True

            for key in source_folder:
                inputs[key] = source_folder[key][()]

            for key in outputs_folder:
                outputs[key] = outputs_folder[key][()]

        truths = {}

        if ps:
            truths["L"] = inputs["L"]
            truths["src_index"] = inputs["src_index"]

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
        """

        # Meow
        from astropy.coordinates import concatenate as cat
        from astropy.time import Time

        bg = cls.from_file(background)
        ps = cls.from_file(point_source)

        bg_events = Events.from_file(background)
        ps_events = Events.from_file(point_source)

        energies = np.hstack((ps_events.energies, bg_events.energies))
        coords = cat((ps_events.coords, bg_events.coords))
        types = np.hstack((ps_events.types, bg_events.types))
        mjd = np.hstack((ps_events.mjd.to_value("mjd"), bg_events.mjd.to_value("mjd")))
        mjd = Time(mjd, format="mjd")
        ang_errs = np.hstack((ps_events.ang_errs, bg_events.ang_errs))

        # Create common event objects
        events = Events(energies, coords, types, ang_errs, mjd)

        # Truths keys
        ps_keys = ["L", "src_index"]
        bg_keys = ["F_atmo", "F_diff", "diff_index"]

        truths = {}
        for key in ps_keys:
            truths[key] = ps.truths[key]
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
                inputs_folder.create_dataset(key, data=ps.inputs[key])
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
    integral_grid,
    integral_grid_2d,
    point_source=False,
    diffuse=False,
    atmospheric=False,
    shared_luminosity=True,
    shared_src_index=True,
):
    """
    Helper function for calculating expected Nnu
    using stan sim_inputs.
    """

    if point_source:
        if shared_src_index:
            try:
                beta_index = sim_inputs["beta_index"]
                beta_index_grid = sim_inputs["beta_index_grid"]
            except KeyError:
                beta_index = None
            src_index = sim_inputs["src_index"]
        else:
            src_index_list = sim_inputs["src_index"]
            try:
                beta_index_list = sim_inputs["beta_index"]
                beta_index_grid = sim_inputs["beta_index_grid"]
            except KeyError:
                beta_index = None
        src_index_grid = sim_inputs["src_index_grid"]

    if diffuse:
        diff_index = sim_inputs["diff_index"]
        diff_index_grid = sim_inputs["diff_index_grid"]

    Ns = sim_inputs["Ns"]

    eps = []

    if point_source:
        for i in range(Ns):
            if shared_src_index:
                src_index_ref = src_index
                if beta_index is not None:
                    beta_index_ref = beta_index
            else:
                src_index_ref = src_index_list[i]
                if beta_index:
                    beta_index_ref = beta_index_list[i]
            if beta_index is not None:
                interp = RegularGridInterpolator((src_index_grid, beta_index_grid), integral_grid_2d[i])
                eps.append(
                    np.exp(interp(np.array([src_index_ref, beta_index_ref])))
                )
            else:
                eps.append(
                    np.exp(np.interp(src_index_ref, src_index_grid, integral_grid[i]))
                )
            """
            else:
                eps.append(
                    np.exp(
                        np.interp(src_index_list[i], src_index_grid, integral_grid[i])
                    )
                )
            """

    if diffuse:
        eps.append(np.exp(np.interp(diff_index, diff_index_grid, integral_grid[Ns])))

    if atmospheric:
        eps.append(sim_inputs["atmo_integ_val"][c])

    eps = np.array(eps) * sim_inputs["T"][c]

    F = []

    if point_source:
        if shared_luminosity:
            for i, (d, Emin_src, Emax_src) in enumerate(
                zip(sim_inputs["D"], sim_inputs["Emin_src"], sim_inputs["Emax_src"])
            ):
                flux = sim_inputs["L"] / (4 * np.pi * np.power(d * 3.086e22, 2))
                if shared_src_index:
                    flux = flux * flux_conv_(
                        src_index,
                        Emin_src,
                        Emax_src,
                    )
                else:
                    flux = flux * flux_conv_(
                        src_index_list[i],
                        Emin_src,
                        Emax_src,
                    )
                F.append(flux)

        else:
            for i, (d, l, Emin_src, Emax_src) in enumerate(
                zip(
                    sim_inputs["D"],
                    sim_inputs["L"],
                    sim_inputs["Emin_src"],
                    sim_inputs["Emax_src"]
                )
            ):
                flux = l / (4 * np.pi * np.power(d * 3.086e22, 2))
                if shared_src_index:
                    flux = flux * flux_conv_(
                        src_index,
                        Emin_src,
                        Emax_src,
                    )
                else:
                    flux = flux * flux_conv_(
                        src_index_list[i],
                        Emin_src,
                        Emax_src,
                    )
                F.append(flux)

    if diffuse:
        F.append(sim_inputs["F_diff"])

    if atmospheric:
        F.append(sim_inputs["F_atmo"])

    return eps * np.array(F)
