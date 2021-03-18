import numpy as np
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
import h5py
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel
import logging
import collections

import ligo.skymap.plot
from pyipn.io.plotting.spherical_circle import SphericalCircle

from .detector.detector_model import DetectorModel
from .detector.icecube import IceCubeDetectorModel
from .detector.northern_tracks import NorthernTracksDetectorModel
from .precomputation import ExposureIntegral
from .source.source import Sources, PointSource, icrs_to_uv
from .source.parameter import Parameter
from .source.flux_model import IsotropicDiffuseBG, flux_conv_
from .source.atmospheric_flux import AtmosphericNuMuFlux
from .source.cosmology import luminosity_distance
from .events import Events, TRACKS

from .stan_interface import (
    generate_atmospheric_sim_code_,
    generate_main_sim_code_,
    generate_main_sim_code_hybrid_,
    STAN_PATH,
)


class Simulation:
    """
    To set up and run simulations.
    """

    @u.quantity_input
    def __init__(
        self,
        sources: Sources,
        detector_model: DetectorModel,
        observation_time: u.year,
    ):
        """
        To set up and run simulations.
        """

        self._sources = sources
        self._detector_model_type = detector_model
        self._observation_time = observation_time

        self._sources.organise()
        # Check source components
        source_types = [type(s) for s in self._sources.sources]
        flux_types = [type(s.flux_model) for s in self._sources.sources]
        self._point_source_comp = PointSource in source_types
        self._diffuse_bg_comp = IsotropicDiffuseBG in flux_types
        self._atmospheric_comp = AtmosphericNuMuFlux in flux_types

        self._stan_path = STAN_PATH

        self._exposure_integral = collections.OrderedDict()

        # Silence log output
        logger = logging.getLogger("hierarchical_nu.backend.code_generator")
        logger.propagate = False

    def precomputation(self):
        """
        Run the necessary precomputation
        """

        for event_type in self._detector_model_type.event_types:

            self._exposure_integral[event_type] = ExposureIntegral(
                self._sources,
                self._detector_model_type,
                event_type=event_type,
            )

    def generate_stan_code(self):

        if self._atmospheric_comp:
            self._generate_atmospheric_sim_code()

        self._generate_main_sim_code()

    def set_stan_filenames(self, atmo_sim_filename, main_sim_filename):

        self._atmo_sim_filename = atmo_sim_filename

        self._main_sim_filename = main_sim_filename

    def compile_stan_code(self, include_paths=None):

        if not include_paths:
            include_paths = [self._stan_path]

        stanc_options = {"include_paths": include_paths}

        if self._atmospheric_comp:

            self._atmo_sim = CmdStanModel(
                stan_file=self._atmo_sim_filename,
                stanc_options=stanc_options,
            )

        self._main_sim = CmdStanModel(
            stan_file=self._main_sim_filename,
            stanc_options=stanc_options,
        )

    def run(self, seed=None, verbose=False):

        self._sim_inputs = self._get_sim_inputs(seed)

        self._expected_Nnu = self._get_expected_Nnu(self._sim_inputs)

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
        )

        self._sim_output = sim_output

        self.events = Events(*self._extract_sim_output())

    def _extract_sim_output(self):

        energies = self._sim_output.stan_variable("Edet")[0] * u.GeV
        dirs = self._sim_output.stan_variable("event")[0].reshape((3, len(energies))).T
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
        ang_errs = np.rad2deg(np.sqrt(1.38 / kappa)) * u.deg

        return energies, coords, event_types, ang_errs

    def save(self, filename):

        with h5py.File(filename, "w") as f:

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
            source_folder.create_dataset(
                "total_flux_int", data=self._sources.total_flux_int().value
            )
            source_folder.create_dataset(
                "f", data=self._sources.associated_fraction().value
            )

        self.events.to_file(filename, append=True)

    def show_spectrum(self):

        Esrc = self._sim_output.stan_variable("Esrc")[0]
        E = self._sim_output.stan_variable("E")[0]
        Edet = self.events.energies.value
        Emin_det = self._get_min_det_energy().to(u.GeV).value

        bins = np.logspace(
            np.log10(Emin_det),
            np.log10(Parameter.get_parameter("Emax").value.to(u.GeV).value),
            20,
            base=10,
        )

        fig, ax = plt.subplots()
        ax.hist(Esrc, bins=bins, label="E at source", alpha=0.5)
        ax.hist(E, bins=bins, label="E at detector", alpha=0.5)
        ax.hist(Edet, bins=bins, label="E reconstructed", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("E")
        ax.legend()

        return fig, ax

    def show_skymap(self):

        lam = list(
            self._sim_output.stan_variable("Lambda")[0] - 1
        )  # avoid Stan-style indexing
        event_type = self._sim_output.stan_variable("event_type")[0]
        Ns = self._sim_inputs["Ns"]
        label_cmap = plt.cm.Set1(list(range(self._sources.N)))
        N_src_ev = sum([lam.count(_) for _ in range(Ns)])
        N_bg_ev = lam.count(Ns)
        N_atmo_ev = lam.count(Ns + 1)

        fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
        fig.set_size_inches((7, 5))

        self.events.coords.representation_type = "spherical"
        for r, d, l, e, t in zip(
            self.events.coords.icrs.ra,
            self.events.coords.icrs.dec,
            lam,
            self.events.ang_errs,
            self.events.types,
        ):
            color = label_cmap[int(l)]

            if t == TRACKS:
                e = e * 5

            circle = SphericalCircle(
                (r, d),
                e,  # to make tracks visible
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
        fig.tight_layout()

        return fig, ax

    def setup_and_run(self, include_paths=None):
        """
        Wrapper around setup functions for convenience.
        """

        self.precomputation()
        self.generate_stan_code()
        self.compile_stan_code(include_paths=include_paths)
        self.run()

    def _get_sim_inputs(self, seed=None):

        atmo_inputs = {}
        sim_inputs = {}

        cz_min = min(
            [
                min(e.effective_area._cosz_bin_edges)
                for _, e in self._exposure_integral.items()
            ]
        )
        cz_max = max(
            [
                max(e.effective_area._cosz_bin_edges)
                for _, e in self._exposure_integral.items()
            ]
        )

        if self._atmospheric_comp:

            atmo_inputs["Esrc_min"] = Parameter.get_parameter("Emin").value.value
            atmo_inputs["Esrc_max"] = Parameter.get_parameter("Emax").value.value

            atmo_inputs["cosz_min"] = cz_min
            atmo_inputs["cosz_max"] = cz_max

            atmo_sim = self._atmo_sim.sample(
                data=atmo_inputs,
                iter_sampling=1000,
                chains=1,
                seed=seed,
            )

            atmo_energies = atmo_sim.stan_variable("energy")
            atmo_directions = atmo_sim.stan_variable("omega")

            # Somehow precision of unit_vector gets lost from Stan to here - check
            atmo_directions = [
                (_ / np.linalg.norm(_)).tolist() for _ in atmo_directions
            ]

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
            icrs_to_uv(s.dec.value, s.ra.value)
            for s in self._sources.sources
            if isinstance(s, PointSource)
        ]

        sim_inputs["Ns"] = len(
            [s for s in self._sources.sources if isinstance(s, PointSource)]
        )
        sim_inputs["z"] = redshift
        sim_inputs["D"] = D
        sim_inputs["varpi"] = src_pos

        for event_type in self._detector_model_type.event_types:
            sim_inputs["Ngrid"] = len(
                self._exposure_integral[event_type].par_grids["index"]
            )
            sim_inputs["alpha_grid"] = self._exposure_integral[event_type].par_grids[
                "index"
            ]
            if (
                event_type == "tracks"
                and len(self._detector_model_type.event_types) > 1
            ):
                sim_inputs["integral_grid_t"] = [
                    _.value.tolist()
                    for _ in self._exposure_integral["tracks"].integral_grid
                ]
            elif (
                event_type == "cascades"
                and len(self._detector_model_type.event_types) > 1
            ):
                sim_inputs["integral_grid_c"] = [
                    _.value.tolist()
                    for _ in self._exposure_integral["cascades"].integral_grid
                ]
            else:
                sim_inputs["integral_grid"] = [
                    _.value.tolist()
                    for _ in self._exposure_integral[event_type].integral_grid
                ]

        if self._atmospheric_comp:
            sim_inputs["atmo_integ_val"] = (
                self._exposure_integral["tracks"].integral_fixed_vals[0].value
            )
        sim_inputs["T"] = self._observation_time.to(u.s).value

        if self._atmospheric_comp:
            sim_inputs["N_atmo"] = len(atmo_energies)
            sim_inputs["atmo_energies"] = atmo_energies
            sim_inputs["atmo_directions"] = atmo_directions
            sim_inputs["atmo_weights"] = np.tile(
                1.0 / len(atmo_energies), len(atmo_energies)
            )

        sim_inputs["alpha"] = Parameter.get_parameter("index").value
        sim_inputs["Esrc_min"] = Parameter.get_parameter("Emin").value.to(u.GeV).value
        sim_inputs["Esrc_max"] = Parameter.get_parameter("Emax").value.to(u.GeV).value

        if self._detector_model_type == IceCubeDetectorModel:

            # Here, we must provide tracks and cascade Emin_det case
            # even if they are equal.
            try:

                sim_inputs["Emin_det_tracks"] = (
                    Parameter.get_parameter("Emin_det").value.to(u.GeV).value
                )
                sim_inputs["Emin_det_cascades"] = (
                    Parameter.get_parameter("Emin_det").value.to(u.GeV).value
                )

            except ValueError:

                sim_inputs["Emin_det_tracks"] = (
                    Parameter.get_parameter("Emin_det_tracks").value.to(u.GeV).value
                )

                sim_inputs["Emin_det_cascades"] = (
                    Parameter.get_parameter("Emin_det_cascades").value.to(u.GeV).value
                )

        else:

            # Otherwise, just use Emin_det, as no ambiguity
            sim_inputs["Emin_det"] = (
                Parameter.get_parameter("Emin_det").value.to(u.GeV).value
            )

        # Set maximum based on Emax to speed up rejection sampling
        Emax = sim_inputs["Esrc_max"]
        if self._detector_model_type == IceCubeDetectorModel:

            for event_type in self._detector_model_type.event_types:
                lbe = self._exposure_integral[event_type].effective_area._tE_bin_edges[
                    :-1
                ]
                aeff_max = np.max(
                    self._exposure_integral[event_type].effective_area._eff_area[
                        lbe < Emax
                    ][:]
                )
                if event_type == "tracks":
                    sim_inputs["aeff_t_max"] = aeff_max + 0.01 * aeff_max
                if event_type == "cascades":
                    sim_inputs["aeff_c_max"] = aeff_max + 0.01 * aeff_max

        else:
            event_type = self._detector_model_type.event_types[0]
            lbe = self._exposure_integral[event_type].effective_area._tE_bin_edges[:-1]
            aeff_max = np.max(
                self._exposure_integral[event_type].effective_area._eff_area[
                    lbe < Emax
                ][:]
            )
            sim_inputs["aeff_max"] = aeff_max + 0.01 * aeff_max

        if (
            self._detector_model_type == NorthernTracksDetectorModel
            or self._detector_model_type == IceCubeDetectorModel
        ):
            # Only sample from Northern hemisphere
            cz_max = max(
                self._exposure_integral["tracks"].effective_area._cosz_bin_edges
            )
            sim_inputs["v_lim"] = (np.cos(np.pi - np.arccos(cz_max)) + 1) / 2
        else:
            sim_inputs["v_lim"] = 0.0

        diffuse_bg = self._sources.diffuse_component()
        sim_inputs["F_diff"] = diffuse_bg.flux_model.total_flux_int.value

        if self._atmospheric_comp:
            atmo_bg = self._sources.atmo_component()
            sim_inputs["F_atmo"] = atmo_bg.flux_model.total_flux_int.value

        sim_inputs["L"] = (
            Parameter.get_parameter("luminosity").value.to(u.GeV / u.s).value
        )

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

        if self._detector_model_type == IceCubeDetectorModel:

            integral_grid_t = sim_inputs["integral_grid_t"]
            Nex_t = _get_expected_Nnu_(
                sim_inputs, integral_grid_t, self._atmospheric_comp
            )

            integral_grid_c = sim_inputs["integral_grid_c"]
            Nex_c = _get_expected_Nnu_(
                sim_inputs, integral_grid_c, self._atmospheric_comp
            )

            Nex = Nex_t + Nex_c

        else:

            integral_grid = sim_inputs["integral_grid"]
            Nex = _get_expected_Nnu_(sim_inputs, integral_grid, self._atmospheric_comp)

        self._expected_Nnu_per_comp = Nex

        return sum(Nex)

    @classmethod
    def from_file(cls, filename):

        pass

    def _generate_atmospheric_sim_code(self):

        atmo_flux_model = self._sources.atmo_component().flux_model

        filename = os.path.join(self._stan_path, "atmo_gen")

        self._atmo_sim_filename = generate_atmospheric_sim_code_(
            filename, atmo_flux_model, theta_points=30
        )

    def _generate_main_sim_code(self):

        ps_spec_shape = self._sources.sources[0].flux_model.spectral_shape

        filename = os.path.join(self._stan_path, "sim_code")

        if len(self._detector_model_type.event_types) > 1:

            self._main_sim_filename = generate_main_sim_code_hybrid_(
                filename,
                ps_spec_shape,
                self._detector_model_type,
                self._diffuse_bg_comp,
                self._atmospheric_comp,
            )

        else:

            self._main_sim_filename = generate_main_sim_code_(
                filename,
                ps_spec_shape,
                self._detector_model_type,
                self._diffuse_bg_comp,
                self._atmospheric_comp,
            )

    def _get_min_det_energy(event_type=None):
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

            Emin_det_t = Parameter.get_parameter("Emin_det_tracks").value

            Emin_det_c = Parameter.get_parameter("Emin_det_cascades").value

            Emin_det = min(Emin_det_t, Emin_det_c)

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
        with h5py.File("output/test_sim_file.h5", "r") as f:

            inputs_folder = f["sim/inputs"]
            source_folder = f["sim/source"]
            outputs_folder = f["sim/outputs"]

            atmo_comp = False
            for key in inputs_folder:
                inputs[key] = inputs_folder[key][()]
                if key == "F_atmo":
                    atmo_comp = True

            for key in source_folder:
                inputs[key] = source_folder[key][()]

            for key in outputs_folder:
                outputs[key] = outputs_folder[key][()]

        truths = {}
        truths["F_diff"] = inputs["F_diff"]
        truths["L"] = inputs["L"]
        truths["Ftot"] = inputs["total_flux_int"]
        truths["f"] = inputs["f"]
        truths["alpha"] = inputs["alpha"]

        if atmo_comp:
            truths["F_atmo"] = inputs["F_atmo"]

        return cls(truths, inputs, outputs)


def _get_expected_Nnu_(sim_inputs, integral_grid, atmospheric_comp=False):
    """
    Helper function for calculating expected Nnu
    using stan sim_inputs.
    """

    alpha = sim_inputs["alpha"]
    alpha_grid = sim_inputs["alpha_grid"]

    eps = []
    for igrid in integral_grid:
        eps.append(np.interp(alpha, alpha_grid, igrid))

    if atmospheric_comp:
        eps.append(sim_inputs["atmo_integ_val"])

    eps = np.array(eps) * sim_inputs["T"]

    F = []
    for d in sim_inputs["D"]:
        flux = sim_inputs["L"] / (4 * np.pi * np.power(d * 3.086e22, 2))
        flux = flux * flux_conv_(alpha, sim_inputs["Esrc_min"], sim_inputs["Esrc_max"])
        F.append(flux)
    F.append(sim_inputs["F_diff"])

    if atmospheric_comp:
        F.append(sim_inputs["F_atmo"])

    return eps * F
