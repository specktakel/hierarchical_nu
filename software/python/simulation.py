import numpy as np
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
import h5py
from matplotlib import pyplot as plt
import stan_utility

from .detector_model import DetectorModel
from .precomputation import ExposureIntegral
from .source.source import Sources, PointSource, icrs_to_uv
from .source.parameter import Parameter
from .source.flux_model import IsotropicDiffuseBG
from .source.atmospheric_flux import AtmosphericNuMuFlux
from .source.cosmology import luminosity_distance
from .events import Events, TRACKS, CASCADES

from .backend.stan_generator import (
    StanFileGenerator,
    FunctionsContext,
    Include,
    DataContext,
    TransformedDataContext,
    ParametersContext,
    TransformedParametersContext,
    GeneratedQuantitiesContext,
    ForLoopContext,
    IfBlockContext,
    ElseIfBlockContext,
    ElseBlockContext,
    WhileLoopContext,
    ModelContext,
    FunctionCall,
)

from .backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ParameterDef,
    ParameterVectorDef,
)
from .backend.expression import StringExpression
from .backend.parameterizations import DistributionMode


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
        output_dir="stan_files",
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

        if not os.path.exists("~/.stan_cache"):
            os.makedirs("~/.stan_cache")

        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def precomputation(self):
        """
        Run the necessary precomputation
        """

        self._exposure_integral = ExposureIntegral(
            self._sources, self._detector_model_type
        )

    def generate_stan_code(self):

        if self._atmospheric_comp:
            self._generate_atmospheric_sim_code()

        self._generate_main_sim_code()

    def compile_stan_code(self):

        this_dir = os.path.abspath("")
        include_paths = [os.path.join(this_dir, self.output_dir)]
        self._atmo_sim = stan_utility.compile_model(
            filename=self._atmo_sim_filename,
            include_paths=include_paths,
            model_name="atmo_sim",
        )
        self._main_sim = stan_utility.compile_model(
            filename=self._main_sim_filename,
            include_paths=include_paths,
            model_name="main_sim",
        )

    def run(self, seed=None):

        self._sim_inputs = self._get_sim_inputs(seed)

        sim_output = self._main_sim.sampling(
            data=self._sim_inputs, iter=1, chains=1, algorithm="Fixed_param", seed=seed
        )

        self._sim_output = sim_output

        energies, coords, event_types = self._extract_sim_output()

        self.events = Events(*self._extract_sim_output())

    def _extract_sim_output(self):

        energies = self._sim_output.extract(["Edet"])["Edet"][0] * u.GeV
        dirs = self._sim_output.extract(["event"])["event"][0]
        coords = SkyCoord(
            dirs.T[0],
            dirs.T[1],
            dirs.T[2],
            representation_type="cartesian",
            frame="icrs",
        )
        event_types = self._sim_output.extract(["event_type"])["event_type"][0]
        event_types = [int(_) for _ in event_types]

        return energies, coords, event_types

    def save(self, filename):

        with h5py.File(filename, "w") as f:

            sim_folder = f.create_group("sim")

            inputs_folder = sim_folder.create_group("inputs")
            for key, value in self._sim_inputs.items():
                inputs_folder.create_dataset(key, data=value)

            outputs_folder = sim_folder.create_group("outputs")
            for key, value in self._sim_output.extract(permuted=True).items():
                outputs_folder.create_dataset(key, data=value)

            source_folder = sim_folder.create_group("source")
            source_folder.create_dataset(
                "total_flux_int", data=self._sources.total_flux_int().value
            )
            source_folder.create_dataset(
                "f", data=self._sources.associated_fraction().value
            )

        self.events.to_file(filename, append=True)

    def show_spectrum(self):

        Esrc = self._sim_output.extract(["Esrc"])["Esrc"][0]
        E = self._sim_output.extract(["E"])["E"][0]
        Edet = self.events.energies.value

        bins = np.logspace(
            np.log10(Parameter.get_parameter("Emin_det").value.to(u.GeV).value),
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

        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection

        lam = list(
            self._sim_output.extract(["Lambda"])["Lambda"][0] - 1
        )  # avoid Stan-style indexing
        Ns = self._sim_inputs["Ns"]
        label_cmap = plt.cm.get_cmap("plasma", self._sources.N)

        N_src_ev = sum([lam.count(_) for _ in range(Ns)])
        N_bg_ev = lam.count(Ns) + lam.count(Ns + 1)

        fig = plt.figure()
        fig.set_size_inches((10, 8))
        ax = fig.add_subplot(111, projection="hammer")

        circles = []
        self.events.coords.representation_type = "spherical"
        for r, d, l in zip(
            self.events.coords.icrs.ra.rad, self.events.coords.icrs.dec.rad, lam
        ):
            color = label_cmap.colors[int(l)]
            circles.append(
                mpatches.Circle((r - np.pi, d), 0.05, color=color, alpha=0.7)
            )  # TODO: Fix this so x-axis labels are correct

        collection = PatchCollection(circles, match_original=True)
        ax.add_collection(collection)

        ax.set_title(
            "N_src_events = %i, N_bg_events = %i" % (N_src_ev, N_bg_ev), pad=30
        )

        return fig, ax

    def setup_and_run(self):
        """
        Wrapper around setup functions for convenience.
        """

        self.precomputation()
        self.generate_stan_code()
        self.compile_stan_code()
        self.run()

    def _get_sim_inputs(self, seed=None):

        atmo_inputs = {}
        sim_inputs = {}

        cz_min = min(self._exposure_integral.effective_area._cosz_bin_edges)
        cz_max = max(self._exposure_integral.effective_area._cosz_bin_edges)

        if self._atmospheric_comp:

            atmo_inputs["Esrc_min"] = Parameter.get_parameter("Emin").value.value
            atmo_inputs["Esrc_max"] = Parameter.get_parameter("Emax").value.value

            atmo_inputs["cosz_min"] = cz_min
            atmo_inputs["cosz_max"] = cz_max

            atmo_sim = self._atmo_sim.sampling(
                data=atmo_inputs,
                iter=1000,
                chains=1,
                algorithm="NUTS",
                seed=seed,
            )

            atmo_energies = atmo_sim.extract(["energy"])["energy"]
            atmo_directions = atmo_sim.extract(["omega"])["omega"]

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

        sim_inputs["Ngrid"] = len(self._exposure_integral.par_grids["index"])
        sim_inputs["alpha_grid"] = self._exposure_integral.par_grids["index"]
        sim_inputs["integral_grid"] = [
            _.value for _ in self._exposure_integral.integral_grid
        ]
        if self._atmospheric_comp:
            sim_inputs["atmo_integ_val"] = self._exposure_integral.integral_fixed_vals[
                0
            ].value
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
        sim_inputs["Edet_min"] = (
            Parameter.get_parameter("Emin_det").value.to(u.GeV).value
        )

        # Set maximum based on Emax to speed up rejection sampling
        lbe = self._exposure_integral.effective_area._tE_bin_edges[:-1]
        Emax = sim_inputs["Esrc_max"]
        aeff_max = np.max(
            self._exposure_integral.effective_area._eff_area[lbe < Emax][:]
        )
        sim_inputs["aeff_max"] = aeff_max + 0.01 * aeff_max

        # Only sample from Northern hemisphere
        sim_inputs["v_lim"] = (np.cos(np.pi - np.arccos(cz_max)) + 1) / 2

        diffuse_bg = self._sources.diffuse_component()
        sim_inputs["F_diff"] = diffuse_bg.flux_model.total_flux_int.value

        if self._atmospheric_comp:
            atmo_bg = self._sources.atmo_component()
            sim_inputs["F_atmo"] = atmo_bg.flux_model.total_flux_int.value

        sim_inputs["L"] = (
            Parameter.get_parameter("luminosity").value.to(u.GeV / u.s).value
        )

        return sim_inputs

    def _generate_atmospheric_sim_code(self):

        if self._sources.atmo_component():
            atmo_flux_model = self._sources.atmo_component().flux_model

        with StanFileGenerator(self.output_dir + "/atmo_gen") as atmo_gen:

            with FunctionsContext():
                _ = Include("utils.stan")
                _ = Include("interpolation.stan")

                # Increasing theta points too much makes compilation very slow
                # Could switch to passing array as data if problematic
                atmu_nu_flux = atmo_flux_model.make_stan_function(theta_points=30)

            with DataContext():
                Esrc_min = ForwardVariableDef("Esrc_min", "real")
                Esrc_max = ForwardVariableDef("Esrc_max", "real")

                cosz_min = ForwardVariableDef("cosz_min", "real")
                cosz_max = ForwardVariableDef("cosz_max", "real")

            with ParametersContext():
                # Simulate from Edet_min and cosz bounds for efficiency
                energy = ParameterDef("energy", "real", Esrc_min, Esrc_max)
                coszen = ParameterDef("coszen", "real", cosz_min, cosz_max)
                phi = ParameterDef("phi", "real", 0, 2 * np.pi)

            with TransformedParametersContext():
                omega = ForwardVariableDef("omega", "vector[3]")
                zen = ForwardVariableDef("zen", "real")
                theta = ForwardVariableDef("theta", "real")

                zen << FunctionCall([coszen], "acos")
                theta << FunctionCall([], "pi") - zen

                omega[1] << FunctionCall([theta], "sin") * FunctionCall([phi], "cos")
                omega[2] << FunctionCall([theta], "sin") * FunctionCall([phi], "sin")
                omega[3] << FunctionCall([theta], "cos")

            with ModelContext():

                logflux = FunctionCall([atmu_nu_flux(energy, omega)], "log")
                StringExpression(["target += ", logflux])

        atmo_gen.generate_single_file()

        self._atmo_sim_filename = atmo_gen.filename

    def _generate_main_sim_code(self):

        ps_spec_shape = self._sources.sources[0].flux_model.spectral_shape

        with StanFileGenerator(self.output_dir + "/sim_code") as sim_gen:

            with FunctionsContext():
                _ = Include("utils.stan")
                _ = Include("vMF.stan")
                _ = Include("interpolation.stan")
                _ = Include("sim_functions.stan")

                spectrum_rng = ps_spec_shape.make_stan_sampling_func("spectrum_rng")
                flux_fac = ps_spec_shape.make_stan_flux_conv_func("flux_conv")

            with DataContext():

                # Sources
                Ns = ForwardVariableDef("Ns", "int")
                Ns_str = ["[", Ns, "]"]
                Ns_1p_str = ["[", Ns, "+1]"]

                varpi = ForwardArrayDef("varpi", "unit_vector[3]", Ns_str)
                D = ForwardVariableDef("D", "vector[Ns]")
                if self._diffuse_bg_comp:
                    z = ForwardVariableDef("z", "vector[Ns+1]")
                else:
                    z = ForwardVariableDef("z", "vector[Ns]")

                # Energies
                alpha = ForwardVariableDef("alpha", "real")
                Edet_min = ForwardVariableDef("Edet_min", "real")
                Esrc_min = ForwardVariableDef("Esrc_min", "real")
                Esrc_max = ForwardVariableDef("Esrc_max", "real")

                # Luminosity/ diffuse flux
                L = ForwardVariableDef("L", "real")
                F_diff = ForwardVariableDef("F_diff", "real")
                F_atmo = ForwardVariableDef("F_atmo", "real")

                # Precomputed quantities
                Ngrid = ForwardVariableDef("Ngrid", "int")
                alpha_grid = ForwardVariableDef("alpha_grid", "vector[Ngrid]")
                if self._diffuse_bg_comp:
                    integral_grid = ForwardArrayDef(
                        "integral_grid", "vector[Ngrid]", Ns_1p_str
                    )
                else:
                    integral_grid = ForwardArrayDef(
                        "integral_grid", "vector[Ngrid]", Ns_str
                    )

                if self._atmospheric_comp:
                    atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")

                aeff_max = ForwardVariableDef("aeff_max", "real")

                v_lim = ForwardVariableDef("v_lim", "real")
                T = ForwardVariableDef("T", "real")

                if self._atmospheric_comp:
                    # Atmo samples
                    N_atmo = ForwardVariableDef("N_atmo", "int")
                    N_atmo_str = ["[", N_atmo, "]"]
                    atmo_directions = ForwardArrayDef(
                        "atmo_directions", "unit_vector[3]", N_atmo_str
                    )
                    atmo_energies = ForwardVariableDef(
                        "atmo_energies", "vector[N_atmo]"
                    )
                    atmo_weights = ForwardVariableDef("atmo_weights", "simplex[N_atmo]")

            with TransformedDataContext():

                if self._diffuse_bg_comp and self._atmospheric_comp:
                    F = ForwardVariableDef("F", "vector[Ns+2]")
                    w_exposure = ForwardVariableDef("w_exposure", "simplex[Ns+2]")
                    eps = ForwardVariableDef("eps", "vector[Ns+2]")
                elif self._diffuse_bg_comp or self._atmospheric_comp:
                    F = ForwardVariableDef("F", "vector[Ns+1]")
                    w_exposure = ForwardVariableDef("w_exposure", "simplex[Ns+1]")
                    eps = ForwardVariableDef("eps", "vector[Ns+1]")
                else:
                    F = ForwardVariableDef("F", "vector[Ns]")
                    w_exposure = ForwardVariableDef("w_exposure", "simplex[Ns]")
                    eps = ForwardVariableDef("eps", "vector[Ns]")

                track_type = ForwardVariableDef("track_type", "int")
                cascade_type = ForwardVariableDef("cascade_type", "int")

                track_type << TRACKS
                cascade_type << CASCADES

                Ftot = ForwardVariableDef("Ftot", "real")
                Fsrc = ForwardVariableDef("Fs", "real")
                f = ForwardVariableDef("f", "real")
                Nex = ForwardVariableDef("Nex", "real")
                N = ForwardVariableDef("N", "int")

                Fsrc << 0.0
                with ForLoopContext(1, Ns, "k") as k:
                    F[k] << StringExpression(
                        [L, "/ (4 * pi() * pow(", D[k], " * ", 3.086e22, ", 2))"]
                    )
                    StringExpression([F[k], "*=", flux_fac(alpha, Esrc_min, Esrc_max)])
                    StringExpression([Fsrc, " += ", F[k]])

                if self._diffuse_bg_comp:
                    StringExpression("F[Ns+1]") << F_diff

                if self._atmospheric_comp:
                    StringExpression("F[Ns+2]") << F_atmo

                if self._diffuse_bg_comp and self._atmospheric_comp:
                    Ftot << Fsrc + F_diff + F_atmo
                elif self._diffuse_bg_comp:
                    Ftot << Fsrc + F_diff
                else:
                    Ftot << Fsrc

                f << StringExpression([Fsrc, "/", Ftot])
                StringExpression(['print("f: ", ', f, ")"])

                if self._atmospheric_comp:
                    eps << FunctionCall(
                        [alpha, alpha_grid, integral_grid, atmo_integ_val, T, Ns],
                        "get_exposure_factor_atmo",
                    )
                else:
                    eps << FunctionCall(
                        [alpha, alpha_grid, integral_grid, T, Ns], "get_exposure_factor"
                    )

                Nex << StringExpression(["get_Nex(", F, ", ", eps, ")"])
                w_exposure << StringExpression(
                    ["get_exposure_weights(", F, ", ", eps, ")"]
                )
                N << StringExpression(["poisson_rng(", Nex, ")"])
                StringExpression(["print(", w_exposure, ")"])
                StringExpression(["print(", Ngrid, ")"])
                StringExpression(["print(", Nex, ")"])
                StringExpression(["print(", N, ")"])

            with GeneratedQuantitiesContext():
                dm_rng = self._detector_model_type(mode=DistributionMode.RNG)
                dm_pdf = self._detector_model_type(mode=DistributionMode.PDF)

                N_str = ["[", N, "]"]
                lam = ForwardArrayDef("Lambda", "int", N_str)
                omega = ForwardVariableDef("omega", "unit_vector[3]")

                Esrc = ForwardVariableDef("Esrc", "vector[N]")
                E = ForwardVariableDef("E", "vector[N]")
                Edet = ForwardVariableDef("Edet", "vector[N]")

                if self._atmospheric_comp:
                    atmo_index = ForwardVariableDef("atmo_index", "int")
                cosz = ForwardArrayDef("cosz", "real", N_str)
                Pdet = ForwardArrayDef("Pdet", "real", N_str)
                accept = ForwardVariableDef("accept", "int")
                detected = ForwardVariableDef("detected", "int")
                ntrials = ForwardVariableDef("ntrials", "int")
                prob = ForwardVariableDef("prob", "simplex[2]")

                event = ForwardArrayDef("event", "unit_vector[3]", N_str)
                Nex_sim = ForwardVariableDef("Nex_sim", "real")

                event_type = ForwardVariableDef("event_type", "vector[N]")

                Nex_sim << Nex

                with ForLoopContext(1, N, "i") as i:

                    lam[i] << FunctionCall([w_exposure], "categorical_rng")

                    accept << 0
                    detected << 0
                    ntrials << 0

                    with WhileLoopContext([StringExpression([accept != 1])]):

                        # Sample position
                        with IfBlockContext([StringExpression([lam[i], " <= ", Ns])]):
                            omega << varpi[lam[i]]
                        with ElseIfBlockContext(
                            [StringExpression([lam[i], " == ", Ns + 1])]
                        ):
                            omega << FunctionCall([1, v_lim], "sphere_lim_rng")
                        if self._atmospheric_comp:
                            with ElseIfBlockContext(
                                [StringExpression([lam[i], " == ", Ns + 2])]
                            ):
                                atmo_index << FunctionCall(
                                    [atmo_weights], "categorical_rng"
                                )
                                omega << atmo_directions[atmo_index]

                        cosz[i] << FunctionCall(
                            [FunctionCall([omega], "omega_to_zenith")], "cos"
                        )
                        # Sample energy
                        with IfBlockContext(
                            [StringExpression([lam[i], " <= ", Ns + 1])]
                        ):
                            Esrc[i] << spectrum_rng(alpha, Esrc_min, Esrc_max)
                            E[i] << Esrc[i] / (1 + z[lam[i]])

                        if self._atmospheric_comp:
                            with ElseIfBlockContext(
                                [StringExpression([lam[i], " == ", Ns + 2])]
                            ):
                                E[i] << atmo_energies[atmo_index]

                        # Test against Aeff
                        with IfBlockContext([StringExpression([cosz[i], ">= 0.1"])]):
                            Pdet[i] << 0
                        with ElseBlockContext():
                            Pdet[i] << dm_pdf.effective_area(E[i], omega) / aeff_max

                        Edet[i] << 10 ** dm_rng.energy_resolution(E[i])

                        prob[1] << Pdet[i]
                        prob[2] << 1 - Pdet[i]
                        StringExpression([ntrials, " += ", 1])

                        with IfBlockContext([StringExpression([ntrials, "< 1000000"])]):
                            detected << FunctionCall([prob], "categorical_rng")
                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            "(",
                                            Edet[i],
                                            " >= ",
                                            Edet_min,
                                            ") && (",
                                            detected == 1,
                                            ")",
                                        ]
                                    )
                                ]
                            ):
                                accept << 1
                        with ElseBlockContext():
                            accept << 1
                            StringExpression(
                                ['print("problem component: ", ', lam[i], ");\n"]
                            )

                    # Detection effects
                    event[i] << dm_rng.angular_resolution(E[i], omega)

                    # To be extended
                    event_type[i] << track_type

        sim_gen.generate_single_file()

        self._main_sim_filename = sim_gen.filename
