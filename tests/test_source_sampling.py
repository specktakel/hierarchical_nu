import numpy as np
from astropy import units as u
import pytest
import os
from cmdstanpy import CmdStanModel

from hierarchical_nu.source.source import PointSource
from hierarchical_nu.source.flux_model import (
    PowerLawSpectrum,
    IsotropicDiffuseBG,
)
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.atmospheric_flux import AtmosphericNuMuFlux
from hierarchical_nu.backend.stan_generator import (
    ParametersContext,
    TransformedParametersContext,
    ModelContext,
    GeneratedQuantitiesContext,
    FunctionsContext,
    Include,
    StanFileGenerator,
    FunctionCall,
)
from hierarchical_nu.backend.variable_definitions import (
    ParameterDef,
    ForwardVariableDef,
)

from hierarchical_nu.backend.expression import StringExpression

from hierarchical_nu.stan.interface import STAN_PATH

from hierarchical_nu.detector.input import mceq

Parameter.clear_registry()

index = Parameter(2.0, "index", fixed=False, par_range=(1.0, 4))
Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)

z = 1.0
Emin_src = Parameter(Emin.value * (1.0 + z), "Emin_src", fixed=True)
Emax_src = Parameter(Emax.value * (1.0 + z), "Emax_src", fixed=True)

Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)


def make_point_source():
    lumi = Parameter(
        5e51 * (u.erg / u.s),
        "luminosity",
        fixed=True,
        par_range=(0, 1e60),
    )

    source = PointSource.make_powerlaw_source(
        "test",
        1 * u.rad,
        2 * u.rad,
        lumi,
        index,
        1,
        Emin_src,
        Emax_src,
    )

    return source


def make_diffuse_flux():
    diffuse_norm = Parameter(
        1.44e-12 / u.GeV / u.m**2 / u.s,
        "diffuse_norm",
        fixed=True,
        par_range=(0, np.inf),
    )

    diffuse_flux_model = IsotropicDiffuseBG(
        PowerLawSpectrum(
            diffuse_norm,
            Enorm.value,
            index,
            Emin_diff.value,
            Emax_diff.value,
        )
    )

    return diffuse_flux_model


source = make_point_source()
diffuse_flux_model = make_diffuse_flux()


def test_atmo_flux():
    atmo_bg_flux = AtmosphericNuMuFlux(1e2 * u.GeV, 1e9 * u.GeV, cache_dir=mceq)

    F = atmo_bg_flux.total_flux_int

    flux_unit = 1 / (u.s * u.m**2)

    assert F.to(flux_unit).value == pytest.approx(0.3139, rel=0.025)


def test_diffuse_flux():
    F = diffuse_flux_model.total_flux_int

    flux_unit = 1 / (u.s * u.m**2)

    assert F.to(flux_unit).value == pytest.approx(0.0001439998)


def test_point_source_flux():
    F = source.flux_model.total_flux_int

    flux_unit = 1 / (u.s * u.m**2)

    assert F.to(flux_unit).value == pytest.approx(0.0043180246)


def generate_source_test_code(output_directory):
    file_name = os.path.join(output_directory, "source_sample")

    atmo_bg_flux = AtmosphericNuMuFlux(1e2 * u.GeV, 1e9 * u.GeV, cache_dir=mceq)

    with StanFileGenerator(file_name) as code_gen:
        with FunctionsContext():
            _ = Include("interpolation.stan")
            _ = Include("utils.stan")

            spectrum_rng = source.flux_model.spectral_shape.make_stan_sampling_func(
                "spectrum_rng"
            )
            diffuse_flux_rng = diffuse_flux_model.make_stan_sampling_func(
                "diffuse_bg_rng"
            )
            atmu_nu_flux = atmo_bg_flux.make_stan_function(theta_points=30)

        with ParametersContext():
            energy = ParameterDef("energy", "real", 1e2, 1e9)
            coszen = ParameterDef("coszen", "real", -1, 1)

        with TransformedParametersContext():
            omega = ForwardVariableDef("omega", "vector[3]")
            zen = ForwardVariableDef("zen", "real")
            sinzen = ForwardVariableDef("sinzen", "real")

            zen << FunctionCall([coszen], "acos")
            sinzen << FunctionCall([zen], "sin")

            omega[1] << sinzen
            omega[2] << 0
            omega[3] << coszen

        with ModelContext():
            logflux = FunctionCall([atmu_nu_flux(energy, omega)], "log")
            StringExpression(["target += ", logflux])

        with GeneratedQuantitiesContext():
            pl_samples = ForwardVariableDef("pl_samples", "real")
            diffuse_events = ForwardVariableDef("diffuse_events", "vector[3]")

            pl_samples << spectrum_rng(2, 1e2, 1e9)
            diffuse_events << diffuse_flux_rng(2, 1e2, 1e9)

        code_gen.generate_single_file()

        return code_gen.filename


def test_source_sampling(output_directory, random_seed):
    stan_file = generate_source_test_code(output_directory)

    stanc_options = {"include-paths": [STAN_PATH]}

    stan_model = CmdStanModel(
        stan_file=stan_file,
        stanc_options=stanc_options,
    )

    output = stan_model.sample(data={}, iter_sampling=10000, chains=1, seed=random_seed)

    diffuse_events = output.stan_variable("diffuse_events")

    atmo_energy = output.stan_variable("energy")

    atmo_coszen = output.stan_variable("coszen")

    pl_samples = output.stan_variable("pl_samples")

    # assert np.mean(pl_samples) == pytest.approx(1495.5050349)

    # assert np.mean(np.sin(diffuse_events[:, 1])) == pytest.approx(-0.03752253, 0.1)

    # assert np.mean(atmo_energy) == pytest.approx(186.838111, 0.1)

    # assert np.mean(atmo_coszen) == pytest.approx(-0.012845337, 0.001)

    # Compare atmo with true spectrum
    atmo_bg_flux = AtmosphericNuMuFlux(1e2 * u.GeV, 1e9 * u.GeV, cache_dir=mceq)

    energies = np.logspace(2, 9, 100) << u.GeV

    log_bins = np.linspace(2, 4, 50)

    integrate_per_log = (
        atmo_bg_flux.integral(
            10 ** log_bins[:-1] * u.GeV,
            10 ** log_bins[1:] * u.GeV,
            (-np.pi / 2) * u.rad,
            (np.pi / 2) * u.rad,
            0 * u.rad,
            2 * np.pi * u.rad,
        )
        / np.diff(log_bins)
        / atmo_bg_flux.total_flux_int
    )

    fluxes = atmo_bg_flux.total_flux(energies) / atmo_bg_flux.total_flux_int

    E_hist, _ = np.histogram(
        np.log10(atmo_energy),
        bins=np.log10(energies.value),
        density=True,
    )

    assert max(E_hist) == pytest.approx(max(integrate_per_log.value), 0.15)
