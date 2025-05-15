import numpy as np
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import (
    LogParabolaSpectrum,
    PowerLawSpectrum,
    PGammaSpectrum,
)

from hierarchical_nu.backend.stan_generator import (
    DataContext,
    GeneratedQuantitiesContext,
    ForLoopContext,
    StanFileGenerator,
)
from hierarchical_nu.backend import StringExpression

from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
)

from cmdstanpy import CmdStanModel

import pytest


def test_pythonic():
    Parameter.clear_registry()
    index = Parameter(0.0, "src_index")
    alpha = Parameter(0.0, "alpha")
    beta = Parameter(0.7, "beta")
    E0 = Parameter(1e7 * u.GeV, "E0_src")

    norm = Parameter(2e-10 / u.GeV / u.s / u.m**2, "norm")

    Emin = Parameter(1e2 * u.GeV, "Emin_src")
    Emax = Parameter(1e9 * u.GeV, "Emax_src")

    pgamma = PGammaSpectrum(norm, E0, Emin.value, Emax.value)
    logp = LogParabolaSpectrum(norm, E0, alpha, beta, E0.value, Emax.value)
    pl_norm = Parameter(logp(E0.value), "pl_norm")
    pl = PowerLawSpectrum(pl_norm, E0.value, index, Emin.value, E0.value)

    # test flux values
    E = np.geomspace(1e4, 1e8, 1_000) << u.GeV
    flux_units = 1 / u.GeV / u.s / u.m**2

    assert pl_norm.value.to_value(flux_units) == norm.value.to_value(flux_units)

    flux_pgamma = pgamma(E).to_value(flux_units)

    flux_pl = pl(E).to_value(flux_units)

    flux_logp = logp(E).to_value(flux_units)

    assert pytest.approx(flux_pgamma) == flux_pl + flux_logp

    flux_units = 1 / u.m**2 / u.s
    assert pytest.approx(
        pgamma.integral(Emin.value, E0.value).to_value(flux_units)
    ) == pl.integral(Emin.value, E0.value).to_value(flux_units)

    assert pytest.approx(
        pgamma.integral(E0.value, Emax.value).to_value(flux_units)
    ) == logp.integral(E0.value, Emax.value).to_value(flux_units)

    assert pytest.approx(
        pgamma.integral(Emin.value, Emax.value).to_value(flux_units)
    ) == logp.integral(E0.value, Emax.value).to_value(flux_units) + pl.integral(
        Emin.value, E0.value
    ).to_value(
        flux_units
    )

    flux_units = u.erg / u.cm**2 / u.s
    assert pytest.approx(pgamma.total_flux_density.to_value(flux_units)) == (
        logp.total_flux_density + pl.total_flux_density
    ).to_value(flux_units)

    assert pytest.approx(pgamma.flux_conv().to_value(1 / u.GeV)) == pgamma.flux_conv_(
        e_low=Emin.value.to_value(u.GeV),
        e_up=Emax.value.to_value(u.GeV),
        e_0=E0.value.to_value(u.GeV),
    )


def test_satanic():
    Parameter.clear_registry()
    index = Parameter(-2.0, "src_index")
    alpha = Parameter(0.0, "alpha")
    beta = Parameter(0.8, "beta")
    E0 = Parameter(1e6 * u.GeV, "E0_src")

    norm = Parameter(2e-10 / u.GeV / u.s / u.m**2, "norm")

    Emin = Parameter(1e2 * u.GeV, "Emin_src")
    Emax = Parameter(1e9 * u.GeV, "Emax_src")

    pgamma = PGammaSpectrum(norm, E0, Emin.value, Emax.value)
    logp = LogParabolaSpectrum(norm, E0, alpha, beta, E0.value, Emax.value)
    pl_norm = Parameter(logp(E0.value), "pl_norm")
    pl = PowerLawSpectrum(pl_norm, E0.value, index, Emin.value, E0.value)

    base_path = ".stan_files/pgamma_test"

    with StanFileGenerator(base_path) as gc:
        PGammaSpectrum.make_stan_utility_func(False, False, False)
        PGammaSpectrum.make_stan_flux_conv_func("flux_conv", False, False, False)
        PGammaSpectrum.make_stan_lpdf_func("src_spectrum_pdf", False, False, False)
        with DataContext():
            N = ForwardVariableDef("N", "int")
            energy = ForwardArrayDef("energy", "real", ["[", N, "]"])
            E0_src = ForwardVariableDef("E0", "real")
        with GeneratedQuantitiesContext():
            lpdf = ForwardArrayDef("lpdf", "real", ["[", N, "]"])
            conv = ForwardArrayDef("conv", "real", ["[", N, "]"])
            with ForLoopContext(1, N, "i") as i:
                lpdf[i] << StringExpression(
                    [
                        "src_spectrum_pdf(energy[",
                        i,
                        "], {1.}, {",
                        E0_src,
                        ", ",
                        1e2,
                        ",",
                        1e9,
                        "}, {0})",
                    ]
                )
                conv[i] << StringExpression(
                    ["flux_conv({0.}, {", energy[i], ",", 1e2, ",", 1e9, "}, {0})"]
                )

        gc.generate_single_file()
        stan_file = gc.filename

    model = CmdStanModel(stan_file=stan_file)
    N = 100
    Emin = 1e3
    Emax = 1e8
    energy = np.geomspace(Emin, Emax, N)
    data = {
        "E0": E0.value.to_value(u.GeV),
        "energy": energy,
        "Emin": Emin,
        "Emax": Emax,
        "N": N,
    }
    samples = model.sample(
        fixed_param=True, data=data, iter_warmup=1, iter_sampling=1, chains=1
    )

    stan_conv = samples.stan_variable("conv").squeeze()
    stan_lpdf = samples.stan_variable("lpdf").squeeze()

    lpdf = np.zeros_like(energy)
    conv = np.zeros_like(energy)

    for c, E in enumerate(energy):
        lpdf[c] = np.log(pgamma.pdf(E * u.GeV, *pgamma.energy_bounds))

    # Abuse energy here as cutoff energy, don't even try to test at the boundaries...
    for c, E in enumerate(energy):
        E0.value = E * u.GeV
        conv[c] = pgamma.flux_conv().to_value(1 / u.GeV)

    assert pytest.approx(conv, rel=8e-2) == stan_conv
    assert pytest.approx(lpdf, rel=8e-2) == stan_lpdf
