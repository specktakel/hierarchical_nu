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
    Ebreak = Parameter(pgamma.Ebreak, "Ebreak")
    logp = LogParabolaSpectrum(norm, E0, alpha, beta, Ebreak.value, Emax.value)
    pl_norm = Parameter(logp(Ebreak.value), "pl_norm")
    pl = PowerLawSpectrum(pl_norm, Ebreak.value, index, Emin.value, Ebreak.value)

    # test flux values
    E = np.geomspace(1e4, 1e8, 1_000) << u.GeV
    flux_units = 1 / u.GeV / u.s / u.m**2
    flux_pgamma = pgamma(E).to_value(flux_units)

    flux_pl = pl(E).to_value(flux_units)

    flux_logp = logp(E).to_value(flux_units)

    assert pytest.approx(flux_pgamma) == flux_pl + flux_logp

    flux_units = 1 / u.m**2 / u.s
    assert pytest.approx(
        pgamma.integral(Emin.value, Ebreak.value).to_value(flux_units)
    ) == pl.integral(Emin.value, Ebreak.value).to_value(flux_units)

    assert pytest.approx(
        pgamma.integral(Ebreak.value, Emax.value).to_value(flux_units)
    ) == logp.integral(Ebreak.value, Emax.value).to_value(flux_units)

    assert pytest.approx(
        pgamma.integral(Emin.value, Emax.value).to_value(flux_units)
    ) == logp.integral(Ebreak.value, Emax.value).to_value(flux_units) + pl.integral(
        Emin.value, Ebreak.value
    ).to_value(
        flux_units
    )

    flux_units = u.erg / u.cm**2 / u.s
    assert pytest.approx(pgamma.total_flux_density.to_value(flux_units)) == (
        logp.total_flux_density + pl.total_flux_density
    ).to_value(flux_units)


"""
def test_stan():
    Parameter.clear_registry()
    index = Parameter(-2.0, "src_index")
    alpha = Parameter(0.0, "alpha")
    beta = Parameter(0.8, "beta")
    E0 = Parameter(1e7 * u.GeV, "E0_src")

    norm = Parameter(2e-10 / u.GeV / u.s / u.m**2, "norm")

    Emin = Parameter(1e2 * u.GeV, "Emin_src")
    Emax = Parameter(1e9 * u.GeV, "Emax_src")

    pgamma = PGammaSpectrum(norm, E0, Emin.value, Emax.value)
    Ebreak = Parameter(pgamma.Ebreak, "Ebreak")
    logp = LogParabolaSpectrum(norm, E0, alpha, beta, Ebreak.value, Emax.value)
    pl_norm = Parameter(logp(Ebreak.value), "pl_norm")
    pl = PowerLawSpectrum(pl_norm, Ebreak.value, index, Emin.value, Ebreak.value)

    with StanFileGenerator("pgamma_test") as gc:
        PGammaSpectrum.make_stan_utility_func(False, False, False)
        PGammaSpectrum.make_stan_flux_conv_func("flux_conv", False, False, False)
        PGammaSpectrum.make_stan_lpdf_func("src_spectrum_pdf", False, False, False)
        with DataContext():
            Emin = ForwardVariableDef("E_min", "real")
            Emax = ForwardVariableDef("E_max", "real")
            N = ForwardVariableDef("N", "int")
            energy = ForwardArrayDef("energy", "real", ["[", N, "]"])
            E0 = ForwardVariableDef("E0", "real")
        with GeneratedQuantitiesContext():
            lpdf = ForwardArrayDef("lpdf", "real"["[", N, "]"])
            conv = ForwardArrayDef("conv", "real"["[", N, "]"])
            with ForLoopContext(1, N, "i") as i:
                lpdf[i] << StringExpression(
                    [
                        "src_spectrum_pdf(energy[",
                        i,
                        "| {",
                        E0,
                        "}, {",
                        Emin,
                        ",",
                        Emax,
                        "}, {0}]",
                    ]
                )
                conv[i] << StringExpression(
                    ["flux_conv({", energy[i], "}, {", Emin, ",", Emax, "}, {0})"]
                )

        file = gc.generate_single_file()

    model = CmdStanModel(stan_file=file)
    N = 100
    Emin = 1e3
    Emax = 1e8
    E0 = 1e6
    energy = np.geomspace(Emin, Emax, N)
    data = {
        "E0": E0,
        "energy": energy,
        "Emin": Emin,
        "Emax": Emax,
        "N": N,
    }
    samples = model.sample(fixed_param=True, data=data, iter_warmup=1, iter_sampling=1)
"""
