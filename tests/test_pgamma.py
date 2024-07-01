import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import (
    LogParabolaSpectrum,
    PowerLawSpectrum,
    PGammaSpectrum,
)

import pytest


def test_pythonic():
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


def test_stan():
    pass
