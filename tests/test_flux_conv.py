import pytest
import numpy as np
from astropy import units as u

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import (
    LogParabolaSpectrum,
    PowerLawSpectrum,
)
from hierarchical_nu.source.source import (
    PointSource,
    Sources,
    DetectorFrame,
    SourceFrame,
)
from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.source.flux_model import integral_power_law

Parameter.clear_registry()


class TestPrecomputation:

    @pytest.fixture
    def setup_point_source(self):
        self.redshift = 0.4
        self.src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
        self.diff_index = Parameter(2.5, "diff_index", fixed=False, par_range=(1, 4))
        self.L = Parameter(
            1.0e47 * (u.erg / u.s),
            "luminosity",
            fixed=True,
            par_range=(0, 1e60) * (u.erg / u.s),
        )
        self.diffuse_norm = Parameter(
            1.0e-13 / u.GeV / u.m**2 / u.s,
            "diffuse_norm",
            fixed=True,
            par_range=(0, np.inf),
        )
        self.Emin_src = Parameter(5e4 * u.GeV, "Emin_src", fixed=True)
        self.Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)
        self.Emin_det = Parameter(6e4 * u.GeV, "Emin_det", fixed=True)

        self.point_source = PointSource.make_powerlaw_source(
            "your_ad_here",
            np.deg2rad(5) * u.rad,
            np.pi * u.rad,
            self.L,
            self.src_index,
            self.redshift,
            self.Emin_src,
            self.Emax_src,
        )

        self.my_sources = Sources()
        self.my_sources.add(self.point_source)

    @pytest.fixture
    def setup_diff_source(self):
        self.diff_index = Parameter(2.5, "diff_index", fixed=False, par_range=(1, 4))
        self.diffuse_norm = Parameter(
            1.0e-13 / u.GeV / u.m**2 / u.s,
            "diffuse_norm",
            fixed=True,
            par_range=(0, np.inf),
        )
        self.Emin_diff = Parameter(1e4 * u.GeV, "Emin_diff", fixed=True)
        self.Emax_diff = Parameter(1e8 * u.GeV, "Emax_diff", fixed=True)

        # set redshift of diffuse spectrum to zero
        try:
            self.my_sources.add_diffuse_component(
                self.diffuse_norm,
                1e5 * u.GeV,
                self.diff_index,
                self.Emin_diff,
                self.Emax_diff,
            )
        except:
            self.my_sources = Sources()
            self.my_sources.add_diffuse_component(
                self.diffuse_norm,
                1e5 * u.GeV,
                self.diff_index,
                self.Emin_diff,
                self.Emax_diff,
            )

    def test_flux_conversion_diff_source(self, setup_diff_source):
        F = self.my_sources.diffuse._parameters["norm"].value.copy()
        F *= integral_power_law(
            self.diff_index.value, 0.0, self.Emin_diff.value, self.Emax_diff.value
        )
        assert self.my_sources.diffuse.flux_model.total_flux_int.value == pytest.approx(
            F.value
        )

    def test_flux_conversion_point_source(self, setup_point_source):
        F = self.L.value / (4 * np.pi * luminosity_distance(0.4) ** 2)
        F *= integral_power_law(
            self.src_index.value,
            0,
            self.Emin_src.value / (1 + self.redshift),
            self.Emax_src.value / (1 + self.redshift),
        )
        F /= integral_power_law(
            self.src_index.value,
            1,
            self.Emin_src.value / (1 + self.redshift),
            self.Emax_src.value / (1 + self.redshift),
        )
        assert self.point_source.flux_model.total_flux_int.value == pytest.approx(
            F.to(1 / (u.second * u.meter**2)).value, rel=1e-5
        )


def test_logparabola():
    Parameter.clear_registry()
    index = Parameter(2.0, "src_index")
    alpha = Parameter(index.value, "alpha")
    beta = Parameter(0.0, "beta")
    norm = Parameter(1e-10 / u.GeV / u.s / u.m**2, "norm")
    Emin = 1e2 * u.GeV
    Emax = 1e8 * u.GeV
    E0 = Parameter(1e5 * u.GeV, "E0_src", fixed=True)
    Enorm = 1e5 * u.GeV

    pl = PowerLawSpectrum(norm, Enorm, index, Emin, Emax)
    log = LogParabolaSpectrum(norm, E0, alpha, beta, Emin, Emax)

    factor_pl = PowerLawSpectrum.flux_conv_(
        alpha=index.value, e_low=Emin.to_value(u.GeV), e_up=Emax.to_value(u.GeV)
    )
    factor_log = log.flux_conv()

    flux_density_pl = pl.total_flux_density.to_value(u.erg / u.m**2 / u.s)
    flux_density_log = log.total_flux_density.to_value(u.erg / u.m**2 / u.s)

    integral_pl = pl.integral(Emin, Emax).to_value(1 / u.m**2 / u.s)
    integral_log = log.integral(Emin, Emax).to_value(1 / u.m**2 / u.s)

    assert integral_pl == pytest.approx(integral_log)

    assert factor_pl == pytest.approx(factor_log)

    assert flux_density_pl == pytest.approx(flux_density_log)
