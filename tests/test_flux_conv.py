import pytest
import numpy as np
from astropy import units as u

from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
from hierarchical_nu.source.flux_model import integral_power_law

Parameter.clear_registry()

class TestPrecomputation():

    @pytest.fixture
    def setup_point_source(self):
        self.redshift = 0.4
        self.src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
        self.diff_index = Parameter(2.5, "diff_index", fixed=False, par_range=(1, 4))
        self.L = Parameter(1.0e47 * (u.erg / u.s), "luminosity", fixed=True, 
                    par_range=(0, 1e60)*(u.erg/u.s))
        self.diffuse_norm = Parameter(1.0e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                                par_range=(0, np.inf))
        self.Epivot = Parameter(1e5 * u.GeV, "Epviot", fixed=True)
        self.Emin_src = Parameter(5e4 * u.GeV, "Emin_src", fixed=True)
        self.Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)
        self.Emin_det = Parameter(6e4 * u.GeV, "Emin_det", fixed=True)

        self.point_source = PointSource.make_powerlaw_source(
            "your_ad_here",
            np.deg2rad(5)*u.rad,
            np.pi*u.rad, 
            self.L,
            self.src_index,
            self.redshift,
            self.Emin_src,
            self.Emax_src,
            self.Epivot
        )

        self.my_sources = Sources()
        self.my_sources.add(self.point_source)


    @pytest.fixture
    def setup_diff_source(self):
        self.diff_index = Parameter(2.5, "diff_index", fixed=False, par_range=(1, 4))
        self.diffuse_norm = Parameter(1.0e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                                par_range=(0, np.inf))
        self.Enorm_src = Parameter(1e5 * u.GeV, "Enorm_src", fixed=True)
        self.Emin = Parameter(1e4 * u.GeV, "Emin", fixed=True)
        self.Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)

        try:
            self.my_sources.add_diffuse_component(self.diffuse_norm, 1e5*u.GeV, self.diff_index)
        except:
            self.my_sources = Sources()
            self.my_sources.add_diffuse_component(self.diffuse_norm, 1e5*u.GeV, self.diff_index)

        
    def test_flux_conversion_diff_source(self, setup_diff_source):
        F = self.my_sources.diffuse._parameters["norm"].value.copy()
        print(F)
        F *= integral_power_law(
            self.diff_index.value,
            0.,
            self.Enorm_src.value,
            self.Emin.value,
            self.Emax.value
        )
        print(F)
        assert self.my_sources.diffuse.flux_model.total_flux_int.value == pytest.approx(
            F.value
        )


    def test_flux_conversion_point_source(self, setup_point_source):
        F = self.L.value / (4 * np.pi * luminosity_distance(0.4)**2)
        F *= integral_power_law(
            self.src_index.value,
            0,
            self.Epivot.value / (1 + self.redshift),
            self.Emin_src.value / (1 + self.redshift),
            self.Emax_src.value / (1 + self.redshift)
        )
        F /= integral_power_law(
            self.src_index.value,
            1,
            self.Epivot.value / (1 + self.redshift),
            self.Emin_src.value / (1 + self.redshift),
            self.Emax_src.value / (1 + self.redshift)
        )
        assert self.point_source.flux_model.total_flux_int.value == pytest.approx(
            F.to(1/(u.second * u.meter**2)).value, rel=1e-5
        )