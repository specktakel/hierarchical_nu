import pytest
import numpy as np
from astropy import units as u

from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
from hierarchical_nu.source.flux_model import integral_power_law

Parameter.clear_registry()


def test_flux_calculation():
    src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
    diff_index = Parameter(2.5, "diff_index", fixed=False, par_range=(1, 4))
    L = Parameter(1.0e47 * (u.erg / u.s), "luminosity", fixed=True, 
                par_range=(0, 1e60)*(u.erg/u.s))
    diffuse_norm = Parameter(1.0e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                            par_range=(0, np.inf))
    Enorm_src = Parameter(1e5 * u.GeV, "Enorm_src", fixed=True)
    Emin = Parameter(1e4 * u.GeV, "Emin", fixed=True)
    Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
    Emin_src = Parameter(5e4 * u.GeV, "Emin_src", fixed=True)
    Emax_src = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)
    Emin_det = Parameter(6e4 * u.GeV, "Emin_det", fixed=True)

    point_source = PointSource.make_powerlaw_source("test", np.deg2rad(5)*u.rad,
                                                    np.pi*u.rad, 
                                                    L, src_index, 0.4, Emin_src, Emax_src, Enorm_src)

    my_sources = Sources()
    my_sources.add(point_source)


    F = L.value / (4 * np.pi * luminosity_distance(0.4)**2)
    F *= integral_power_law(src_index.value, 0, Enorm_src.value / (1 + 0.4), Emin_src.value / (1 + 0.4), Emax_src.value / (1 + 0.4))
    F /= integral_power_law(src_index.value, 1, Enorm_src.value / (1 + 0.4), Emin_src.value / (1 + 0.4), Emax_src.value / (1 + 0.4))
    assert point_source.flux_model.total_flux_int.value == pytest.approx(F.to(1/(u.second * u.meter**2)).value, rel=1e-5)