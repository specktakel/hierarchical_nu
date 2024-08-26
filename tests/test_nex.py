import numpy as np
import astropy.units as u
import pytest

from hierarchical_nu.utils.roi import RectangularROI, ROIList
from hierarchical_nu.detector.icecube import IC86_II

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
indices = np.linspace(1.8, 3.6, 10, endpoint=True)

min_energy = 1e5
max_energy = 1e8
norm_energy = 1e5
diff_norm = 1.0e-13 / u.GeV / u.m**2 / u.s


class TestNex:
    @pytest.fixture
    def setup_hnu_ps(self):
        from hierarchical_nu.source.parameter import Parameter
        from hierarchical_nu.source.source import Sources, PointSource
        from hierarchical_nu.simulation import Simulation

        Parameter.clear_registry()
        ROIList.clear_registry()
        src_index = Parameter(2.8, "src_index", fixed=False, par_range=(1, 4))
        L = Parameter(
            1.0e47 * (u.erg / u.s),
            "luminosity",
            fixed=True,
            par_range=(0, 1e60) * (u.erg / u.s),
        )
        Emin = Parameter(min_energy * u.GeV, "Emin", fixed=True)
        Emax = Parameter(max_energy * u.GeV, "Emax", fixed=True)
        Emin_src = Parameter(
            Emin.value * 1.4, "Emin_src", fixed=True
        )  # correct for redshift
        Emax_src = Parameter(Emax.value * 1.4, "Emax_src", fixed=True)
        Emin_det = Parameter(1.0e1 * u.GeV, "Emin_det", fixed=True)

        ps_hnu = PointSource.make_powerlaw_source(
            "test",
            np.deg2rad(-30) * u.rad,
            np.pi * u.rad,
            L,
            src_index,
            0.4,
            Emin_src,
            Emax_src,
        )

        my_sources = Sources()
        my_sources.add(ps_hnu)
        roi = RectangularROI()
        logger.warning(roi)
        sim = Simulation(my_sources, IC86_II, 1 * u.year)
        sim.precomputation()

        return (my_sources, sim)

    @pytest.fixture
    def setup_it_nu_calc(self):
        from icecube_tools.detector.effective_area import EffectiveArea
        from icecube_tools.detector.energy_resolution import EnergyResolution
        from icecube_tools.detector.detector import IceCube
        from icecube_tools.source.flux_model import PowerLawFlux
        from icecube_tools.source.source_model import DiffuseSource, PointSource
        from icecube_tools.detector.r2021 import R2021IRF
        from icecube_tools.neutrino_calculator import NeutrinoCalculator, PhiSolver

        aeff = EffectiveArea.from_dataset("20210126", "IC86_II")

        point_flux_norm = 1e-19
        point_power_law = PowerLawFlux(
            point_flux_norm, norm_energy, 2.8, min_energy, max_energy
        )
        ps_it = PointSource(point_power_law, z=0.0, coord=(np.pi, np.deg2rad(-30)))
        sources = [ps_it]
        nu_calc = NeutrinoCalculator(sources, aeff)
        return nu_calc

    def test_rate(self, setup_it_nu_calc, setup_hnu_ps):
        self._nu_calc = setup_it_nu_calc
        self._hnu_sources, self._hnu_sim = setup_hnu_ps
        nu_calc = self._nu_calc
        sim = self._hnu_sim
        ps_hnu = self._hnu_sources[0]
        norm = (
            ps_hnu.flux_model._parameters["norm"]
            .value.to(1 / (u.GeV * u.cm**2 * u.s))
            .value
        )
        nu_calc._sources[0].flux_model._normalisation = norm

        for idx in indices:
            ps_hnu._parameters["index"].value = idx
            nex_hnu = sim._exposure_integral[IC86_II].calculate_rate(ps_hnu).value
            nu_calc._sources[0].flux_model._index = idx
            nex_it = nu_calc(
                time=1,  # years
                min_energy=min_energy,
                max_energy=max_energy,  # energy range
                min_cosz=-1,
                max_cosz=1,
            )[0] / (
                365 * 24 * 3600
            )  # cos(zenith)

            assert nex_hnu == pytest.approx(nex_it, rel=9e-2)

    @pytest.fixture
    def calc_hnu_nex(self):
        from hierarchical_nu.source.parameter import Parameter
        from hierarchical_nu.source.source import Sources, PointSource
        from hierarchical_nu.simulation import Simulation
        from hierarchical_nu.source.cosmology import luminosity_distance as dl
        from hierarchical_nu.source.flux_model import integral_power_law as ipl

        Parameter.clear_registry()
        ROIList.clear_registry()
        src_index = Parameter(2.8, "src_index", fixed=False, par_range=(1, 4))
        diff_index = Parameter(2.1, "diff_index", fixed=False, par_range=(1.5, 4))
        L = Parameter(
            4.0e47 * (u.erg / u.s),
            "luminosity",
            fixed=True,
            par_range=(0, 1e60) * (u.erg / u.s),
        )
        diffuse_norm = Parameter(
            diff_norm, "diffuse_norm", fixed=True, par_range=(0, np.inf)
        )
        Emin = Parameter(1e5 * u.GeV, "Emin", fixed=True)
        Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
        Emin_diff = Parameter(1e5 * u.GeV, "Emin_diff", fixed=True)
        Emax_diff = Parameter(1e8 * u.GeV, "Emax_diff", fixed=True)
        Emin_src = Parameter(Emin.value * 1.4, "Emin_src", fixed=True)
        Emax_src = Parameter(Emax.value * 1.4, "Emax_src", fixed=True)
        Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
        Emin_det = Parameter(1.0e1 * u.GeV, "Emin_det", fixed=True)

        ps_hnu = PointSource.make_powerlaw_source(
            "test",
            np.deg2rad(-30) * u.rad,
            np.pi * u.rad,
            L,
            src_index,
            0.4,
            Emin_src,
            Emax_src,
        )

        my_sources = Sources()
        my_sources.add(ps_hnu)

        my_sources.add_diffuse_component(
            diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff
        )
        roi = RectangularROI()
        logger.warning(roi)
        sim = Simulation(my_sources, IC86_II, 1 * u.year)
        sim.precomputation()
        Nex_ps_hnu = []
        Nex_diff_hnu = []
        calculated_norm = []

        for c, idx in enumerate(indices):
            src_index.value = idx
            diff_index.value = idx

            _ = sim._get_expected_Nnu(sim._get_sim_inputs())
            Nex_ps_hnu.append(sim._expected_Nnu_per_comp[0])
            Nex_diff_hnu.append(sim._expected_Nnu_per_comp[1])

            calculated_norm.append(
                (
                    L.value
                    / 4.0
                    / np.pi
                    / dl(0.4) ** 2
                    / ipl(
                        src_index.value,
                        1.0,
                        Emin.value,
                        Emax.value,
                        Enorm.value,
                    )
                ).to(1 / (u.GeV * u.m**2 * u.s))
            )

        self._hnu_ps_norm = calculated_norm
        self._Nex_ps_hnu = np.array(Nex_ps_hnu)
        self._Nex_diff_hnu = np.array(Nex_diff_hnu)

    @pytest.fixture
    def calc_it_nex(self, calc_hnu_nex):
        from icecube_tools.detector.effective_area import EffectiveArea
        from icecube_tools.detector.energy_resolution import EnergyResolution
        from icecube_tools.detector.detector import IceCube
        from icecube_tools.source.flux_model import PowerLawFlux
        from icecube_tools.source.source_model import DiffuseSource, PointSource
        from icecube_tools.detector.r2021 import R2021IRF
        from icecube_tools.neutrino_calculator import NeutrinoCalculator, PhiSolver
        from hierarchical_nu.detector.r2021 import R2021DetectorModel
        from hierarchical_nu.simulation import Simulation

        aeff = EffectiveArea.from_dataset("20210126", "IC86_II")
        Nex_ps_it = []
        Nex_diff_it = []

        for idx, norm_ps in zip(indices, self._hnu_ps_norm):
            point_flux_norm = norm_ps.to(1 / (u.GeV * u.cm**2 * u.s)).value
            point_power_law = PowerLawFlux(
                point_flux_norm, norm_energy, idx, min_energy, max_energy
            )
            point_source = PointSource(
                point_power_law, z=0.0, coord=(np.pi, np.deg2rad(-30))
            )

            diff_flux_norm = diff_norm.to(1 / (u.GeV * u.cm**2 * u.s)).value / (
                4 * np.pi
            )
            diff_power_law = PowerLawFlux(
                diff_flux_norm, norm_energy, idx, min_energy, max_energy
            )
            diff_source = DiffuseSource(diff_power_law, z=0.0)
            sources = [point_source, diff_source]
            nu_calc = NeutrinoCalculator(sources, aeff)
            n = nu_calc(
                time=1,  # years
                min_energy=min_energy,
                max_energy=max_energy,  # energy range
                min_cosz=-1,
                max_cosz=1,
            )  # cos(zenith) range
            Nex_ps_it.append(n[0])
            Nex_diff_it.append(n[1])

        self._Nex_ps_it = np.array(Nex_ps_it)
        self._Nex_diff_it = np.array(Nex_diff_it)

    def test_nex(self, calc_it_nex):
        assert self._Nex_ps_it == pytest.approx(self._Nex_ps_hnu, rel=1e-1)
        assert self._Nex_diff_it == pytest.approx(self._Nex_diff_hnu, rel=1e-1)
