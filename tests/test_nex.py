import numpy as np
import astropy.units as u
import pytest


indices = np.linspace(1.8, 3.6, 10, endpoint=True)

min_energy = 1e5
max_energy = 1e8
norm_energy = 1e5

class TestNex():

    
    @pytest.fixture
    def setup_hnu_ps(self):

        from hierarchical_nu.source.parameter import Parameter
        from hierarchical_nu.source.source import Sources, PointSource
        from hierarchical_nu.simulation import Simulation
        from hierarchical_nu.detector.r2021 import R2021DetectorModel

        Parameter.clear_registry()
        src_index = Parameter(2.8, "src_index", fixed=False, par_range=(1, 4))
        L = Parameter(1.0e47 * (u.erg / u.s), "luminosity", fixed=True, 
                    par_range=(0, 1e60)*(u.erg/u.s))
        Emin = Parameter(min_energy * u.GeV, "Emin", fixed=True)
        Emax = Parameter(max_energy * u.GeV, "Emax", fixed=True)
        Emin_src = Parameter(Emin.value*1.4, "Esrc_min", fixed=True)   # correct for redshift
        Emax_src = Parameter(Emax.value*1.4, "Esrc_max", fixed=True)
        Epivot = Parameter(norm_energy*1.4 * u.GeV, "Epivot", fixed=True)
        Emin_det = Parameter(1.e1 * u.GeV, "Emin_det", fixed=True)

        ps_hnu = PointSource.make_powerlaw_source("test", np.deg2rad(-30)*u.rad,
                                                        np.pi*u.rad, 
                                                        L, src_index, 0.4, Emin_src, Emax_src, Epivot)



        my_sources = Sources()
        my_sources.add(ps_hnu)

        sim = Simulation(my_sources, R2021DetectorModel, 1*u.year)
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
        irf = R2021IRF.from_period("IC86_II")
        #just to be able to be used as both
        detector = IceCube(aeff, irf, irf, "IC86_II")

        # Define simple sources (see source model notebook for more info)
        point_flux_norm = 1e-19   # 1/GeV/cm**2/s, will be overridden later

        point_power_law = PowerLawFlux(point_flux_norm, norm_energy, 2.8, 
                                min_energy, max_energy)
        ps_it = PointSource(point_power_law, z=0., coord=(np.pi, np.deg2rad(-30)))
        sources = [ps_it]
        nu_calc = NeutrinoCalculator(sources, aeff)
        return nu_calc

    def test_rate(self, setup_it_nu_calc, setup_hnu_ps):

        self._nu_calc = setup_it_nu_calc
        self._hnu_sources, self._hnu_sim = setup_hnu_ps
        nu_calc = self._nu_calc
        sim = self._hnu_sim
        ps_hnu = self._hnu_sources[0]
        norm = ps_hnu.flux_model._parameters["norm"].value.to(1/(u.GeV*u.cm**2*u.s)).value
        nu_calc._sources[0].flux_model._normalisation = norm

        for idx in indices:
            ps_hnu._parameters["index"].value = idx
            #ps_hnu._parameters["norm"].fixed = False
            #ps_hnu._parameters["norm"].value = norm
            #ps_hnu._parameters["norm"].fixed = True
            nex_hnu = sim._exposure_integral["tracks"].calculate_rate(ps_hnu).value
            nu_calc._sources[0].flux_model._index = idx
            nex_it = nu_calc(time=1, # years
                min_energy=min_energy, max_energy=max_energy, # energy range
                min_cosz=-1, max_cosz=1)[0] / (365*24*3600) # cos(zenith)
            
            assert nex_hnu == pytest.approx(nex_it, rel=1e-2)
            
