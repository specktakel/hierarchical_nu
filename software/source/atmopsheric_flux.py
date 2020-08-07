from MCEq.core import config, MCEqRun
import crflux.models as crf
# matplotlib used plotting. Not required to run the code.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import scipy
import pickle

from ..backend import UserDefinedFunction
from .cache import Cache
from .source.flux_model import FluxModel

Cache.set_cache_dir(".cache")

class AtmosphericNuMuFlux(FluxModel, UserDefinedFunction):
    
    CACHE_FNAME = "mceq_flux.pickle"
    EMAX = 1E9
    EMIN = 1
    THETA_BINS = 50
    ENERGY_POINTS = 100
    
    def __init__(self):
         UserDefinedFunction.__init__(
            self,
            "AtmopshericNumuFlux",
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real")
            
        FluxModel.__init__(self)
        
        self._setup()
        
        theta_grid = np.degrees(np.arccos(np.linspace(0, 1, self.THETA_BINS)))
        energy_grid = np.linspace(self.EMIN, self.EMAX, self.ENERGY_POINTS)
        
        
        with self:
            truncated_e = TruncatedParameterization(
                "true_energy", self.EMIN, self.EMAX)
            log_trunc_e = LogParameterization(truncated_e)
            
            
            hist = SimpleHistogram(
                self._eff_area,
                [self._tE_bin_edges, self._cosz_bin_edges],
                "NorthernTracksEffAreaHist")

            # z = cos(theta)
            cos_dec = "cos(pi() - acos(true_dir[3]))"
            # cos_dir = FunctionCall(["true_dir"], "cos")
            _ = ReturnStatement([hist("true_energy", cos_dir)])
        
        
        
        
      
    def _setup(self):
        if self.CACHE_FNAME in Cache:
            with Cache.open(self.CACHE_FNAME, "rb") as fr:
                self._flux_spline = pickle.load(fr)
        else:
            mceq = MCEqRun(
                # High-energy hadronic interaction model
                interaction_model='SIBYLL23C',

                # cosmic ray flux at the top of the atmosphere
                primary_model = (crf.HillasGaisser2012, 'H3a'),

                # zenith angle
                theta_deg = 0., 
                )

            theta_grid = np.degrees(np.arccos(np.linspace(0, 1, self.THETA_BINS)))
            numu_fluxes = []
            for theta in theta_grid:
                mceq.set_theta_deg(theta)
                mceq.solve()
                numu_fluxes.append(
                    (mceq.get_solution('numu') +
                     mceq.get_solution('antinumu')))

            emask = (mceq.e_grid < self.EMAX) & (mceq.e_grid > self.EMIN)
            splined_flux = scipy.interpolate.RectBivariateSpline(
                np.cos(np.radians(theta_grid)),
                np.log10(mceq.e_grid[emask]),
                np.log10(numu_fluxes)[:, emask],
                )
            self._flux_spline = splined_flux
            with Cache.open(self.CACHE_FNAME, "wb") as fr:
                pickle.dump(splined_flux, fr)

    def spectrum(self, energy, dec):
        energy = np.atleast_1d(energy)
        if np.any((energy > self.EMAX) | (energy < self.EMIN)):
            raise ValueError("Energy needs to be in {} < E {}".format(self.EMIN, self.EMAX))
        
        dec = np.atleast_1d(dec)
        zenith = np.pi/2-dec
        zenith[zenith > np.pi/2] = np.pi - zenith[zenith > np.pi/2]
        zenith[zenith < 0] = -zenith[zenith < 0]
        
       
        try:
            result = self._flux_spline(
                np.cos(zenith), np.log10(energy))
        except ValueError as e:
            print("Error in spline evaluation. Are the evaluation points ordered?")
            raise e
        
        return np.squeeze(result)
    
    def integrated_spectrum(self, energy, zenith):
        pass
    
    


        
        

        