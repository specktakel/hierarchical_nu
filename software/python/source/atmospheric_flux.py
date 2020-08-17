from MCEq.core import config, MCEqRun
import crflux.models as crf
# matplotlib used plotting. Not required to run the code.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import scipy
import pickle

from astromodels.functions.function import Function2D, FunctionMeta

from .source import DiffuseSource
from ..cache import Cache
from ..backend import (
    UserDefinedFunction,
    StanArray,
    TruncatedParameterization,
    LogParameterization,
    StringExpression,
    FunctionCall,
    ReturnStatement,
    ForwardVariableDef
)

Cache.set_cache_dir(".cache")


class _AtmosphericNuMuFluxStan(UserDefinedFunction):
    def __init__(
            self,
            splined_flux,
            log_energy_grid,
            theta_points=50):
        
        UserDefinedFunction.__init__(
            self,
            "AtmopshericNumuFlux",
            ["true_energy", "true_dir"],
            ["real", "vector"],
            "real")
            

        self.theta_points = theta_points
        
        cos_theta_grid = np.linspace(0, 1, self.theta_points)
        cos_theta_centers = 0.5*(cos_theta_grid[1:] + cos_theta_grid[:-1])
       
        
        spl_evals = np.empty((self.theta_points, len(log_energy_grid)))
        
        for i, cos_theta in enumerate(cos_theta_grid):
            spl_evals[i] = splined_flux(cos_theta, log_energy_grid).squeeze()
        
        with self:       
            spl_evals_stan = StanArray(
                    "AtmosphericFluxPolyCoeffs",
                    "real",
                    spl_evals)
            
            cos_theta_grid_stan = StanArray("cos_theta_grid", "real", cos_theta_grid)
            log_energy_grid_stan = StanArray("log_energy_grid", "real", log_energy_grid)
            
            truncated_e = TruncatedParameterization(
                "true_energy", 10**log_energy_grid[0], 10**log_energy_grid[-1])
            log_trunc_e = LogParameterization(truncated_e)

            # abs() since the flux is symmetric around the horizon
            cos_dir = StringExpression(["abs(cos(pi() - acos(true_dir[3])))"])
            cos_theta_bin_index = FunctionCall([cos_dir, cos_theta_grid_stan], "binary_search", 2)
            
            vect_spl_vals_low = FunctionCall([spl_evals_stan[cos_theta_bin_index]], "to_vector")
            vect_spl_vals_high = FunctionCall([spl_evals_stan[cos_theta_bin_index+1]], "to_vector")
            vect_log_e_grid = FunctionCall([log_energy_grid_stan], "to_vector")
            
            interpolated_energy_low = FunctionCall([vect_log_e_grid, vect_spl_vals_low, log_trunc_e], "interpolate", 3)
            interpolated_energy_high = FunctionCall([vect_log_e_grid, vect_spl_vals_high, log_trunc_e], "interpolate", 3)
            
            vector_log_trunc_e = ForwardVariableDef(
                "vector_interp_energies",
                "vector[2]")
            vector_coz_grid_points = ForwardVariableDef(
                "vector_coz_grid_points",
                "vector[2]")
            vector_log_trunc_e[1] << interpolated_energy_low
            vector_log_trunc_e[2] << interpolated_energy_high
            vector_coz_grid_points[1] << cos_theta_grid_stan[cos_theta_bin_index]
            vector_coz_grid_points[2] << cos_theta_grid_stan[cos_theta_bin_index+1]
            
            interpolate_cosz = FunctionCall([vector_coz_grid_points, vector_log_trunc_e, cos_dir], "interpolate", 3)
            _ = ReturnStatement([interpolate_cosz])
        
class AtmosphericNuMuFlux():
    
    CACHE_FNAME = "mceq_flux.pickle"
    EMAX = 1E9
    EMIN = 1
    THETA_BINS = 100
    
    def __init__(self):
        self._setup()
       
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

    def spectrum(self, energy, zenith):
        energy = np.atleast_1d(energy)
        if np.any((energy > self.EMAX) | (energy < self.EMIN)):
            raise ValueError("Energy needs to be in {} < E {}".format(self.EMIN, self.EMAX))
        
        zenith = np.atleast_1d(zenith)
        try:
            result = self._flux_spline(
                np.cos(zenith), np.log10(energy))
        except ValueError as e:
            print("Error in spline evaluation. Are the evaluation points ordered?")
            raise e
        
        return np.squeeze(result)
    
    def integrated_spectrum(self, energy, zenith):
        pass
                 
    def make_stan_function(self, energy_points=100, theta_points=50):
        log_energy_grid = np.linspace(np.log10(self.EMIN), np.log10(self.EMAX), energy_points)
        return _AtmosphericNuMuFluxStan(
            self._flux_spline,
            log_energy_grid,
            theta_points)

    
class AtmosphericNuMuSpectrum(Function2D, metaclass=FunctionMeta):
    r"""
        description :
        
            Atmospheric numu flux
        
        latex : $ $
                
        parameters : {}
        
        """
    
    def _setup(self):
        self.atmo_flux = AtmosphericNuMuFlux()
        
    def _set_units(self):
        pass
    
    
    def evaluate(self, x, y, *args):        
        return self.atmo_flux(x, np.pi() - y)

    def get_boundaries(self):
        return (self.atmo_flux.EMIN, self.atmo_flux.EMAX), (-np.pi/2, np.pi/2)
