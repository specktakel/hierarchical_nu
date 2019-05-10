import numpy as np
from scipy import integrate, optimize
from scipy.stats import lognorm
import h5py
from tqdm.autonotebook import tqdm as progress_bar

# Minimum/maximum energy in GeV
AEFF_EMIN = 1.0E3
AEFF_EMAX = 1.0E7


def unit_vector_to_cosz(unit_vector):
    """
    Convert 3D unit vector to cos(zenith) value.
    """
    return np.cos(np.pi - np.arccos(unit_vector[2]))

def p_gt_Eth(E, f_E, Eth):
    """
    Get probability that Edet > Eth | E
    assuming a lognormal distribution 
    characterised by f_E.
    """
    return 1 - lognorm.cdf(Eth, f_E, 0, E)

def get_min_bound(f_E, Eth):
    """
    Find at what E P(Edet > Eth | E) -> 0.
    """    
    fac=10*f_E
    E = optimize.fsolve(p_gt_Eth, Eth/fac, args=(f_E, Eth))
    out = round(E[0])
    # Assume Aeff = 0 below 1e3 GeV
    if (out < 1e3):
        out = 1e3
    return out

class ExposureIntegral(object):
    """
    Everything you need for precomputing exposure integrals.
    
    @author Francesca Capel
    @date April 2019
    """

    def __init__(self, source_redshift, source_position, bg_redshift, effective_area, Emin,
                 filename=None, n_points=50, f_E=None):
        """
        Everything you need for precomputing exposure integrals.
        
        :param source_redshift: list of source redshifts.
        :param source_position: list of unit vectors representing source positions.
        :param bg_redshift: single background redshift.
        :param effective_area: effective_area object from aeff_calculator.
        :param Emin: minimum energy of sample in GeV.
        :param filename: file to save to.
        :param n_points: nummber of points to evaluate integral on.
        :param f_E: energy detection uncertainty.
        """

        self.source_redshift = source_redshift
        self.source_position = source_position
        self.bg_redshift = bg_redshift
        self.Aeff = effective_area
        self.Emin = Emin
        self.filename = filename
        self.f_E = f_E
        
        self.alpha_grid = np.logspace(np.log(1.0), np.log(4.0), n_points, base=np.e)
        
        self._calculate_exposure_integral()
        self._save_to_file()

        
    def _source_integrand(self, E, position, z, alpha):
        """
        Integrand for point sources.
        """

        log10E = np.log10(E) # GeV
        cosz = unit_vector_to_cosz(position)
        Aeff = np.power(10, self.Aeff.eval(log10E, cosz)[0][0]) # m^2

        return  Aeff * ( ((1+z)*E) / self.Emin )**(-alpha)

    
    def _bg_integrand(self, E, cosz, z, alpha):
        """
        Integrand for isotropic background at a certain redshift.
        """

        log10E = np.log10(E) # GeV
        Aeff = np.power(10, self.Aeff.eval(log10E, cosz)[0][0])

        return Aeff * ( ((1+z)*E) / self.Emin )**(-alpha)

    
    def _source_integrand_th(self, E, position, z, alpha):
        """
        Integrand for point sources, including energy threshold
        effects.
        """

        log10E = np.log10(E)
        cosz = unit_vector_to_cosz(position)
        Aeff = np.power(10, self.Aeff.eval(log10E, cosz)[0][0]) # m^2
        p_gt_th = 1-lognorm.cdf(self.Emin, self.f_E, 0, E)
        
        return  p_gt_th * Aeff * ( ((1+z)*E) / self.Emin )**(-alpha)

    
    def _bg_integrand_th(self, E, cosz, z, alpha):
        """
        Integrand for isotropic background at a certain redshift,
        including energy threshold effects.
        """

        log10E = np.log10(E) # GeV
        Aeff = np.power(10, self.Aeff.eval(log10E, cosz)[0][0])
        p_gt_th = 1-lognorm.cdf(self.Emin, self.f_E, 0, E)
        
        return p_gt_th * Aeff * ( ((1+z)*E) / self.Emin )**(-alpha)

    
    def _get_source_integral(self, position, z, alpha):

        if self.f_E:
            integ, err = integrate.quad(self._source_integrand_th, AEFF_EMIN, AEFF_EMAX,
                                        args=(position, z, alpha))
        else:
            integ, err = integrate.quad(self._source_integrand, self.Emin, AEFF_EMAX,
                                        args=(position, z, alpha))
        return integ

    
    def _get_bg_integral(self, z, alpha):

        if self.f_E:
            integ, err = integrate.dblquad(self._bg_integrand_th, -1, 1,
                                           lambda E: AEFF_EMIN, lambda E: AEFF_EMAX,
                                           args=(z, alpha))
        else:
            integ, err = integrate.dblquad(self._bg_integrand, -1, 1,
                                           lambda E: self.Emin, lambda E: AEFF_EMAX,
                                           args=(z, alpha))
        return integ * 0.5 # factor of 2pi/4pi
    
    
    def _calculate_exposure_integral(self):
        """
        Run calculation.
        """

        # Sources
        self.integral_grid = []
        for i in progress_bar(range(len(self.source_position)), desc='Source integrals'):
            position = self.source_position[i]
            z = self.source_redshift[i]
            I_k = []
            for alpha in self.alpha_grid:
                I_k.append(self._get_source_integral(position, z, alpha))
            self.integral_grid.append(I_k)

        # Background
        I_k = []
        for i in progress_bar(range(len(self.alpha_grid)), desc='Background integrals'):
            alpha = self.alpha_grid[i] 
            I_k.append(self._get_bg_integral(self.bg_redshift, alpha))
        self.integral_grid.append(I_k)

        
    def _save_to_file(self):
        """
        Save to file if self.filename is defined.
        """

        if self.filename:

            with h5py.File(self.filename, 'w') as f:
                f.create_dataset('integral_grid', data=self.integral_grid)
                f.create_dataset('alpha_grid', data=self.alpha_grid)
                f.create_dataset('Emin', data=self.Emin)
                

        

