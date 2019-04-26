import numpy as np
from scipy import integrate
import h5py
from tqdm.autonotebook import tqdm as progress_bar

# Maximum energy in GeV
Emax = 1.0E7

# Convert 3D unit vector to cos(zenith) value
def unit_vector_to_cosz(unit_vector):
    return np.cos(np.pi - np.arccos(unit_vector[2]))

class ExposureIntegral(object):
    """
    Everything you need from precomputing expousre integrals.
    
    @author Francesca Capel
    @date April 2019
    """

    def __init__(self, source_redshift, source_position, bg_redshift, effective_area, Emin, filename=None, n_points=50):
        """
        Everything you need from precomputing expousre integrals.
        
        :param source_redshift: list of source redshifts.
        :param source_position: list of unit vectors representing source positions.
        :param bg_redshift: single background redshift.
        :param effective_area: effective_area object from aeff_calculator.
        :param Emin: minimum energy of sample in GeV.
        :param filename: file to save to.
        :param n_points: nummber of points to evaluate integral on.
        """

        self.source_redshift = source_redshift
        self.source_position = source_position
        self.bg_redshift = bg_redshift
        self.Aeff = effective_area
        self.Emin = Emin
        self.filename = filename

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

    
    def _get_source_integral(self, position, z, alpha):

        integ, err = integrate.quad(self._source_integrand, self.Emin, Emax,
                                    args=(position, z, alpha))
        return integ

    
    def _get_bg_integral(self, z, alpha):

        integ, err = integrate.dblquad(self._bg_integrand, -1, 1,
                                       lambda E: self.Emin, lambda E: Emax,
                                       args=(z, alpha))
        return integ
    
    
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
                

        

