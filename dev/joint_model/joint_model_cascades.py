"""
Utils for the `joint_model_cascades` notebook.

@author Francesca Capel
@date February 2019
"""

import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import ndimage
import pandas as pd

# Cosmology

Om = 0.3
Ol = 0.7
H0 = 70 # km s^-1 Mpc^-1
c = 3E5 # km s^-1
DH = c / H0 # Mpc

Mpc_to_m = 3.086E22

def E(z):
    Omp = Om * (1 + z)**3
    return np.sqrt(Omp + Ol)

def hubble_factor(z):
    Omp = Om * (1 + z)**3
    return H0 * E(z)

def comoving_distance(z):    

    scale = lambda z : 1 / E(z) 
    result, err = integrate.quad(scale, 0, z)

    return DH * result 

def luminosity_distance(z):
    return (1 + z) * comoving_distance(z)


class IceCubeAeff():

    
    def __init__(self, filename, selected_E = None):
        """
        Define IceCube effective area from public data files for easy use in 
        some test Stan models.
        """

        self._filename = filename
        self._read_from_file()
        self._interpolation()

        self.selected_E = selected_E
        self._get_m()
        self._get_M()
        
        
        
    def _read_from_file(self):
        """
        Read in data from file. Energies in [GeV] and Aeff in [m^2].
        """
        
        filelayout = ['Emin', 'Emax', 'cos(z)min', 'cos(z)max', 'Aeff']
        output = pd.read_csv(self._filename, comment = '#',
                             delim_whitespace = True,
                             names = filelayout)

        output_dict = output.to_dict()
        Emin = list(output_dict['Emin'].values())
        Emax = list(output_dict['Emax'].values())
        coszmin = list(output_dict['cos(z)min'].values())
        coszmax = list(output_dict['cos(z)max'].values())
        Aeff = list(output_dict['Aeff'].values())

        # find bin centres
        Emin = np.sort(list(set(Emin)))
        Emax = np.sort(list(set(Emax)))
        coszmin = np.sort(list(set(coszmin)))
        coszmax = np.sort(list(set(coszmax)))

        self._energy = (Emin + Emax)/2 # GeV
        self._cosz = (coszmin + coszmax)/2
        self._Aeff = np.reshape(Aeff, (70, 200))
        

    def _interpolation(self):
        """
        Perform interpolation and smoothing. 
        """
        
        sigma = [0.001, 5]
        Aeff_smooth = ndimage.filters.gaussian_filter(self._Aeff, sigma, mode='constant')
        self._Aeff_interp = interpolate.RectBivariateSpline(self._energy, self._cosz, Aeff_smooth, s=0.0)
        self._Aeff_max = Aeff_smooth.max()

        
    def select_energy(self, selected_E):
        """
        Select a certain energy to work with in [GeV] 
        """

        self.selected_E = selected_E

    def _get_m(self):
        """
        Get function to return projection factor from interpolated Aeff.
        """

        # Only defined if certain energy selected
        if not self.selected_E:

            return 0

        def m(zenith_angle):
            """projection factor"""
            cosz = np.cos(zenith_angle)
            return self._Aeff_interp(self.selected_E, cosz)[0][0] / self._Aeff_max
        
        self.m = m

            
    def _get_M(self):
        """
        Calculate the integral of the projection factor over all solid angles. 
        """

        # Only defined if certain energy selected
        if not self.selected_E:

            return 0
        
        def m_integrand(theta):
            """
            Integrand for \int d(omega) mu(omega).
            
            Expressed as an integral in spherical coordinates for
            theta[0, pi]. For use with scipy.integrate.quad.
            
            m is an interpolation function representing the projection factor
            from IceCube data.

            NB: in the spherical coordinate system, southern zenith = np.pi - theta
            """

            zenith_angle = np.pi - theta
            m = self._Aeff_interp(self.selected_E, np.cos(zenith_angle))[0] / self._Aeff_max 
            return 2 * np.pi * m * np.sin(theta)

        self.M, err = integrate.quad(m_integrand, 0, np.pi)
              

    
