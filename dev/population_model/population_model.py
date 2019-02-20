import numpy as np

import h5py

from scipy import integrate
from scipy import optimize

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm.autonotebook import tqdm as progress_bar

from collections import Counter

"""
Testing out a generative model for neutrinos sources. 
Just build some messy stuff here for now.

@author Francesca Capel
@date Febraury 2019
"""

Om = 0.3
Ol = 0.7
H0 = 70 # km s^-1 Mpc^-1
c = 3E5 # km s^-1
DH = c / H0 # Mpc

Mpc_to_cm = 3.086E+24
cm_to_Mpc = 3.24078E-25
GeV_to_erg = 0.00160218
TeV_to_erg = GeV_to_erg * 1.0E+3
m_to_cm = 1.0E+2
yr_to_s = 3.154E+7

# Cosmology
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

def comoving_volume(z):
    dl = luminosity_distance(z)
    H = hubble_factor(z)
    return (4 * np.pi * dl**2) / ( (1 + z)**2 * H )

def differential_comoving_volume(z):
    dc = comoving_distance(z)
    return (DH * dc**2) / E(z)

def total_energy_flux(n0, L, source_evolution, alpha):
    
    integrand = lambda z : source_evolution(z) * np.power(1 + z, -alpha) / E(z)
    integral, err = integrate.quad(integrand, 0, np.inf)
    
    return DH * n0 * L * integral
    
def power_density(total_energy_flux, source_evolution, alpha):
    """
    :param total_energy_flux: Measured flux in [TeV cm^-2 s^-1]
    :param source_evolution: Function describing source evolution, takes z as param
    :param alpha: Source spectral index
    """
    total_energy_flux = total_energy_flux * np.power(cm_to_Mpc, -2) # TeV Mpc^-2 s^-1
                            
    integrand = lambda z : source_evolution(z) * np.power(1 + z, -alpha) / E(z)
    integral, err = integrate.quad(integrand, 0, np.inf)

    return total_energy_flux / (DH * integral)

# Other
def SFR_evolution(z):
    """
    SFR - using the parameteristion of Cole et al. (2001) and the results of 
    Yüksel et al. (2008).
    """
    a = 3.4
    b = -0.3
    c = -3.3
    eta = -10
    B = 5000
    C = 9
    return ( (1 + z)**(a * eta) + ( (1 + z) / B )**(b * eta) + 
             ((1 + z) / C)**(c * eta) )**(1 / eta)

def n_alpha(alpha):
    """
    Factor coming from the definition of power law normalisation 
    through Lumniosity = \int_Emin^10Emin dE dN/dEdt.
    
    :param alpha: Spectral index at source.
    """
    
    if alpha == 2:
        
        return 1 / np.log(10)
    
    else:
        
        afac = (alpha - 2) / (alpha - 1)
        b = (1 - np.power(10, 2 - alpha))
        return afac / b

class Population(object):


    def __init__(self, local_source_density, source_evolution, luminosity, Emin, alpha, zmin=0, zmax=6):
        """        
        Initialise a population.

        :param local_source_density: Local density in [Mpc^-3]
        :param source_evolution: Function describing source evolution - takes z as argument
        :param luminosity: Luminosity of all sources in [TeV s^-1]
        :param Emin: Minumum energy at z = 0 in [TeV]
        :alpha: Spectral index of sources
        """

        self._local_source_density = local_source_density
        self._source_evolution  = source_evolution
        self._luminosity = luminosity
        self._zmin = zmin
        self._zmax = zmax
        self._Emin = Emin
        self._alpha = alpha
        
        # Create population
        self._sample_redshift()
        self._sample_position()
        
        
    def _differential_comoving_volume(self, z):
        """
        dV/dzdO [Mpc^3 sr^-1]

        :param z: Redshift.
        """
        dc = comoving_distance(z)
        return (DH * dc**2) / E(z)
        
    def _source_distribution(self, z):
        """
        dN/dz = dN/dV * dV/dz 

        :param z: Redshift.
        """

        n0 = self._local_source_density
        dNdV = n0 * self._source_evolution(z)

        dVdz = self._differential_comoving_volume(z)

        return dNdV * dVdz / (1 + z)

    
    def _sample_redshift(self):
        """
        Sample redshifts from a given redshift distribution.
        
        :param N: Number of samples.
        :param zmin: Minimum redshift.
        :param zmax: Maximum redshift.
        """

        # Find N
        Nex, err = integrate.quad(self._source_distribution, self._zmin, self._zmax)
        self.Ns = np.random.poisson(Nex)
    
        # Find maximum and minimum
        tmp = np.linspace(0, self._zmax, 1E3)
        dist = [self._source_distribution(_) for _ in tmp]
        ymax = np.max(dist)
        ymin = np.min(dist)

        # Rejection sampling
        zout = []
        for i in progress_bar(range(self.Ns), desc='Sampling redshifts'):
            accepted = False
            while not accepted:
                y = np.random.uniform(ymin, ymax)
                z = np.random.uniform(self._zmin, self._zmax)
                if y < self._source_distribution(z):
                    zout.append(z)
                    accepted = True

        # Update population 
        self.redshift = zout
        self._get_integral_flux()
        

    def show_redshift_dist(self):
        """
        Plot a histogram of the redshift distribution and
        cumulative distribution.
        """

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True)
        fig.set_size_inches(12, 4)

        ax[0].hist(self.redshift, alpha=0.2);
        ax[0].set_title('Distribution')
        ax[0].set_xlabel('$z$')

        ax[1].hist(self.redshift, alpha=0.5, cumulative=True);
        ax[1].set_title('Cumulative distribution')
        ax[1].set_xlabel('$z$');
        
        return fig

    def _sample_position(self):
        """
        Sample positions on the sky for sources.
        Positions are randomly sampled randomly on the sphere.
        and are described by astropy SkyCoords.
        """

        # NB probably much faster to vectorize sampling

        position = []
        for i in progress_bar(range(self.Ns), desc='Sampling positions'):

            # Sample random positions
            phi = np.random.uniform(0, 2*np.pi, 1)[0]
            costheta = np.random.uniform(-1, 1, 1)[0]
            theta = np.arccos(costheta)

            z = self.redshift[i]

            # Transform
            xcoord = z * np.sin(theta) * np.cos(phi)
            ycoord = z * np.sin(theta) * np.sin(phi)
            zcoord = z * np.cos(theta)
            position.append([xcoord, ycoord, zcoord])

        # Update population    
        self.position = position
        

    def show(self, cmap='viridis', **kwargs):
        """
        Plot the population on a cosmological sphere.
        """

        # Get x, y, z coordinates
        positionT = np.transpose(self.position)
        xcoord = positionT[0]
        ycoord = positionT[1]
        zcoord = positionT[2]
        
        # Plot on sphere
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(xcoord, ycoord, zcoord,
                    c=self.flux,
                    cmap=cmap,
                    norm=mpl.colors.LogNorm(vmin=min(self.flux),
                                            vmax=max(self.flux)),
                    **kwargs)
        ax.set_axis_off()

    def _get_integral_source_emission(self):
        """
        Integral source emission above Emin.
        """

        na = n_alpha(self._alpha)
        Qsrc = (self._luminosity * na) / (self._Emin) # s^-1
        
        return Qsrc
        
    def _get_integral_flux(self):
        """
        Integral flux above Emin from a source at z.
        """

        Qsrc = self._get_integral_source_emission() # s^-1 
        
        Q = []
        for z in self.redshift:
            dl = luminosity_distance(z) * Mpc_to_cm # cm
            Q.append( (Qsrc * np.power(1+z, 1-self._alpha)) / (4*np.pi * np.power(dl, 2)) ) # cm^-2 s^-1

        self.flux = Q

        
    def to_file(self, filename):
        """
        Write the population to file.
        """

        with h5py.File(filename, 'w') as f:

            # Inputs
            input = f.create_group('input')
            input.create_dataset('H0', data = H0)
            input.create_dataset('Om', data = Om)
            input.create_dataset('Ol', data = Ol)
            input.create_dataset('local_source_density', data = self._local_source_density)
            dt = h5py.special_dtype(vlen=str)
            dset = input.create_dataset('source_evolution', (1,), dtype = dt)
            dset[0] =  self._source_evolution.__name__

            # Outputs
            output = f.create_group('output')
            output.create_dataset('Ns', data = self.Ns)
            output.create_dataset('redshift', data = self.redshift)
            output.create_dataset('position', data = self.position)  
            
            
        
