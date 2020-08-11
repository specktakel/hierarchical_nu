import numpy as np
from abc import ABC, abstractmethod

from .power_law import BoundedPowerLaw

"""
Module for simple flux models used in 
neutrino detection calculations
"""

class FluxModel(ABC):
    """
    Abstract base class for flux models.
    """        
        
    @abstractmethod
    def spectrum(self):

        pass

    @abstractmethod
    def integrated_spectrum(self):

        pass


    
class PowerLawFlux(FluxModel):
    """
    Power law flux models.
    """

    def __init__(self, normalisation, normalisation_energy, index,
                 lower_energy=1e2, upper_energy=np.inf):
        """
        Power law flux models. 

        :param normalisation: Flux normalisation [GeV^-1 cm^-2 s^-1 sr^-1] or [GeV^-1 cm^-2 s^-1] for point sources.
        :param normalisation energy: Energy at which flux is normalised [GeV].
        :param index: Spectral index of the power law.
        :param lower_energy: Lower energy bound [GeV].
        :param upper_energy: Upper enegry bound [GeV], unbounded by default.
        """

        super().__init__()
        
        self._normalisation = normalisation

        self._normalisation_energy = normalisation_energy

        self._index = index

        self._lower_energy = lower_energy
        self._upper_energy = upper_energy


    def spectrum(self, energy):
        """
        dN/dEdAdt or dN/dEdAdtdO depending on flux_type.
        """

        if (energy < self._lower_energy) or (energy > self._upper_energy):

            return np.nan

        else:
        
            return self._normalisation * np.power(energy / self._normalisation_energy, -self._index)

    
    def integrated_spectrum(self, lower_energy_bound, upper_energy_bound):
        """
        \int spectrum dE over finite energy bounds.
        
        :param lower_energy_bound: [GeV]
        :param upper_energy_bound: [GeV]
        """

        """
        if lower_energy_bound < self._lower_energy and upper_energy_bound < self._lower_energy:
            return 0
        elif lower_energy_bound < self._lower_energy and upper_energy_bound > self._lower_energy:
            lower_energy_bound = self._lower_energy

        if upper_energy_bound > self._upper_energy and lower_energy_bound > self._upper_energy:
            return 0
        elif upper_energy_bound > self._upper_energy and lower_energy_bound < self._upper_energy:
            upper_energy_bound = self._upper_energy
        """

        norm = self._normalisation / ( np.power(self._normalisation_energy, -self._index) * (1 - self._index) )

        return norm * ( np.power(upper_energy_bound, 1-self._index) - np.power(lower_energy_bound, 1-self._index) )


    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.
        
        :param min_energy: Minimum energy to sample from [GeV].
        :param N: Number of samples.
        """
        
        self.power_law = BoundedPowerLaw(self._index, self._lower_energy, self._upper_energy)        
        
        return self.power_law.samples(N)

    
    def _rejection_sample(self, min_energy):
        """
        Sample energies from the power law.
        Uses rejection sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        """

        dist_upper_lim = self.spectrum(min_energy)

        accepted = False

        while not accepted:

            energy = np.random.uniform(min_energy, 1e3*min_energy)
            dist = np.random.uniform(0, dist_upper_lim)

            if dist < self.spectrum(energy):

                accepted = True

        return energy
