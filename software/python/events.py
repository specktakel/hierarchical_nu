import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord


class Events:
    """
    Events class for the storage of event observables
    """

    @u.quantity_input
    def __init__(self, energies: u.GeV, coords: SkyCoord, event_types):
        """
        Events class for the storage of event observables
        """

        self._recognised_types = ["track", "cascade"]

        self.N = len(energies)

        self._energies = energies

        coords.representation_type = "spherical"
        self._coords = coords
        coords.representation_type = "cartesian"
        self._unit_vectors = np.array(
            [coords.x.value, coords.y.value, coords.z.value]
        ).T

        if all([et in self._recognised_types for et in event_types]):
            self._event_types = event_types
        else:
            raise ValueError("Event types not recognised")

    def remove(self, i):

        self._energies.pop(i)
        self._coords.pop(i)
        self._unit_vectors.pop(i)
        self._event_types.pop(i)
        self.N -= 1

    @property
    def energies(self):

        return self._energies

    @property
    def coords(self):

        return self._coords

    @property
    def event_types(self):

        return self._event_types

    @classmethod
    def from_file(cls, filename):

        pass
