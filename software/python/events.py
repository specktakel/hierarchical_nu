import numpy as np
import os
import h5py

from astropy import units as u
from astropy.coordinates import SkyCoord

TRACKS = 0
CASCADES = 1


class Events:
    """
    Events class for the storage of event observables
    """

    @u.quantity_input
    def __init__(self, energies: u.GeV, coords: SkyCoord, event_types):
        """
        Events class for the storage of event observables
        """

        self._recognised_types = [TRACKS, CASCADES]

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
    def unit_vectors(self):

        return self._unit_vectors

    @property
    def event_types(self):

        return self._event_types

    @classmethod
    def from_file(cls, filename):

        pass

    def to_file(self, filename, append=False):

        keys = ["energies", "unit_vectors", "event_types"]
        values = [self.energies.to(u.GeV).value, self.unit_vectors, self.event_types]

        if append:
            with h5py.File(filename, "r+") as f:

                event_folder = f.create_group("events")

                for key, value in zip(keys, values):

                    event_folder.create_dataset(key, data=value)

        else:
            with h5py.File(filename, "w") as f:

                event_folder = f.create_group("events")

                for key, value in zip(keys, values):

                    event_folder.create_dataset(key, data=value)
