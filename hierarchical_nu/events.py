import numpy as np
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
    def __init__(
        self,
        energies: u.GeV,
        coords: SkyCoord,
        types,
        ang_errs: u.deg = None,
    ):
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

        if all([t in self._recognised_types for t in types]):
            self._types = types
        else:
            raise ValueError("Event types not recognised")

        self._ang_errs = ang_errs

    def remove(self, i):

        self._energies.pop(i)
        self._coords.pop(i)
        self._unit_vectors.pop(i)
        self._types.pop(i)
        self._ang_errs.pop(i)
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
    def types(self):

        return self._types

    @property
    def ang_errs(self):

        return self._ang_errs

    @property
    def kappas(self):

        return 1.38 / self._ang_errs.to(u.rad).value ** 2

    @classmethod
    def from_file(cls, filename):

        with h5py.File(filename, "r") as f:

            events_folder = f["events"]

            energies = events_folder["energies"][()] * u.GeV
            uvs = events_folder["unit_vectors"][()]
            types = events_folder["event_types"][()]
            ang_errs = events_folder["ang_errs"][()] * u.deg

        coords = SkyCoord(
            uvs.T[0], uvs.T[1], uvs.T[2], representation_type="cartesian", frame="icrs"
        )

        return cls(energies, coords, types, ang_errs)

    def to_file(self, filename, append=False):

        self._file_keys = ["energies", "unit_vectors", "event_types", "ang_errs"]
        self._file_values = [
            self.energies.to(u.GeV).value,
            self.unit_vectors,
            self.types,
            self.ang_errs.to(u.deg).value,
        ]

        if append:
            with h5py.File(filename, "r+") as f:

                event_folder = f.create_group("events")

                for key, value in zip(self._file_keys, self._file_values):

                    event_folder.create_dataset(key, data=value)

        else:
            with h5py.File(filename, "w") as f:

                event_folder = f.create_group("events")

                for key, value in zip(self._file_keys, self._file_values):

                    event_folder.create_dataset(key, data=value)
