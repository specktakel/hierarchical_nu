import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Callable

from .flux_model import PointSourceFluxModel


class Source(ABC):
    """
    Abstract base class for sources.
    """

    def __init__(self, name: str, *args, **kwargs):
        self._name = name
        self._parameters = []
        self._flux_model = None

    @property
    def parameters(self):
        return self._parameters

    @property
    def name(self):
        return self._name

    @property
    def flux_model(self):
        return self._flux_model

    def flux(self, energy: float, dec: float, ra: float):
        return self._flux_model(energy, dec, ra)
        pass

    @abstractmethod
    def redshift_factor(self, z: float):
        """
        Needs to be implemented in subclass.
        Factor that appears when evaluating the flux in the local frame
        """
        pass


class PointSource(Source):
    """
    Pointsource

    Parameters:
        name: str
        coord: Tuple[float, float]
            Sky coordinate of the source (dec, ra)
        redshift: float
        spectral_shape:
            Spectral shape of the source. Should return units 1/(GeV cm^2 s)
    """

    def __init__(
            self,
            name: str,
            coord: Tuple[float, float],
            redshift: float,
            spectral_shape: Callable[[float], float],
            *args, **kwargs):

        super().__init__(name)
        self._coord = coord
        self._redshift = redshift
        self._flux_model = PointSourceFluxModel(spectral_shape, coord)
        self._parameters = self._flux_model.parameters

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, value):
        self._coord = value

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        self._redshift = value

    def redshift_factor(self, z: float):
        return self._flux_model.redshift_factor(z)


class DiffuseSource(Source):
    """
    DiffuseSource

    Parameters:
        name: str
        redshift: float
        flux_model
            Flux model of the source. Should return units 1/(GeV cm^2 s sr)
    """

    def __init__(self, name: str, redshift: float, flux_model, *args, **kwargs):

        super().__init__(name)
        self._redshift = redshift
        self._flux_model = flux_model
        self._parameters = flux_model.parameters

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        self._redshift = value

    def flux(self, energy: float, dec: float, ra: float):
        return self._flux_model(energy, dec, ra)

    def total_flux(self, energy: float):
        return self._flux_model.total_flux(energy)

    def integral_flux(self, dec: float, ra: float, lower: float, upper: float):
        return self._flux_model.integral_flux(dec, ra, lower, upper)

    def integral_total_flux(self, lower: float, upper: float):
        return self._flux_model.integral_toal_flux(lower, upper)

    def redshift_factor(self, z: float):
        return self._flux_model.redshift_factor(z)


class SourceList(ABC):
    """
    Abstract base class for container of a list of sources.
    """

    def __init__(self):

        super().__init__()
        self.N = 0
        self._sources = []

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, value):
        if not isinstance(value, list):
            raise ValueError(str(value) + " is not a list")

        if not isinstance(value[0], Source):
            raise ValueError(str(value) + " is not a recognised source list")

        else:
            self._sources = value
            self.N = len(self._sources)

    def add(self, value):
        if not isinstance(value, Source):
            raise ValueError(str(value) + " is not a recognised source")
        else:
            self._sources.append(value)
            self.N += 1

    def remove(self, index):
        self._sources.pop(index)
        self.N -= 1

    def __iter__(self):
        for source in self._sources:
            yield source


class TestSourceList(SourceList):
    def __init__(self, filename, spectral_shape=None):
        """
        Simple source list from test file used in
        development. Can be adapted to different
        catalogs.

        :param filename: File to read from.
        :param spectral_shape: Option to specify spectral shape for all sources
        """

        super().__init__()
        self._filename = filename
        self._spectral_shape = spectral_shape
        self._read_from_file()

    def _read_from_file(self):

        import h5py

        with h5py.File(self._filename, "r") as f:
            redshift = f["output/redshift"][()]
            position = f["output/position"][()]

        unit_vector = position / np.linalg.norm(position, axis=1)[:, np.newaxis]
        ra, dec = uv_to_icrs(unit_vector)

        for i, (r, d, z) in enumerate(zip(ra, dec, redshift)):
            source = PointSource(str(i), (d, r), z, self._spectral_shape)
            self.add(source)

    def select_below_redshift(self, zth):
        self._zth = zth
        self.sources = [s for s in self.sources if s.redshift <= zth]


def uv_to_icrs(unit_vector):
    """
    convert unit vector to ICRS coords (ra, dec)
    """

    if len(np.shape(unit_vector)) > 1:

        theta = np.arccos(unit_vector.T[2])

        phi = np.arctan(unit_vector.T[1] / unit_vector.T[0])

    else:

        theta = np.arccos(unit_vector[2])

        phi = np.arccos(unit_vector[1] / unit_vector[0])
    ra, dec = spherical_to_icrs(theta, phi)

    return ra, dec


def spherical_to_icrs(theta, phi):
    """
    convert spherical coordinates to ICRS
    ra, dec.
    """

    ra = phi

    dec = np.pi / 2 - theta

    return ra, dec


def lists_to_tuple(list1, list2):

    return [(list1[i], list2[i]) for i in range(0, len(list1))]
