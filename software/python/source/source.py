from abc import ABC, abstractmethod
from typing import Callable

from astropy import units as u
import numpy as np

from .flux_model import (
    PointSourceFluxModel,
    PowerLawSpectrum,
    IsotropicDiffuseBG,
)
from .atmospheric_flux import AtmosphericNuMuFlux
from .cosmology import luminosity_distance
from .parameter import Parameter, ParScale


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

    @u.quantity_input
    def flux(
        self, energy: u.GeV, dec: u.rad, ra: u.rad
    ) -> 1 / (u.GeV * u.m ** 2 * u.s * u.sr):
        return self._flux_model(energy, dec, ra)

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
            Sky coordinate of the source (dec, ra) [deg]

        redshift: float
        spectral_shape:
            Spectral shape of the source. Should return units 1/(GeV cm^2 s)
    """

    @u.quantity_input
    def __init__(
        self,
        name: str,
        dec: u.rad,
        ra: u.rad,
        redshift: float,
        spectral_shape: Callable[[float], float],
        *args,
        **kwargs
    ):

        super().__init__(name)
        self._dec = dec
        self._ra = ra
        self._redshift = redshift
        self._flux_model = PointSourceFluxModel(spectral_shape, dec, ra)
        self._parameters = self._flux_model.parameters

        # calculate luminosity
        total_flux_int = self._flux_model.total_flux_density
        self._luminosity = total_flux_int * (
            4 * np.pi * luminosity_distance(redshift) ** 2
        )

    @classmethod
    @u.quantity_input
    def make_powerlaw_source(
        cls,
        name: str,
        dec: u.rad,
        ra: u.rad,
        luminosity: Parameter,
        index: Parameter,
        redshift: float,
        lower: Parameter,
        upper: Parameter,
    ):
        """
        Factory class for creating sources with powerlaw spectrum and given luminosity.

        Parameters:
            name: str
                Source name
            dec: u.rad,
                Declination of the source
            ra: u.rad,
                Right Ascension of the source
            luminosity: Parameter,
                luminosity
            index: Parameter
                Spectral index
            redshift: float
            lower: Parameter
                Lower energy bound
            upper: Parameter
                Upper energy bound
        """

        total_flux = luminosity.value / (
            4 * np.pi * luminosity_distance(redshift) ** 2
        )  # here flux is W / m^2

        # Each source has an independent normalization, thus use the source name as identifier
        norm = Parameter(
            1 / (u.GeV * u.s * u.m ** 2),
            "{}_norm".format(name),
            fixed=False,
            par_range=(0, np.inf),
            scale=ParScale.log,
        )

        shape = PowerLawSpectrum(norm, 1e5 * u.GeV, index, lower.value, upper.value)
        total_power = shape.total_flux_density

        norm.value *= total_flux / total_power
        norm.value = norm.value.to(1 / (u.GeV * u.m ** 2 * u.s))
        norm.fixed = True
        return cls(name, dec, ra, redshift, shape, luminosity)

    @classmethod
    def make_powerlaw_sources_from_file(
        cls,
        filename: str,
        luminosity: Parameter,
        index: Parameter,
        lower_energy: Parameter,
        upper_energy: Parameter,
    ):
        """
        Factory for power law sources defined in HDF5 files
        with shared luminosity and spectral index.
        """

        import h5py

        with h5py.File(filename, "r") as f:
            redshift = f["output/redshift"][()]
            position = f["output/position"][()]

        unit_vector = position / np.linalg.norm(position, axis=1)[:, np.newaxis]
        ra, dec = uv_to_icrs(unit_vector)

        source_list = []
        for i, (r, d, z) in enumerate(zip(ra, dec, redshift)):
            source = PointSource.make_powerlaw_source(
                str(i),
                d,
                r,
                luminosity,
                index,
                z,
                lower_energy,
                upper_energy,
            )
            source_list.append(source)

        return source_list

    @property
    def dec(self):
        return self._dec

    @dec.setter
    def dec(self, value):
        self._dec = value

    @property
    def ra(self):
        return self._ra

    @ra.setter
    def ra(self, value):
        self._ra = value

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        self._redshift = value

    def redshift_factor(self, z: float):
        return self._flux_model.redshift_factor(z)

    @property
    @u.quantity_input
    def luminosity(self) -> u.erg / u.s:
        return self._luminosity

    @luminosity.setter
    @u.quantity_input
    def luminosity(self, value: u.erg / u.s):
        self._luminosity = value


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

    def redshift_factor(self, z: float):
        return self._flux_model.redshift_factor(z)


class Sources:
    """
    Container for sources with a set of factory methods
    for and easy source setup interface.
    """

    def __init__(self):

        # Number of source components
        self.N = 0

        # Initialise the source list
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

    def add(self, source):
        """
        Add any instance of a class inheriting from Source.
        """

        if isinstance(source, Source):
            self._sources.append(source)
            self.N += 1

        elif isinstance(source, list) and all(isinstance(s, Source) for s in source):
            for s in source:
                self.add(s)

        else:
            raise ValueError(
                str(source) + " is not a recognised source or list of sources"
            )

    @u.quantity_input
    def add_diffuse_component(self, flux_norm: Parameter, norm_energy: u.GeV):
        """
        Add diffuse component based on point
        source component definition.

        :param flux_norm: The flux normalization for this component
        :param norm_energy: The energy at which the flux norm is defined
        """

        # grab parameters from point sources
        index = Parameter.get_parameter("index")
        Emin = Parameter.get_parameter("Emin")
        Emax = Parameter.get_parameter("Emax")

        # Check maximum redshift of supplied sources
        zth = self._get_max_ps_redshift()

        # define flux model
        spectral_type = self._get_ps_spectral_type()
        spectral_shape = spectral_type(
            flux_norm, norm_energy, index, Emin.value, Emax.value
        )
        flux_model = IsotropicDiffuseBG(spectral_shape)

        # define component
        diffuse_component = DiffuseSource("diffuse_bg", zth, flux_model=flux_model)

        self.add(diffuse_component)

    def _get_max_ps_redshift(self):
        """
        Check maximum redshift of exisiting point sources.
        """

        z = []
        for source in self.sources:

            if isinstance(source, PointSource):
                z.append(source.redshift)

        return max(z)

    def _get_ps_spectral_type(self):
        """
        Check the spectral type of point sources in the list.
        """

        types = []
        for source in self.sources:

            if isinstance(source, PointSource):
                types.append(type(source.flux_model.spectral_shape))

        # Check all the same
        if not types[1:] == types[:-1]:

            raise ValueError("Not all point sources have the same spectral_shape")

        return types[0]

    def add_atmospheric_component(self):
        """
        Add an atmospheric flux component based on the IceCube observations.
        """

        Emin = Parameter.get_parameter("Emin")
        Emax = Parameter.get_parameter("Emax")

        flux_model = AtmosphericNuMuFlux(Emin.value, Emax.value)

        atmospheric_component = DiffuseSource("atmo_bg", 0, flux_model=flux_model)

        self.add(atmospheric_component)

    def select_below_redshift(self, zth):
        """
        remove sources with redshift above a certain threshold.
        """

        self.sources = [s for s in self.sources if s.redshift <= zth]

    def remove(self, i):
        self._sources.pop(i)
        self.N -= 1

    def total_flux_int(self):
        tot = 0
        for source in self:
            tot += source.flux_model.total_flux_int
        return tot

    def associated_fraction(self):

        point_source_ints = sum(
            [
                s.flux_model.total_flux_int
                for s in self.sources
                if isinstance(s, PointSource)
            ]
        )

        return point_source_ints / self.total_flux_int()

    def __iter__(self):
        for source in self._sources:
            yield source

    def __getitem__(self, key):
        return self._sources[key]


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

    def remove(self, i):
        self._sources.pop(i)
        self.N -= 1

    def total_flux_int(self):
        tot = 0
        for source in self:
            tot += source.flux_model.total_flux_int
        return tot

    def __iter__(self):
        for source in self._sources:
            yield source

    def __getitem__(self, key):
        return self._sources[key]


@u.quantity_input
class TestSourceList(SourceList):
    def __init__(
        self,
        filename,
        luminosity: Parameter,
        index: Parameter,
        lower_energy: Parameter,
        upper_energy: Parameter,
    ):
        """
        Simple source list from test file used in
        development. Can be adapted to different
        catalogs.

        :param filename: File to read from.
        :param luminosity: Luminosity shared by all sources
        :param index: Spectral index of power law
        :param lower_energy: Lower energy for L definition
        :param upper_energy: Upper energy for L definition
        """

        super().__init__()
        self._filename = filename
        self._luminosity = luminosity
        self._index = index
        self._lower_energy = lower_energy
        self._upper_energy = upper_energy
        self._read_from_file()

    def _read_from_file(self):

        import h5py

        with h5py.File(self._filename, "r") as f:
            redshift = f["output/redshift"][()]
            position = f["output/position"][()]

        unit_vector = position / np.linalg.norm(position, axis=1)[:, np.newaxis]
        ra, dec = uv_to_icrs(unit_vector)

        for i, (r, d, z) in enumerate(zip(ra, dec, redshift)):
            source = PointSource.make_powerlaw_source(
                str(i),
                d,
                r,
                self._luminosity,
                self._index,
                z,
                self._lower_energy,
                self._upper_energy,
            )
            self.add(source)

    def select_below_redshift(self, zth):
        self._zth = zth
        self.sources = [s for s in self.sources if s.redshift <= zth]


from astropy.coordinates import SkyCoord
from astropy import units as u


class Direction:
    """
    Input the unit vector vMF samples and
    store x, y, and z and galactic coordinates
    of direction in Mpc.
    """

    def __init__(self, unit_vector_3d):
        """
        Input the unit vector samples and
        store x, y, and z and galactic coordinates
        of direction in Mpc.

        :param unit_vector_3d: a 3-dimensional unit vector.
        """

        self.unit_vector = unit_vector_3d
        transposed_uv = np.transpose(self.unit_vector)
        self.x = transposed_uv[0]
        self.y = transposed_uv[1]
        self.z = transposed_uv[2]
        self.d = SkyCoord(
            self.x,
            self.y,
            self.z,
            unit="mpc",
            representation_type="cartesian",
            frame="icrs",
        )
        self.d.representation_type = "spherical"
        self.lons = self.d.galactic.l.wrap_at(360 * u.deg).deg
        self.lats = self.d.galactic.b.wrap_at(180 * u.deg).deg


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


def icrs_to_uv(dec, ra):
    theta = np.pi / 2 - dec
    x = np.sin(theta) * np.cos(ra)
    y = np.sin(theta) * np.sin(ra)
    z = np.cos(theta)
    return [x, y, z]


def spherical_to_icrs(theta, phi):
    """
    convert spherical coordinates to ICRS
    ra, dec.
    """

    ra = phi
    dec = np.pi / 2 - theta

    return ra * u.rad, dec * u.rad


def lists_to_tuple(list1, list2):

    return [(list1[i], list2[i]) for i in range(0, len(list1))]
