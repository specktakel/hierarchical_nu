from abc import ABC, abstractmethod
import h5py
from astropy import units as u
import numpy as np
from collections.abc import Callable

from .flux_model import (
    PointSourceFluxModel,
    PowerLawSpectrum,
    TwiceBrokenPowerLaw,
    IsotropicDiffuseBG,
    integral_power_law,
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
    ) -> 1 / (u.GeV * u.m**2 * u.s * u.sr):
        return self._flux_model(energy, dec, ra)


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
        Luminosity and all energies given as arguments/parameters live in the source frame
        and are converted to detector frame internally.

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
        All parameters are taken to be defined in the source frame.
        """

        total_flux = luminosity.value / (
            4 * np.pi * luminosity_distance(redshift) ** 2
        )  # here flux is W / m^2, lives in the detector frame

        # Each source has an independent normalization, thus use the source name as identifier
        # Normalisation to dN/(dEdtdA)
        norm = Parameter(
            # is defined at the detector!
            1 / (u.GeV * u.s * u.m**2),
            "{}_norm".format(name),
            fixed=False,
            par_range=(0, np.inf),
            scale=ParScale.log,
        )

        # Use Enorm if set, otherwise fix to 1e5 GeV
        try:
            Enorm_value = Parameter.get_parameter("Enorm").value
        except ValueError:
            Enorm_value = 1e5 * u.GeV

        shape = PowerLawSpectrum(
            norm,
            Enorm_value,
            index,
            lower.value / (1 + redshift),
            upper.value / (1 + redshift),
        )
        total_power = shape.total_flux_density
        norm.value *= total_flux / total_power
        norm.value = norm.value.to(1 / (u.GeV * u.m**2 * u.s))
        norm.fixed = True
        return cls(name, dec, ra, redshift, shape, luminosity)

    @classmethod
    @u.quantity_input
    def make_twicebroken_powerlaw_source(
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
        Luminosity and all energies given as arguments/parameters live in the source frame
        and are converted to detector frame internally.

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
                Lower energy bound in source frame
            upper: Parameter
                Upper energy bound in source frame
        Takes additionally Emin and Emax (in the detector frame) to limit the spectral model.
        `index` is the valid spectral index between `lower` and `upper` energ
        """
        # raise NotImplementedError
        total_flux = luminosity.value / (
            4 * np.pi * luminosity_distance(redshift) ** 2
        )  # here flux is W / m^2, lives in the detector frame

        # Each source has an independent normalization, thus use the source name as identifier
        # Normalisation to dN/(dEdtdA)
        norm = Parameter(
            # is defined at the detector!
            1 / (u.GeV * u.s * u.m**2),
            "{}_norm".format(name),
            fixed=False,
            par_range=(0, np.inf),
            scale=ParScale.log,
        )

        shape = TwiceBrokenPowerLaw(
            norm,
            1e5 * u.GeV,
            index,
            lower.value / (1 + redshift),
            upper.value / (1 + redshift),
        )
        total_power = shape.total_flux_density
        norm.value *= total_flux / total_power
        norm.value = norm.value.to(1 / (u.GeV * u.m**2 * u.s))
        norm.fixed = True
        return cls(name, dec, ra, redshift, shape, luminosity)

    @classmethod
    def make_powerlaw_sources_from_file(
        cls,
        file_name: str,
        lower_energy: Parameter,
        upper_energy: Parameter,
        include_undetected: bool = False,
    ):
        """
        Factory for power law sources defined in
        HDF5 files ( update: output from popsynth).

        :param lower_energy: Lower energy bound in definition of the luminosity.
        :param upper_energy: Upper energy bound in definition of the luminosity.
        :param include_undetected: Include sources that are not detected in population.
        """

        # Sensible bounds on luminosity
        lumi_range = (0, 1e60) * (u.erg / u.s)

        # Load values
        with h5py.File(file_name, "r") as f:
            luminosities = f["luminosities"][()] * (u.erg / u.s)

            spectral_indices = f["auxiliary_quantities/spectral_index/obs_values"][()]

            redshifts = f["distances"][()]

            ras = f["phi"][()] * u.rad

            decs = (f["theta"][()] - np.pi / 2) * u.rad

            selection = f["selection"][()]

        # Apply selection
        if not include_undetected:
            luminosities = luminosities[selection]

            spectral_indices = spectral_indices[selection]

            redshifts = redshifts[selection]

            ras = ras[selection]

            decs = decs[selection]

        # Make cut on declination, current implementations are only defined in the Northern Sky
        # dec > -5deg

        mask = decs >= np.deg2rad(-5) * u.rad

        luminosities = luminosities[mask]
        spectral_indices = spectral_indices[mask]
        redshifts = redshifts[mask]
        ras = ras[mask]
        decs = decs[mask]

        # Make list of point sources
        source_list = []

        for i, (L, index, ra, dec, z) in enumerate(
            zip(
                luminosities,
                spectral_indices,
                ras,
                decs,
                redshifts,
            )
        ):
            # Check for shared luminosity parameter
            try:
                luminosity = Parameter.get_parameter("luminosity")

            # Else, create individual ps_%i_luminosity parameters
            except ValueError:
                luminosity = Parameter(
                    L,
                    "ps_%i_luminosity" % i,
                    fixed=True,
                    par_range=lumi_range,
                )

            # Check for shared src_index parameter
            try:
                src_index = Parameter.get_parameter("src_index")

            # Else, create individual ps_%i_src_index parameters
            except ValueError:
                src_index = Parameter(
                    index,
                    "ps_%i_src_index" % i,
                    fixed=False,
                    par_range=(1, 4),
                )

            # Create source
            source = PointSource.make_powerlaw_source(
                "ps_%i" % i,
                dec,
                ra,
                luminosity,
                src_index,
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
    @u.quantity_input
    def dec(self, value: u.rad):
        self._dec = value

    @property
    def ra(self):
        return self._ra

    @ra.setter
    @u.quantity_input
    def ra(self, value: u.rad):
        self._ra = value

    @property
    def cosz(self):
        # only valid for IceCube
        # TODO: move to detector model
        return np.cos(self._dec.value + np.pi / 2)

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        self._redshift = value

    @property
    @u.quantity_input
    def luminosity(self) -> u.erg / u.s:
        return self._luminosity

    @luminosity.setter
    @u.quantity_input
    # TODO add calculation for fluxes etc.
    # needs to be defined in the source frame
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

    def __len__(self):
        return self.N

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, value):
        if not isinstance(value, list):
            raise ValueError(str(value) + " is not a list")

        elif not all(isinstance(s, Source) for s in value):
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
    def add_diffuse_component(
        self,
        flux_norm: Parameter,
        norm_energy: u.GeV,
        diff_index: Parameter,
        lower: Parameter,
        upper: Parameter,
        z: float = 0,
    ):
        """
        Add diffuse component based on point
        source component definition. By default, the diffuse background is
        defined at the Earth (z=0) and so the associated background parameters
        are *observed* background properties and not *source population*
        properties.

        :param flux_norm: The flux normalization for this component
        :param norm_energy: The energy at which the flux norm is defined
        :param diff_index: The index of the power law spectrum
        :param lower: Lower energy bound of spectrum, defined at z
        :param upper: Upper energy bound of spectrum, defined at z
        :param z: The redshift of the background shell
        """

        spectral_shape = PowerLawSpectrum(
            flux_norm,
            norm_energy,
            diff_index,
            lower.value / (1.0 + z),
            upper.value / (1.0 + z),
        )
        flux_model = IsotropicDiffuseBG(spectral_shape)

        # define component
        diffuse_component = DiffuseSource("diffuse_bg", z, flux_model=flux_model)

        self.add(diffuse_component)

    def _get_max_ps_redshift(self):
        """
        Check maximum redshift of exsiting point sources.
        """

        z = []
        for source in self.sources:
            if isinstance(source, PointSource):
                z.append(source.redshift)

        return max(z)

    def _get_point_source_spectrum(self):
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

        self._point_source_spectrum = types[0]

    def add_atmospheric_component(self):
        """
        Add an atmospheric flux component based on the IceCube observations.
        """

        Emin = Parameter.get_parameter("Emin").value.to(u.GeV)
        Emax = Parameter.get_parameter("Emax").value.to(u.GeV)

        flux_model = AtmosphericNuMuFlux(Emin, Emax)

        atmospheric_component = DiffuseSource(
            "atmo_bg",
            0,
            flux_model=flux_model,
        )

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

    def f_arr(self):
        """
        Associated fraction of arrival flux at Earth
        from sources.
        """

        flux_units = 1 / (u.m**2 * u.s)

        point_source_ints = (
            sum(
                [
                    s.flux_model.total_flux_int.to(flux_units).value
                    for s in self.sources
                    if isinstance(s, PointSource)
                ]
            )
            * flux_units
        )

        total_ints = self.total_flux_int().to(flux_units)

        return point_source_ints / total_ints

    def f_arr_astro(self):
        """
        Same as `f_arr`, but ignoring any atmospheric
        contribution.
        """

        flux_units = 1 / (u.m**2 * u.s)

        point_source_ints = (
            sum(
                [
                    s.flux_model.total_flux_int.to(flux_units).value
                    for s in self.sources
                    if isinstance(s, PointSource)
                ]
            )
            * flux_units
        )

        if self.diffuse:
            diff_ints = self.diffuse.flux_model.total_flux_int.to(flux_units)

        else:
            diff_ints = 0 << flux_units

        return point_source_ints / (point_source_ints + diff_ints)

    def organise(self):
        """
        Check what sources are in list and make
        sure diffuse and atmo components are second-to-last
        and last respectively (if they exist).

        NB: assumes only one of each diff and atmo component.
        """

        self._point_source = []
        self._diffuse = None
        self._atmospheric = None

        for source in self.sources:
            if isinstance(source, PointSource):
                self._point_source.append(source)

            elif isinstance(source, DiffuseSource):
                if isinstance(source.flux_model, IsotropicDiffuseBG):
                    self._diffuse = source

                elif isinstance(source.flux_model, AtmosphericNuMuFlux):
                    self._atmospheric = source

        if self._point_source:
            self._get_point_source_spectrum()

        new_list = self._point_source.copy()

        if self._diffuse:
            new_list.append(self._diffuse)

        if self._atmospheric:
            new_list.append(self._atmospheric)

        self.sources = new_list

    @property
    def point_source(self):
        self.organise()

        return self._point_source

    @property
    def point_source_spectrum(self):
        self.organise()

        if self._point_source:
            return self._point_source_spectrum

        else:
            raise ValueError("No point sources in  source list")

    @property
    def diffuse(self):
        self.organise()

        return self._diffuse

    @property
    def diffuse_spectrum(self):
        self.organise()

        if self._diffuse:
            return self._diffuse.flux_model.spectral_shape

        else:
            raise ValueError("No diffuse background in source list")

    @property
    def atmospheric(self):
        self.organise()

        return self._atmospheric

    @property
    def atmospheric_flux(self):
        self.organise()

        return self._atmospheric.flux_model

    def __iter__(self):
        for source in self._sources:
            yield source

    def __getitem__(self, key):
        return self._sources[key]

    def save(self, file_name):
        """
        Write the Sources to an HDF5 file.
        """

        raise NotImplementedError()

    @classmethod
    def from_file(cls, file_name):
        """
        Load Sources from the HDF5 file output
        by the save() method.
        """

        raise NotImplementedError()


def uv_to_icrs(unit_vector):
    """
    convert unit vector to ICRS coords (ra, dec)
    """

    if len(np.shape(unit_vector)) > 1:
        theta = np.arccos(unit_vector.T[2])

        phi = np.arctan2(unit_vector.T[1], unit_vector.T[0])
        # Convert to [0, 2pi)
        phi[phi < 0] += 2 * np.pi

    else:
        theta = np.arccos(unit_vector[2])

        phi = np.arctan2(unit_vector[1], unit_vector[0])

        # Convert to [0, 2pi)
        if phi < 0:
            phi += 2 * np.pi
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
