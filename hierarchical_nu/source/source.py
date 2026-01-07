from abc import ABC, abstractmethod
import h5py
from astropy import units as u
import numpy as np
import numpy.typing as npt
from collections.abc import Callable
from typing import Union
from pathlib import Path

from .flux_model import (
    PointSourceFluxModel,
    PowerLawSpectrum,
    LogParabolaSpectrum,
    TwiceBrokenPowerLaw,
    IsotropicDiffuseBG,
    PGammaSpectrum,
)
from .atmospheric_flux import AtmosphericNuMuFlux
from .seyfert_model import SeyfertNuMuSpectrum
from .cosmology import luminosity_distance
from .parameter import Parameter, ParScale
from ..utils.config import HierarchicalNuConfig
from ..detector.r2021_bg_llh import R2021BackgroundLLH

import logging


logger = logging.getLogger(__name__)


class ReferenceFrame(ABC):
    """
    Abstract base class for source frames.
    """

    @staticmethod
    def func(list, *args):
        """
        Method to avoid stan generator issues.
        """

        prev = args[-1]
        try:
            for arg in reversed(args[:-1]):
                prev = arg[prev]
            out = list[prev]
        except UnboundLocalError:
            out = list[prev]
        return out

    @classmethod
    @abstractmethod
    def stan_to_det(cls, E, z):
        pass

    @classmethod
    @abstractmethod
    def stan_to_src(cls, E, z):
        pass

    @classmethod
    @abstractmethod
    def transform(cls, z):
        pass


class DetectorFrame(ReferenceFrame):

    name = "detector"

    @classmethod
    @u.quantity_input
    def transform(cls, E: u.GeV, z: Union[int, float, None] = None):
        return E

    @classmethod
    def stan_to_src(cls, E, z, *indices):
        return E * (1 + cls.func(z, *indices))

    @classmethod
    def stan_to_det(cls, E, z, *indices):
        return E


class SourceFrame(ReferenceFrame):

    name = "source"

    @classmethod
    @u.quantity_input
    def transform(cls, E: u.GeV, z: Union[int, float]):
        return E / (1.0 + z)

    @classmethod
    def stan_to_src(cls, E, z, *indices):
        return E

    @classmethod
    def stan_to_det(cls, E, z, *indices):
        return E / (1.0 + cls.func(z, *indices))


class Source(ABC):
    """
    Abstract base class for sources.
    """

    def __init__(self, name: str, frame: ReferenceFrame, *args, **kwargs):
        self._name = name
        self._frame = frame
        self._parameters = {}
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
    ) -> u.Quantity[1 / (u.GeV * u.m**2 * u.s * u.sr)]:
        return self._flux_model(energy, dec, ra)

    @property
    def frame(self):
        return self._frame


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
        frame: Instance of `ReferenceFrame` in which energies are defined
    """

    @u.quantity_input
    def __init__(
        self,
        name: str,
        dec: u.rad,
        ra: u.rad,
        redshift: float,
        spectral_shape: Callable[[float], float],
        frame: ReferenceFrame = SourceFrame,
        *args,
        **kwargs,
    ):
        super().__init__(name, frame)
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
        frame: ReferenceFrame = SourceFrame,
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
            frame: ReferenceFrame
                Reference frame in which source energy is defined
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

        # Use Enorm if set, otherwise fix to 1e5 GeV, arbitrary in any case
        try:
            Enorm_value = Parameter.get_parameter("Enorm").value
        except ValueError:
            Enorm_value = 1e5 * u.GeV

        # Transform energies to detector frame
        spectral_shape = PowerLawSpectrum(
            norm,
            Enorm_value,
            index,
            frame.transform(lower.value, redshift),
            frame.transform(upper.value, redshift),
        )

        total_power = spectral_shape.total_flux_density
        norm.value *= total_flux / total_power
        norm.value = norm.value.to(1 / (u.GeV * u.m**2 * u.s))
        norm.fixed = True
        return cls(name, dec, ra, redshift, spectral_shape, frame)

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
        frame: ReferenceFrame = SourceFrame,
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
            frame: ReferenceFrame
                Reference frame in which source energy is defined
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
        try:
            Enorm_value = Parameter.get_parameter("Enorm").value
        except ValueError:
            Enorm_value = 1e5 * u.GeV

        # Transform energies to detector frame
        spectral_shape = TwiceBrokenPowerLaw(
            norm,
            Enorm_value,
            index,
            frame.transform(lower.value, redshift),
            frame.transform(upper.value, redshift),
        )

        total_power = spectral_shape.total_flux_density
        norm.value *= total_flux / total_power
        norm.value = norm.value.to(1 / (u.GeV * u.m**2 * u.s))
        norm.fixed = True
        return cls(name, dec, ra, redshift, spectral_shape, frame)

    @classmethod
    @u.quantity_input
    def make_logparabola_source(
        cls,
        name: str,
        dec: u.rad,
        ra: u.rad,
        luminosity: Parameter,
        alpha: Parameter,
        beta: Parameter,
        redshift: float,
        lower: Parameter,
        upper: Parameter,
        normalisation_energy: Parameter,
        frame: ReferenceFrame = SourceFrame,
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
            alpha: Parameter
                Spectral index
            beta: Parameter
                Curvature parameter
            redshift: float
            lower: Parameter
                Lower energy bound
            upper: Parameter
                Upper energy bound
            normalisation_energy: Parameter
                Normalisation energy of spectrum
            frame: ReferenceFrame
                Reference frame in which source energy is defined
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

        fixed = normalisation_energy.fixed
        # TODO this is really ugly, find a better solution to this
        normalisation_energy.fixed = False
        val = normalisation_energy.value
        normalisation_energy.value = frame.transform(val, redshift)
        par_min, par_max = normalisation_energy.par_range
        par_range = (
            frame.transform(par_min, redshift),
            frame.transform(par_max, redshift),
        )
        normalisation_energy.par_range = par_range
        normalisation_energy.fixed = fixed
        spectral_shape = LogParabolaSpectrum(
            norm,
            normalisation_energy,
            alpha,
            beta,
            frame.transform(lower.value, redshift),
            frame.transform(upper.value, redshift),
        )

        total_power = spectral_shape.total_flux_density
        norm.value *= total_flux / total_power
        norm.value = norm.value.to(1 / (u.GeV * u.m**2 * u.s))
        norm.fixed = True
        return cls(name, dec, ra, redshift, spectral_shape, frame)

    @classmethod
    @u.quantity_input
    def make_pgamma_source(
        cls,
        name: str,
        dec: u.rad,
        ra: u.rad,
        luminosity: Parameter,
        redshift: float,
        E0_src: Parameter,
        lower: Parameter,
        upper: Parameter,
        frame: ReferenceFrame = SourceFrame,
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
            redshift: float
            E0_src: Parameter
                Energy at which flat spectrum evolves into logparabola, is defined at detector irregardless of `frame`
                NB: Choose wide enough s.t. redshifting does not affect the result
            lower: Parameter
                Lower energy bound
            upper: Parameter
                Upper energy bound
            frame: ReferenceFrame
                Reference frame in which source energy is defined
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

        spectral_shape = PGammaSpectrum(
            norm,
            E0_src,
            frame.transform(lower.value, redshift),
            frame.transform(upper.value, redshift),
        )

        total_power = spectral_shape.total_flux_density
        norm.value *= total_flux / total_power
        norm.value = norm.value.to(1 / (u.GeV * u.m**2 * u.s))
        norm.fixed = True
        return cls(name, dec, ra, redshift, spectral_shape, frame)

    @classmethod
    def make_seyfert_source(
        cls,
        name: str,
        dec: u.rad,
        ra: u.rad,
        logLx: float,
        P: Parameter,
        eta: Parameter,
        redshift: float,
        energy_points: int = 80,
        eta_points: int = 100,
    ):
        """
        Create source with Seyfert II neutrino flux.
        Assumes default energy range of 1e2GeV to 1e7GeV in the detector frame.
        Parameters:
            name: str
                Source name
            dec: u.rad,
                Declination of the source
            ra: u.rad,
                Right Ascension of the source
            logLx: float
                x-ray luminosity of source in log10(L_x / (erg / s)),
                rounds to nearest .01
            P: Parameter,
                Cosmic ray pressure to thermal pressure ratio, acts as normalisation
            eta: Parameter
                Inverse magnetic turbulence strength
            redshift: float
            energy_points: int
                Number of grid points for energy interpolation in stan
            eta_points: int
                Number of grid points for eta interpolation in stan
        """

        spectral_shape = SeyfertNuMuSpectrum(
            logLx, P, eta, redshift, energy_points, eta_points, name
        )
        return cls(
            name,
            dec,
            ra,
            redshift,
            spectral_shape,
            DetectorFrame,
        )

    @classmethod
    def make_seyfert_sources_from_file(
        cls, file_name: Union[Path, str], shared_P: bool, shared_eta: bool
    ):
        with h5py.File(file_name, "r") as f:
            # Subject to change, what do we need here
            ras = f["phi"][()] * u.rad
            selection = f["selection"][()]  # TODO properly implement or leave out?
            decs = -(f["theta"][()] - np.pi / 2) * u.rad
            distances = f["distances"][()]  # redshift, for luminosity distance
            logLx = f["luminosities"][()]  # x-ray luminosity, properly implement
            pressure_ratios = f["pressure_ratio"][()]  # only important for simulations
            etas = f["eta"][()]  # same

        if shared_eta:
            eta = Parameter(40, "eta", fixed=False, par_range=(2, 150))
        if shared_P:
            P = Parameter(0.2, "pressure_ratio", fixed=True, par_range=(0.0, 0.5))

        source_list = []
        for c, (r, d, z, lLx) in enumerate(zip(ras, decs, distances, logLx)):
            if not shared_eta:
                eta = Parameter(etas[c], f"ps_{c}_eta", fixed=False, par_range=(2, 150))
            if not shared_P:
                P = Parameter(
                    pressure_ratios[c],
                    f"ps_{c}_pressure_ratio",
                    fixed=True,
                    par_range=(0, 0.5),
                )
            ps = PointSource.make_seyfert_source(
                f"ps_{c}",
                d,
                r,
                lLx,
                P,
                eta,
                z,
            )
            source_list.append(ps)
        return source_list

    @classmethod
    def _make_sources_from_file(
        cls,
        file_name: str,
        lower_energy: Parameter,
        upper_energy: Parameter,
        method: Callable,
        frame: ReferenceFrame = SourceFrame,
        include_undetected: bool = False,
        config: Union[None, HierarchicalNuConfig] = None,
        normalisation_energy: Union[u.Quantity[u.GeV], None] = None,
    ):
        # Sensible bounds on luminosity
        lumi_range = (0, 1e60) * (u.erg / u.s)
        if config is not None:
            lumi_range = tuple(config.parameter_config.L_range) * u.erg / u.s

        # Load values
        with h5py.File(file_name, "r") as f:
            luminosities = f["luminosities"][()] * (u.erg / u.s)

            spectral_indices = f["auxiliary_quantities/spectral_index/obs_values"][()]

            redshifts = f["distances"][()]

            ras = f["phi"][()] * u.rad

            decs = -(f["theta"][()] - np.pi / 2) * u.rad

            selection = f["selection"][()]

        # Apply selection
        if not include_undetected:
            luminosities = luminosities[selection]

            spectral_indices = spectral_indices[selection]

            redshifts = redshifts[selection]

            ras = ras[selection]

            decs = decs[selection]

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
                par_range = (1, 4)
                if config is not None:
                    par_range = config.parameter_config.src_index_range
                src_index = Parameter(
                    index,
                    "ps_%i_src_index" % i,
                    fixed=False,
                    par_range=par_range,
                )

            # Create source
            source = method(
                "ps_%i" % i,
                dec,
                ra,
                luminosity,
                src_index,
                z,
                lower_energy,
                upper_energy,
                frame=frame,
            )

            source_list.append(source)

        return source_list

    @classmethod
    def make_powerlaw_sources_from_file(
        cls,
        file_name: str,
        lower_energy: Parameter,
        upper_energy: Parameter,
        frame: ReferenceFrame = SourceFrame,
        include_undetected: bool = False,
        config: Union[None, HierarchicalNuConfig] = None,
    ):
        """
        Factory for power law sources defined in
        HDF5 files ( update: output from popsynth).

        :param file_name: File name of source list.
        :param lower_energy: Lower energy bound in definition of the luminosity.
        :param upper_energy: Upper energy bound in definition of the luminosity.
        :param frame: Reference frame in which source energy is defined
        :param include_undetected: Include sources that are not detected in population.
        :param config: Instance of HierarchicalNuConfig to check for parameter bounds.
        """

        source_list = cls._make_sources_from_file(
            file_name,
            lower_energy,
            upper_energy,
            cls.make_powerlaw_source,
            frame,
            include_undetected,
            config,
        )

        return source_list

    @classmethod
    def make_broken_powerlaw_sources_from_file(
        cls,
        file_name: str,
        lower_energy: Parameter,
        upper_energy: Parameter,
        frame: ReferenceFrame = SourceFrame,
        include_undetected: bool = False,
        config: Union[None, HierarchicalNuConfig] = None,
    ):
        """
        Factory for power law sources defined in
        HDF5 files ( update: output from popsynth).

        :param file_name: File name of source list.
        :param lower_energy: Lower energy bound in definition of the luminosity.
        :param upper_energy: Upper energy bound in definition of the luminosity.
        :param frame: Reference frame in which source energy is defined
        :param include_undetected: Include sources that are not detected in population.
        :param config: Instance of HierarchicalNuConfig to check for parameter bounds.
        """

        source_list = cls._make_sources_from_file(
            file_name,
            lower_energy,
            upper_energy,
            cls.make_twicebroken_powerlaw_source,
            frame,
            include_undetected,
            config,
        )

        return source_list

    '''
    @classmethod
    def make_logparabola_sources_from_file(
        cls,
        file_name: str,
        lower_energy: Parameter,
        upper_energy: Parameter,
        normalisation_energy: u.GeV
        frame: ReferenceFrame = SourceFrame,
        include_undetected: bool = False,
        config: Union[None, HierarchicalNuConfig] = None,
    ):
        """
        Factory for power law sources defined in
        HDF5 files ( update: output from popsynth).

        :param file_name: File name of source list.
        :param lower_energy: Lower energy bound in definition of the luminosity.
        :param upper_energy: Upper energy bound in definition of the luminosity.
        :param frame: Reference frame in which source energy is defined
        :param include_undetected: Include sources that are not detected in population.
        :param config: Instance of HierarchicalNuConfig to check for parameter bounds.
        """

        source_list = cls._make_sources_from_file(
            file_name,
            lower_energy,
            upper_energy,
            cls.make_logparabola_source,
            frame,
            include_undetected,
            config,
            normalisation_energy=normalisation_energy,
        )

        return source_list
    '''

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
        return np.cos(self._dec.to_value(u.rad) + np.pi / 2)

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        self._redshift = value

    @property
    @u.quantity_input
    def luminosity(self) -> u.Quantity[u.erg / u.s]:
        return self._luminosity

    """
    @luminosity.setter
    @u.quantity_input
    # TODO add calculation for fluxes etc.
    # needs to be defined according to the ReferenceFrame
    def luminosity(self, value: u.Quantity[u.erg / u.s]):
        self._luminosity = value
    """


class DiffuseSource(Source):
    """
    DiffuseSource

    Parameters:
        name: str
        redshift: float
        flux_model
            Flux model of the source. Should return units 1/(GeV cm^2 s sr)
        frame
            Reference frame in which the source is defined
    """

    def __init__(
        self,
        name: str,
        redshift: float,
        flux_model,
        frame: ReferenceFrame = SourceFrame,
        *args,
        **kwargs,
    ):
        super().__init__(name, frame)
        self._redshift = redshift
        self._flux_model = flux_model
        self._parameters = flux_model.parameters

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        self._redshift = value


class BackgroundSource(Source):
    """
    Class that models background with data
    """

    def __init__(self, name, *detector_model):
        from ..detector.r2021_bg_llh import R2021BackgroundLLH

        super().__init__(name, DetectorFrame)
        self._name = name
        self._flux_model = None
        self._detector_model = detector_model

        self._likelihoods = {_: R2021BackgroundLLH(_.P) for _ in detector_model}
        self._frame = DetectorFrame

    def flux(self):
        raise NotImplementedError("Data-driven background model has no flux.")


class Sources:
    """
    Container for sources with a set of factory methods
    for an easy source setup interface.
    """

    def __init__(self):
        # Initialise the source list
        self._sources = []

    def __len__(self):
        return len(self.sources)

    @property
    def N(self):
        return len(self)

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

    def add(self, source):
        """
        Add any instance of a class inheriting from Source.
        """

        if isinstance(source, Source):
            self._sources.append(source)

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
        frame: ReferenceFrame = SourceFrame,
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
            frame.transform(lower.value, z),
            frame.transform(upper.value, z),
        )
        flux_model = IsotropicDiffuseBG(spectral_shape)

        # define component
        diffuse_component = DiffuseSource(
            "diffuse_bg", z, flux_model=flux_model, frame=frame
        )

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

    def _get_point_source_frame(self):
        """
        Check what frame point sources are defined in.
        """

        frames = []
        for source in self.sources:
            if isinstance(source, PointSource):
                frames.append(source.frame)

        if not frames[1:] == frames[:-1]:
            raise NotImplementedError(
                "All point sources must be defined in the same RefrenceFrame"
            )

        self._point_source_frame = frames[0]

    @property
    def point_source_frame(self):
        self._get_point_source_frame()
        return self._point_source_frame

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

    @property
    def point_source_spectrum(self):
        self._get_point_source_spectrum()
        return self._point_source_spectrum

    def add_atmospheric_component(self, index: float = 0.0, cache_dir: str = ".cache"):
        """
        Add an atmospheric flux component based on the IceCube observations.
        """

        Emin = Parameter.get_parameter("Emin").value.to(u.GeV)
        Emax = Parameter.get_parameter("Emax").value.to(u.GeV)

        flux_model = AtmosphericNuMuFlux(Emin, Emax, index=index, cache_dir=cache_dir)

        # Create a parameter for F_atmo to carry information on the par_range
        # The value itself is irrelevant for fits
        F_atmo = Parameter(
            flux_model.total_flux_int,
            "F_atmo",
            par_range=(0, 3.0) * (1 / (u.m**2 * u.s)),
            fixed=True,
        )

        atmospheric_component = DiffuseSource(
            "atmo_bg",
            0,
            flux_model=flux_model,
            frame=DetectorFrame,
        )

        self.add(atmospheric_component)

    def add_background(self, *detector_model):
        self.add(BackgroundSource("bg", *detector_model))

    def select(self, mask: npt.NDArray[np.bool_], only_point_sources: bool = False):
        """
        Select some subset of existing sources by providing a mask.
        NB: Assumes only one diffuse and one atmospheric component
        :param mask: Array of bools with same length as the number of sources.
        :param only_point_sources: Set `True` to only make selections on point sources
        """

        if only_point_sources:
            assert len(mask) == len(self.point_source)

            # Remove parameters describing removed point sources
            for i, s in enumerate(self.point_source):
                if not mask[i]:
                    Parameter.remove_parameter("%s_src_index" % s.name)
                    Parameter.remove_parameter("%s_luminosity" % s.name)
                    Parameter.remove_parameter("%s_norm" % s.name)

            # Make selection on point sources
            _point_sources = np.array(self._point_source)

            _point_sources = _point_sources[mask]

            _sources = list(_point_sources)

            # Add back diffuse and atmospheric components
            _sources.append(self.diffuse)
            _sources.append(self.atmospheric)
            _sources.append(self.background)

        else:
            assert len(mask) == self.N

            # Remove parameters describing removed point sources
            for i, s in enumerate(self.sources):
                if not mask[i] and isinstance(s, PointSource):
                    Parameter.remove_parameter("%s_src_index" % s.name)
                    Parameter.remove_parameter("%s_luminosity" % s.name)
                    Parameter.remove_parameter("%s_norm" % s.name)

            # Select on over all sources
            _sources = np.array(self.sources)

            _sources = _sources[mask]

        self.sources = list(_sources)

    def select_below_redshift(self, zth):
        """
        Remove sources with redshift above a certain threshold.
        NB: Replaced by `Sources.select()`, but kept for backwards compatibility.
        """

        self.sources = [s for s in self.sources if s.redshift <= zth]

    def remove(self, i):
        self._sources.pop(i)

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
        self._background = None

        for source in self.sources:
            if isinstance(source, PointSource):
                self._point_source.append(source)

            elif isinstance(source, DiffuseSource):
                if isinstance(source.flux_model, IsotropicDiffuseBG):
                    self._diffuse = source

                elif isinstance(source.flux_model, AtmosphericNuMuFlux):
                    self._atmospheric = source
            elif isinstance(source, BackgroundSource):
                self._background = source

        if self._point_source:
            self._get_point_source_spectrum()
            self._get_point_source_frame()

        new_list = self._point_source.copy()

        if self._diffuse:
            new_list.append(self._diffuse)

        if self._atmospheric:
            new_list.append(self._atmospheric)

        if self._background:
            new_list.append(self._background)

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
            logger.warning("No point sources in source list")
            return None

    @property
    def point_source_frame(self):
        self.organise()

        if self._point_source:
            return self._point_source_frame
        else:
            logger.warning("No point sources in source list")
            return None

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

    @property
    def background(self):
        self.organise()

        return self._background

    @property
    def background_flux(self):
        raise NotImplementedError()

    def __iter__(self):
        for source in self._sources:
            yield source

    def __getitem__(self, key):
        return self._sources[key]

    def __bool__(self):
        return bool(len(self))

    def make_seyfert_functions(self):
        lpdf = []
        flux_tab = []
        flux_conv = []
        for s in self.point_source:
            ret = s._flux_model.spectral_shape.make_stan_functions()
            lpdf.append(ret[0])
            flux_tab.append(ret[1])
            flux_conv.append(ret[2])

        return lpdf, flux_tab, flux_conv


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
