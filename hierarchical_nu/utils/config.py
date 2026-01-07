import os
from pathlib import Path
from typing import List, Tuple, Any
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import numpy as np
import logging
import astropy.units as u


_local_config_path = Path(".")
_config_name = Path("hnu_config.yml")


# Overwrite global config with local config
_local_config_file = _local_config_path / _config_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ParameterConfig:
    source_type: str = (
        "twice-broken-power-law"  # Currently only supports one type for all sources,
        # other options: "power-law" covering the entire energy range,
        # "logparabola", "pgamma" or "seyfert". If logparabola, up to two fit parameters used
        # (out of src_index, beta_index and E0_src) need to be defined
        # in the field "fit_params", e.g. fit_params: ["src_index", "beta_index"]
    )
    # Takes spectral shape parameters as arguments, i.e. src_index, beta_index, E0_src
    # depending on chosen spectral type. Optional add Nex_src to use directly
    # number of expected source events as fit parameter. This bypasses the luminosity
    # (which is demoted to transformed parameter) and its prior.
    fit_params: List[str] = field(default_factory=lambda: ["src_index"])
    src_index: List[float] = field(default_factory=lambda: [2.3])
    share_src_index: bool = True
    src_index_range: Tuple = (1.0, 4.0)
    beta_index: List[float] = field(default_factory=lambda: [0.0])
    beta_index_range: Tuple = (-1.0, 1.0)
    eta: List[float] = field(
        default_factory=lambda: [40.0]
    )  # inverse turbulence strength
    eta_range: Tuple = (2.0, 150.0)
    P: List[float] = field(
        default_factory=lambda: [0.4]
    )  # cosmic ray to thermal pressure ratio
    P_range: Tuple = (0.0, 0.5)
    Nex_src_range: Tuple = (0.0, 100.0)
    E0_src: List[str] = field(default_factory=lambda: ["1e6 GeV"])
    E0_src_range: Tuple[str] = ("1e3 GeV", "1e8 GeV")
    diff_index: float = 2.5
    diff_index_range: Tuple = (1.0, 4.0)
    diff_norm: str = "2.26e-13 GeV-1 m-2 s-1"  # defined in the detector frame
    diff_norm_range: Tuple[str] = (
        "1e-14 GeV-1 m-2 s-1",
        "1e-11 GeV-1 m-2 s-1",
    )
    F_atmo_range: Tuple = ("0.1 m-2 s-1", "0.5 m-2 s-1")
    L: List[str] = field(
        default_factory=lambda: ["8e45 erg s-1"]
    )  # u.erg / u.s, defined in the source frame
    share_L: bool = True
    L_range: Tuple = ("0 GeV s-1", "1e60 GeV s-1")
    src_dec: List[str] = field(default_factory=lambda: ["0.0 deg"])
    src_ra: List[str] = field(default_factory=lambda: ["90.0 deg"])
    Enorm: str = "1e5 GeV"  # defined in the detector frame
    Emin: str = "1e2 GeV"  # defined in the detector frame
    Emax: str = "1e8 GeV"
    Emin_src: str = "1.4e4 GeV"
    Emax_src: str = "1.4e7 GeV"
    Emin_diff: str = "1e2 GeV"  # defined in the detector frame
    Emax_diff: str = "1e8 GeV"  # defined in the detector frame
    z: List[float] = field(
        default_factory=lambda: [0.4]
    )  # cosmological redshift, dimensionless, only for point source

    # If True, use same Emin_det for all
    # If False, use separate for tracks and cascades
    Emin_det_eq: bool = True

    # Entries for un-used detector models are disregarded by the sim/fit/model check
    # defined in the detector frame
    Emin_det: str = "1e4 GeV"
    Emin_det_northern_tracks: str = "6e4 GeV"
    Emin_det_cascades: str = "6e4 GeV"
    Emin_det_IC40: str = "6e4 GeV"
    Emin_det_IC59: str = "6e4 GeV"
    Emin_det_IC79: str = "6e4 GeV"
    Emin_det_IC86_I: str = "6e4 GeV"
    Emin_det_IC86_II: str = "6e4 GeV"

    # Can be NT, CAS or IC40 through IC86_II or any combination,
    # see `hierarchical_nu.detector.icecube.Refrigerator`
    # needs to be the Python-string, accessed through e.g. NT.P
    # due to merging of the yaml config and this config here

    # two options:
    # either provide a list of detector names and in the same order
    # the obs times
    # OR
    # provide mjd_min, mjd_max to automatically determine the detectors and their obs times
    detector_model_type: List[str] = field(default_factory=lambda: ["IC86_II"])
    frame: str = "source"
    obs_time: List[str] = field(default_factory=lambda: ["3.0 yr"])

    # With these default values obs_time takes precedence
    MJD_min: float = 98.0
    MJD_max: float = 100.0
    # restrict DM selection from MJD to selection in detector_model_type
    # useful because of overlap near season changes
    restrict_to_list: bool = False

    # Background components
    atmospheric: bool = True
    diffuse: bool = True
    data_bg: bool = False  # Use this for data-driven background likelihood

    # Asimov data - fix simulated event numbers to nearest integer of expected number
    asimov: bool = False

    # exp event selection
    scramble_ra: bool = False
    scramble_mjd: bool = False  # TODO: add functionality

    # use event tags, only relevant for multi ps fits
    use_event_tag: bool = False

    logLx: List[float] = field(
        default_factory=lambda: [43.62]
    )  # log10(x-ray luminosity) in 2-10keV, only used for Seyfert spectra


@dataclass
class StanConfig:
    # Within-chain parallelisation
    threads_per_chain: int = 1
    chains: int = 1
    iterations: int = 1000
    iter_warmup: int = 1000
    adapt_delta: float = 0.8  # Default of cdmstanpy
    max_treedepth: int = 10  # Default of cmdstanpy
    seed: int = 42


@dataclass
class SinglePriorConfig:
    name: str = "LogNormalPrior"
    mu: Any = 1.0  # Should be str or float, but alas, OmegaConf does not support Unions
    sigma: Any = 1.0


@dataclass
class PriorConfig:
    src_index: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="NormalPrior", mu=[2.5], sigma=[0.5]
        )
    )

    beta_index: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="NormalPrior", mu=[0.0], sigma=[0.1]
        )
    )
    E0_src: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="LogNormalPrior", mu=["1e5 GeV"], sigma=[3.0]
        )
    )
    diff_index: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="NormalPrior", mu=2.52, sigma=0.04
        )
    )
    L: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="LogNormalPrior", mu=["1e49 GeV s-1"], sigma=[3.0]
        )
    )
    diff_flux: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="NormalPrior",
            mu="2.26e-13  GeV-1 s-1 m-2",
            sigma="0.19e-13 GeV-1 s-1 m-2",
        )
    )
    atmo_flux: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="NormalPrior", mu="0.314 m-2 s-1", sigma="0.08 m-2 s-1"
        )
    )
    eta: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="Ignorance",
            mu=[1.0],
            sigma=[1.0],
        )
    )
    P: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="Ignorance",
            mu=[1.0],
            sigma=[1.0],
        )
    )
    Nex_src: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="Ignorance",
            mu=[1.0],
            sigma=[1.0]
        )
    )


@dataclass
class ROIConfig:
    roi_type: str = (
        "CircularROI"  # can be "CircularROI", "FullSkyROI", or "RectangularROI"
    )
    size: str = (
        "5.0 deg"  # for circular: radius, fullsky: disregarded, rectangular: center +/- size in RA and DEC
    )
    apply_roi: bool = False

    # If config has default values size data field takes precedence
    RA_min: str = "-1.0 deg"
    RA_max: str = "361.0 deg"

    DEC_min: str = "-91.0 deg"
    DEC_max: str = "91.0 deg"

    # If RA, DEC take default values, they will be overridden by the source selections
    RA: str = "-1.0 deg"
    DEC: str = "-91.0 deg"


@dataclass
class PPCConfig:
    n_samples: int = 100
    seed: int = 42
    directory: str = "ppc"


@dataclass
class HierarchicalNuConfig:
    parameter_config: ParameterConfig = field(default_factory=lambda: ParameterConfig())
    prior_config: PriorConfig = field(default_factory=lambda: PriorConfig())
    roi_config: ROIConfig = field(default_factory=lambda: ROIConfig())
    stan_config: StanConfig = field(default_factory=lambda: StanConfig())
    ppc_config: PPCConfig = field(default_factory=lambda: PPCConfig())

    @classmethod
    def from_path(cls, path):
        """
        Load config from path
        """

        hnu_config = OmegaConf.structured(cls)
        local_config = OmegaConf.load(path)
        hnu_config = OmegaConf.merge(hnu_config, local_config)
        return hnu_config

    @classmethod
    def save_default(cls, path: Path = _local_config_file):
        """
        Save default config to path.
        If the path does not exist, it is created
        """

        if not isinstance(path, Path):
            path = Path(path)
        hnu_config = OmegaConf.structured(cls)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            OmegaConf.save(config=hnu_config, f=f.name)

    @classmethod
    def load_default(cls):
        """
        Load default config
        """

        hnu_config = OmegaConf.structured(cls)
        return hnu_config

    @classmethod
    def save(cls, path: Path, config):
        path.parents[0].mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            OmegaConf.save(config=config, f=f.name)

    @classmethod
    def make_config(cls, sources):
        from ..source.parameter import Parameter

        # this is useful for recreating a source list when loading a previously saved fit.
        # is not intended to exactly recreate everything, maybe in future edits
        config = HierarchicalNuConfig.load_default()

        if sources.point_source:
            ra = []
            dec = []
            z = []
            fit_params = []
            ps = sources.point_source[0]
            spectrum = ps.flux_model.spectral_shape.name
            try:
                index = ps.flux_model.parameters["index"]
                config.parameter_config.src_index_range = list(index.par_range)
                if not index.fixed:
                    fit_params.append("src_index")
            except KeyError:
                pass
            try:
                beta = ps.flux_model.parameters["beta"]
                config.parameter_config.beta_index_range = list(beta.par_range)
                if not beta.fixed:
                    fit_params.append("beta_index")
            except KeyError:
                pass
            try:
                E0 = ps.flux_model.parameters["norm_energy"]
                config.parameter_config.E0_src_range = [
                    E0.par_range[0].to(u.GeV).to_string(),
                    E0.par_range[1].to(u.GeV).to_string(),
                ]
                if not E0.fixed:
                    fit_params.append("E0_src")
            except KeyError:
                pass
            config.parameter_config.fit_params = fit_params
            for ps in sources.point_source:
                ra.append(ps.ra.to(u.deg).to_string())
                dec.append(ps.dec.to(u.deg).to_string())
                z.append(ps.redshift)
            config.parameter_config.source_type = spectrum
            config.parameter_config.src_ra = ra
            config.parameter_config.src_dec = dec
            config.parameter_config.z = z
            config.parameter_config.share_src_index
            config.parameter_config.share_L
            config.parameter_config.frame = sources.point_source_frame.name
            config.parameter_config.Emin_src = (
                Parameter.get_parameter("Emin_src").value.to(u.GeV).to_string()
            )
            config.parameter_config.Emax_src = (
                Parameter.get_parameter("Emax_src").value.to(u.GeV).to_string()
            )

        config.parameter_config.diffuse = True if sources.diffuse else False
        config.parameter_config.atmospheric = True if sources.atmospheric else False

        return config
