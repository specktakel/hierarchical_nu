import os
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import numpy as np
import logging


_local_config_path = Path(".")
_config_name = Path("hnu_config.yml")


# Overwrite global config with local config
_local_config_file = _local_config_path / _config_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ParameterConfig:
    source_type: str = (
        "twice-broken-power-law"  # Currently only support one type for all sources, other option "power-law" covering the entire energy range
    )
    src_index: List[float] = field(default_factory=lambda: [2.3])
    share_src_index: bool = True
    src_index_range: tuple = (1.0, 4.0)
    diff_index: float = 2.5
    diff_index_range: tuple = (1.0, 4.0)
    F_diff_range: tuple = (1e-6, 1e-3) # 1 / m**2 / s
    F_atmo_range: tuple = (0.1, 0.5) # 1 / m**2 / s
    L: List[float] = field(
        default_factory=lambda: [8e45]
    )  # u.erg / u.s, defined in the source frame
    share_L: bool = True
    L_range: tuple = (0, 1e60)
    src_dec: List[float] = field(default_factory=lambda: [0.0])  # u.deg
    src_ra: List[float] = field(default_factory=lambda: [90.0])  # u.deg
    Enorm: float = 1e5  # u.GeV, defined in the detector frame
    Emin: float = 1e2  # u.GeV, defined in the detector frame
    Emax: float = 1e8  # u.GeV
    Emin_src: float = 1.4e4  # u.GeV, defined in the source frame at redshift z
    Emax_src: float = 1.4e7  # u.GeV
    Emin_diff: float = 1e2  # u.GeV, defined in the detector frame
    Emax_diff: float = 1e8  # u.GeV
    diff_norm: float = (
        2e-13  # 1 / (u.GeV * u.m**2 * u.s), defined in the detector frame
    )
    z: List[float] = field(
        default_factory=lambda: [0.4]
    )  # cosmological redshift, dimensionless, only for point source

    # If True, use same Emin_det for all
    # If False, use separate for tracks and cascades
    Emin_det_eq: bool = True

    # Entries for un-used detector models are disregarded by the sim/fit/model check
    # defined in the detector frame
    Emin_det: float = 1e4  # u.GeV
    Emin_det_northern_tracks: float = 6e4  # u.GeV
    Emin_det_cascades: float = 6e4  # u.GeV
    Emin_det_IC40: float = 6e4  # u.GeV
    Emin_det_IC59: float = 6e4  # u.GeV
    Emin_det_IC79: float = 6e4  # u.GeV
    Emin_det_IC86_I: float = 6e4  # u.GeV
    Emin_det_IC86_II: float = 6e4  # u.GeV

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
    obs_time: List = field(default_factory=lambda: [3.0])  # years

    # With these default values obs_time takes precedence
    MJD_min: float = 98.0
    MJD_max: float = 100.0
    # restrict DM selection from MJD to selection in detector_model_type
    # useful because of overlap near season changes
    restrict_to_list: bool = False

    # Within-chain parallelisation
    threads_per_chain: int = 1
    chains: int = 1
    iterations: int = 1000
    iter_warmup: int = 1000

    # Background components
    atmospheric: bool = True
    diffuse: bool = True

    # Asimov data - fix simulated event numbers to nearest integer of expected number
    asimov: bool = False

    # exp event selection
    scramble_ra: bool = False


@dataclass
class SinglePriorConfig:
    name: str = "LogNormalPrior"
    mu: float = 1.0
    sigma: float = 1.0


@dataclass
class PriorConfig:
    src_index: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(name="NormalPrior", mu=2.0, sigma=1.5)
    )
    diff_index: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="NormalPrior", mu=2.52, sigma=0.04
        )
    )
    L: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="LogNormalPrior", mu=1e49, sigma=3
        )
    )

    diff_flux: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="LogNormalPrior", mu=3e-8, sigma=0.5
        )
    )
    atmo_flux: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="NormalPrior", mu=0.3, sigma=0.08
        )
    )


@dataclass
class ROIConfig:
    roi_type: str = (
        "CircularROI"  # can be "CircularROI", "FullSkyROI", or "RectangularROI"
    )
    size: float = (
        5.0  # size in degrees; for circular: radius, fullsky: disregarded, rectangular: center +/- size in RA and DEC
    )
    apply_roi: bool = False


@dataclass
class HierarchicalNuConfig:
    parameter_config: ParameterConfig = field(default_factory=lambda: ParameterConfig())
    prior_config: PriorConfig = field(default_factory=lambda: PriorConfig())
    roi_config: ROIConfig = field(default_factory=lambda: ROIConfig())

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
