import os
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import numpy as np
import logging


from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
from hierarchical_nu.detector.icecube import Refrigerator, EventType, IC86_II

_config_path = Path("~/.config/hierarchical_nu/").expanduser()
_local_config_path = Path(".")
_config_name = Path("hnu_config.yml")

_config_file = _config_path / _config_name

# Overwrite global config with local config
_local_config_file = _local_config_path / _config_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class FileConfig:
    sim_filename: str = os.path.join(STAN_GEN_PATH, "sim_code.stan")
    fit_filename: str = os.path.join(STAN_GEN_PATH, "model_code.stan")
    include_paths: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.include_paths = [STAN_PATH]


@dataclass
class ParameterConfig:
    src_index: List[float] = field(default_factory=lambda: [2.3])
    share_src_index: bool = True
    src_index_range: tuple = (1.0, 4.0)
    diff_index: float = 2.5
    diff_index_range: tuple = (1.0, 4.0)
    L: List[float] = field(
        default_factory=lambda: [4e46]
    )  # u.erg / u.s, defined in the source frame
    share_L: bool = True
    L_range: tuple = (0, 1e60)
    src_dec: List[float] = field(default_factory=lambda: [0.0])  # u.deg
    src_ra: List[float] = field(default_factory=lambda: [90.0])  # u.deg
    Enorm: float = 1e5  # u.GeV, defined in the detector frame
    Emin: float = 5e4  # u.GeV, defined in the detector frame
    Emax: float = 1e8  # u.GeV
    Emin_src: float = 1.4e5  # u.GeV, defined in the source frame at redshift z
    Emax_src: float = 1.4e8  # u.GeV
    Emin_diff: float = 5e4  # u.GeV, defined in the detector frame
    Emax_diff: float = 1e8  # u.GeV
    diff_norm: float = (
        2e-13  # 1 / (u.GeV * u.m**2 * u.s), defined in the detector frame
    )
    z: List[float] = field(
        default_factory=lambda: [0.4]
    )  # cosmological redshift, dimensionless, only for point source

    # If True, use same Emin_det for all
    # If False, use separate for tracks and cascades
    Emin_det_eq: bool = False

    # Entries for un-used detector models are disregarded by the sim/fit/model check
    # defined in the detector frame
    Emin_det: float = 1e5  # u.GeV
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
    detector_model_type: List[str] = field(default_factory=lambda: ["IC86_II"])

    obs_time: List[float] = field(default_factory=lambda: [6.0])  # years

    # Within-chain parallelisation
    threads_per_chain: int = 1
    chains: int = 1
    iterations: int = 1000
    iter_warmup: int = 1000

    # Background components
    atmospheric: bool = True
    diffuse: bool = True

    # Asimov data - fix simulated event numbers to nearest integer of expected number
    # asimov: bool = False


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
            name="NormalPrior", mu=2.37, sigma=0.09
        )
    )
    L: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="LogNormalPrior", mu=1e49, sigma=3
        )
    )

    diff_flux: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="LogNormalPrior", mu=9.4e-5, sigma=1.0
        )
    )
    atmo_flux: SinglePriorConfig = field(
        default_factory=lambda: SinglePriorConfig(
            name="NormalPrior", mu=3e-1, sigma=0.08
        )
    )


@dataclass
class ROIConfig:
    roi_type: str = (
        "CircularROI"  # can be "CircularROI", "FullSkyROI", or "RectangularROI"
    )
    size: float = 5.0  # size in degrees; for circular: radius, fullsky: disregarded, rectangular: center +/- size in RA and DEC
    apply_roi: bool = True


@dataclass
class HierarchicalNuConfig:
    file_config: FileConfig = field(default_factory=lambda: FileConfig())
    parameter_config: ParameterConfig = field(default_factory=lambda: ParameterConfig())
    prior_config: PriorConfig = field(default_factory=lambda: PriorConfig())
    roi_config: ROIConfig = field(default_factory=lambda: ROIConfig())


# Load default config
hnu_config: HierarchicalNuConfig = OmegaConf.structured(HierarchicalNuConfig)

if _local_config_file.is_file():
    logger.info("local config found")
    _local_config = OmegaConf.load(_local_config_file)

    hnu_config: HierarchicalNuConfig = OmegaConf.merge(
        hnu_config,
        _local_config,
    )

elif _config_file.is_file():
    logger.info("global config found")
    _local_config = OmegaConf.load(_config_file)

    hnu_config: HierarchicalNuConfig = OmegaConf.merge(
        hnu_config,
        _local_config,
    )

else:
    # Prints should be converted to logger at some point
    logger.info("No config found, creating new global config from default")
    _config_path.mkdir(parents=True, exist_ok=True)

    with _config_file.open("w") as f:
        OmegaConf.save(config=hnu_config, f=f.name)
