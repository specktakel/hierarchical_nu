import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import numpy as np


from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH

_config_path = Path("~/.config/hierarchical_nu/").expanduser()
_local_config_path = Path(".")
_config_name = Path("hnu_config.yml")

_config_file = _config_path / _config_name
# Overwrite global config with local config
_local_config_file = _local_config_path / _config_name


@dataclass
class FileConfig:
    sim_filename: str = os.path.join(STAN_GEN_PATH, "sim_code.stan")
    fit_filename: str = os.path.join(STAN_GEN_PATH, "model_code.stan")
    include_paths: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.include_paths = [STAN_PATH]


@dataclass
class ParameterConfig:
    src_index: float = 2.3
    src_index_range: tuple = (1.0, 4.0)
    diff_index: float = 2.5
    diff_index_range: tuple = (1.0, 4.0)
    L: float = 5e46  # u.erg / u.s
    L_range: tuple = (0, 1e60)
    src_dec: float = 0.0  # u.deg
    src_ra: float = 90.0  # u.deg
    Enorm: float = 1e5  # u.GeV
    Emin: float = 5e4  # u.GeV
    Emax: float = 1e8  # u.GeV
    diff_norm: float = 5e-14  # 1 / (u.GeV * u.m**2 * u.s)
    z: float = 0.43  # cosmological redshift, dimensionless

    # If True, use same Emin_det for all
    # If False, use separate for tracks and cascades
    Emin_det_eq: bool = False

    Emin_det: float = 1e5  # u.GeV
    Emin_det_tracks: float = 1e5  # u.GeV
    Emin_det_cascades: float = 6e4  # u.GeV

    # Can be "icecube", "northern_tracks", "cascades", or "r2021"
    detector_model_type: str = "r2021"

    obs_time: float = 10  # years

    # Within-chain parallelisation
    nshards: int = 1
    threads_per_chain: int = nshards


@dataclass
class SinglePriorConfig:
    name: str = "LogNormalPrior"
    mu: float = 1.0
    sigma: float = 1.0


@dataclass
class PriorConfig:
    src_index: SinglePriorConfig = SinglePriorConfig(
        name="NormalPrior", mu=2.0, sigma=1.5
    )
    diff_index: SinglePriorConfig = SinglePriorConfig(
        name="NormalPrior", mu=2.5, sigma=1.5
    )
    L: SinglePriorConfig = SinglePriorConfig(name="LogNormalPrior", mu=1e52, sigma=10.0)
    diff_flux: SinglePriorConfig = SinglePriorConfig(
        name="LogNormalPrior", mu=1e-6, sigma=1.0
    )
    atmo_flux: SinglePriorConfig = SinglePriorConfig(
        name="LogNormalPrior", mu=1e-6, sigma=1.0
    )


@dataclass
class HierarchicalNuConfig:
    file_config: FileConfig = FileConfig()
    parameter_config: ParameterConfig = ParameterConfig()
    prior_config: PriorConfig = PriorConfig()


# Load default config
hnu_config: HierarchicalNuConfig = OmegaConf.structured(HierarchicalNuConfig)


if not _config_file.is_file() or not _local_config_file.is_file():
    # Prints should be converted to logger at some point
    print("No config found, creating new one")
    _config_path.mkdir(parents=True, exist_ok=True)

    with _config_file.open("w") as f:
        OmegaConf.save(config=hnu_config, f=f.name)

elif _local_config_file.is_file():
    print("local config found")
    _local_config = OmegaConf.load(_local_config_file)

    hnu_config: HierarchicalNuConfig = OmegaConf.merge(
        hnu_config,
        _local_config,
    )

elif _config_file.is_file():
    print("global config found")
    _local_config = OmegaConf.load(_config_file)

    hnu_config: HierarchicalNuConfig = OmegaConf.merge(
        hnu_config,
        _local_config,
    )
