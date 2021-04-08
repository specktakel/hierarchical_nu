import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from omegaconf import OmegaConf


from ..stan_interface import STAN_PATH

_config_path = Path("~/.config/hierarchical_nu/").expanduser()

_config_name = Path("hnu_config.yml")

_config_file = _config_path / _config_name


@dataclass
class FileConfig:

    atmo_sim_filename: str = os.path.join(STAN_PATH, "atmo_gen.stan")
    main_sim_filename: str = os.path.join(STAN_PATH, "sim_code.stan")
    fit_filename: str = os.path.join(STAN_PATH, "model_code.stan")
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
    Enorm: float = 1e5  # u.GeV
    Emin: float = 5e4  # u.GeV
    Emax: float = 1e8  # u.GeV
    diff_norm: float = 5e14  # 1 / (u.GeV * u.m**2 * u.s)

    # If True, use same Emin_det for all
    # If False, use separate for tracks and cascades
    Emin_det_eq: bool = False

    Emin_det: float = 1e5  # u.GeV
    Emin_det_tracks: float = 1e5  # u.GeV
    Emin_det_cascades: float = 6e4  # u.GeV

    # Can be "icecube", "northern_tracks" or "cascades"
    detector_model_type: str = "icecube"


@dataclass
class HierarchicalNuConfig:

    file_config: FileConfig = FileConfig()
    parameter_config: ParameterConfig = ParameterConfig()


# Load default config
hnu_config: HierarchicalNuConfig = OmegaConf.structured(HierarchicalNuConfig)

# Merge user config
if _config_file.is_file():

    _local_config = OmegaConf.load(_config_file)

    hnu_config: HierarchicalNuConfig = OmegaConf.merge(
        hnu_config,
        _local_config,
    )

# Write defaults
else:

    # Make directory if needed
    _config_path.mkdir(parents=True, exist_ok=True)

    with _config_file.open("w") as f:

        OmegaConf.save(config=hnu_config, f=f.name)
