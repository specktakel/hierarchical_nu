import tempfile
import yaml
from pathlib import Path
from omegaconf import OmegaConf
import pytest

from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.utils.config_parser import ConfigParser
from astropy.units.core import UnitConversionError
from astropy import units as u

u.imperial.enable()


def test_default_configuration():
    """
    Just load the default config.
    """

    default_config = HierarchicalNuConfig()


def test_configuration_write():
    """
    Save config locally and reload.
    """

    hnu_config = HierarchicalNuConfig()

    with tempfile.NamedTemporaryFile() as f:
        OmegaConf.save(config=hnu_config, f=f.name)

        loaded_config = OmegaConf.load(f.name)

    assert hnu_config == loaded_config


def test_user_config_merge():
    """
    Make partial user configs and merge.
    """

    hnu_config = HierarchicalNuConfig()

    user_configs = [
        {"parameter_config": {"src_index": [2.6], "L": ["1e47 GeV s-1"]}},
    ]

    for i, config in enumerate(user_configs):
        path = Path(f"config_{i}.yml")

        with path.open("w") as f:
            yaml.dump(stream=f, data=config, Dumper=yaml.SafeDumper)

        loaded_config = OmegaConf.load(path)

        hnu_config = OmegaConf.merge(hnu_config, loaded_config)

        path.unlink()

    assert hnu_config["parameter_config"]["src_index"][0] == 2.6

    assert hnu_config["parameter_config"]["L"][0] == "1e47 GeV s-1"


def test_wrong_unit():
    """
    Test that exceptions are raised when loading configs with erroneous units
    """

    hnu_config = HierarchicalNuConfig.load_default()
    hnu_config.parameter_config.L[0] = "1e47 GeV"
    hnu_config.prior_config.src_index.mu = "2.2 fur Ry"
    parser = ConfigParser(hnu_config)
    with pytest.raises(UnitConversionError):
        sources = parser.sources
    with pytest.raises(UnitConversionError):
        priors = parser.priors
