import os
from omegaconf import OmegaConf
import pytest

from hierarchical_nu.model_check import ModelCheck
from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.utils.config import _local_config_file


def run_model_check(output_directory, random_seed):
    n_jobs = 1
    n_subjobs = 1

    output_file = os.path.join(output_directory, f"fit_sim_{random_seed}_test.h5")

    config = HierarchicalNuConfig.from_path(_local_config_file)
    model_check = ModelCheck(config=config)

    # Run
    model_check.parallel_run(
        n_jobs=n_jobs, n_subjobs=n_subjobs, seed=random_seed, adapt_delta=0.95
    )

    # Save
    model_check.save(output_file)

    # Load
    model_check = ModelCheck.load(output_file)

    return model_check


def test_short_run_r2021(output_directory, random_seed):
    # Edit configuration and save
    hnu_config = HierarchicalNuConfig.load_default()
    with _local_config_file.open("w") as f:
        OmegaConf.save(config=hnu_config, f=f.name)

    ModelCheck.initialise_env(output_dir=output_directory, config=hnu_config)

    model_check = run_model_check(output_directory, random_seed)

    # Check diagnostics
    ind_not_ok = model_check.diagnose()
    assert len(ind_not_ok) == 0

    # Check visualisations
    for p in [True, False]:
        fig, ax = model_check.compare(show_prior=p)

    # Clear config
    _local_config_file.unlink()


@pytest.mark.skip(reason="Detector model no longer maintained")
def test_short_run_icecube(output_directory, random_seed):
    # Edit configuration and save
    hnu_config = HierarchicalNuConfig.load_default()
    hnu_config["parameter_config"]["detector_model_type"] = "icecube"
    hnu_config["parameter_config"]["src_index"] = 2.0
    hnu_config["parameter_config"]["L"] = 1e47
    hnu_config["parameter_config"]["diff_norm"] = 1e-13
    hnu_config["parameter_config"]["Emin_det_tracks"] = 1e5
    hnu_config["parameter_config"]["obs_time"] = 1

    with _local_config_file.open("w") as f:
        OmegaConf.save(config=hnu_config, f=f.name)

    ModelCheck.initialise_env(output_dir=output_directory)

    model_check = run_model_check(output_directory, random_seed)

    # Check diagnostics
    ind_not_ok = model_check.diagnose()
    assert len(ind_not_ok) == 0

    # Check visualisations
    for p in [True, False]:
        fig, ax = model_check.compare(show_prior=p)

    # Clear config
    _local_config_file.unlink()
