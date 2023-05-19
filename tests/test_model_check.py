import os
from omegaconf import OmegaConf

from hierarchical_nu.model_check import ModelCheck
from hierarchical_nu.utils.config import hnu_config
from hierarchical_nu.utils.config import _local_config_file


def test_initialisation(output_directory):
    ModelCheck.initialise_env(output_dir=output_directory)


def run_model_check(output_directory, random_seed):
    n_jobs = 1
    n_subjobs = 1

    output_file = os.path.join(output_directory, f"fit_sim_{random_seed}_test.h5")

    model_check = ModelCheck()

    # Run
    model_check.parallel_run(n_jobs=n_jobs, n_subjobs=n_subjobs, seed=random_seed)

    # Save
    model_check.save(output_file)

    # Load
    file_list = [output_file]
    model_check = ModelCheck.load(file_list)

    return model_check


def test_short_run_r2021(output_directory, random_seed):
    # Edit configuration and save
    hnu_config["parameter_config"]["detector_model_type"] = "r2021"

    with _local_config_file.open("w") as f:
        OmegaConf.save(config=hnu_config, f=f.name)

    model_check = run_model_check(output_directory, random_seed)

    # Check diagnostics
    ind_not_ok = model_check.diagnose()
    assert len(ind_not_ok) == 0

    # Check visualisations
    for p in [True, False]:
        fig, ax = model_check.compare(show_prior=p)


def test_short_run_icecube(output_directory, random_seed):
    # Edit configuration and save
    hnu_config["parameter_config"]["detector_model_type"] = "icecube"

    with _local_config_file.open("w") as f:
        OmegaConf.save(config=hnu_config, f=f.name)

    model_check = run_model_check(output_directory, random_seed)

    # Check diagnostics
    ind_not_ok = model_check.diagnose()
    assert len(ind_not_ok) == 0

    # Check visualisations
    for p in [True, False]:
        fig, ax = model_check.compare(show_prior=p)
