import os

from hierarchical_nu.model_check import ModelCheck


def test_initialisation(output_directory):

    ModelCheck.initialise_env(output_dir=output_directory)


def test_short_run(output_directory, random_seed):

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

    # Check visualisations
    for p in [True, False]:

        fig, ax = model_check.compare(show_prior=p)
