import numpy as np
import os
import pytest
from cmdstanpy import CmdStanModel

from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.backend.stan_generator import (
    GeneratedQuantitiesContext,
    DataContext,
    FunctionsContext,
    Include,
    ForLoopContext,
    StanFileGenerator,
)
from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
)
from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.stan.interface import STAN_PATH


def test_file_generation_northern_tracks(output_directory):

    file_name = os.path.join(output_directory, "northern_tracks")

    with StanFileGenerator(file_name) as code_gen:

        _ = NorthernTracksDetectorModel(mode=DistributionMode.PDF)

        _ = NorthernTracksDetectorModel(mode=DistributionMode.RNG)

        code_gen.generate_files()


def generate_distribution_test_code(output_directory):

    file_name = os.path.join(output_directory, "nt_distributions")

    e_true_name = "e_trues"
    e_reco_name = "e_recos"
    true_dir_name = "true_dirs"
    reco_zenith_name = "reco_zeniths"

    with StanFileGenerator(file_name) as code_gen:

        with FunctionsContext():

            _ = Include("interpolation.stan")
            _ = Include("utils.stan")
            _ = Include("vMF.stan")

        with DataContext():

            array_length = ForwardVariableDef("n", "int")
            array_length_str = ["[", array_length, "]"]

            e_trues = ForwardArrayDef(e_true_name, "real", array_length_str)
            e_recos = ForwardArrayDef(e_reco_name, "real", array_length_str)
            true_dirs = ForwardArrayDef(true_dir_name, "vector[3]", array_length_str)
            reco_zenith = ForwardArrayDef(reco_zenith_name, "real", array_length_str)

        with GeneratedQuantitiesContext():

            ntd = NorthernTracksDetectorModel()

            array_length_2d_str = ["[", array_length, ",", array_length, "]"]
            e_res_result = ForwardArrayDef("e_res", "real", array_length_2d_str)
            eff_area_result = ForwardArrayDef("eff_area", "real", array_length_2d_str)
            ang_res_result = ForwardArrayDef("ang_res", "real", array_length_2d_str)

            reco_dir_ang_res = ForwardVariableDef("reco_dir", "vector[3]")
            true_dir_ang_res = ForwardVariableDef("true_dir", "vector[3]")
            true_dir_ang_res << StringExpression("[sin(pi()/2), 0, cos(pi()/2)]'")

            with ForLoopContext(1, array_length, "i") as i:

                with ForLoopContext(1, array_length, "j") as j:

                    eff_area_result[i][j] << ntd.effective_area(
                        e_trues[i], true_dirs[j]
                    )
                    e_res_result[i][j] << ntd.energy_resolution(e_trues[i], e_recos[j])
                    reco_dir_ang_res << StringExpression(
                        ["[sin(", reco_zenith[j], "), 0, cos(", reco_zenith[j], ")]'"]
                    )
                    ang_res_result[i][j] << ntd.angular_resolution(
                        e_trues[i], true_dir_ang_res, reco_dir_ang_res
                    )

    code_gen.generate_single_file()

    return code_gen.filename


def test_distributions_northern_tracks(output_directory, random_seed):

    model_file = generate_distribution_test_code(output_directory)

    stanc_options = {"include-paths": [STAN_PATH]}

    # Compile model
    stan_model = CmdStanModel(
        stan_file=model_file,
        stanc_options=stanc_options,
    )

    n = 100
    e_reco = np.logspace(2, 9, n)
    e_true = np.logspace(2, 7, n)
    reco_zeniths = np.radians(np.linspace(85, 95, n))
    thetas = np.pi - np.radians(np.linspace(85, 180, n, endpoint=False))
    true_dir = np.asarray([np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)]).T

    e_true_name = "e_trues"
    e_reco_name = "e_recos"
    true_dir_name = "true_dirs"
    reco_zenith_name = "reco_zeniths"

    data = {
        e_true_name: e_reco,
        e_reco_name: e_true,
        true_dir_name: true_dir,
        reco_zenith_name: reco_zeniths,
        "n": n,
    }

    output = stan_model.sample(
        data=data,
        iter_sampling=1,
        chains=1,
        fixed_param=True,
        seed=random_seed,
    )

    e_res = output.stan_variable("e_res")

    ang_res = output.stan_variable("ang_res")

    eff_area = output.stan_variable("eff_area")

    assert np.mean(e_res) == pytest.approx(-13.9071762520618, 0.1)

    assert np.mean(ang_res) == pytest.approx(-43.526770393359996, 0.1)

    assert np.max(eff_area) == pytest.approx(28718.9)

    assert np.min(eff_area) == 0.0


def generate_rv_test_code(output_directory):

    rng_file_name = os.path.join(output_directory, "nt_rng")
    pdf_file_name = os.path.join(output_directory, "nt_pdf")

    with StanFileGenerator(rng_file_name) as code_gen_rng:

        with FunctionsContext():

            Include("interpolation.stan")
            Include("utils.stan")
            Include("vMF.stan")

            with DataContext():

                true_energy = ForwardVariableDef("true_energy", "real")
                true_dir = ForwardVariableDef("true_dir", "vector[3]")

            with GeneratedQuantitiesContext():

                ntd_rng = NorthernTracksDetectorModel(mode=DistributionMode.RNG)
                rec_energy = ForwardVariableDef("rec_energy", "real")
                rec_dir = ForwardVariableDef("rec_dir", "vector[3]")

                rec_energy << ntd_rng.energy_resolution(true_energy)
                rec_dir << ntd_rng.angular_resolution(true_energy, true_dir)

        code_gen_rng.generate_single_file()

    with StanFileGenerator(pdf_file_name) as code_gen_pdf:

        with FunctionsContext():

            Include("interpolation.stan")
            Include("utils.stan")
            Include("vMF.stan")

        with DataContext():

            true_energy = ForwardVariableDef("true_energy", "real")
            e_recos = ForwardArrayDef("e_recos", "real", ["[100]"])

        with GeneratedQuantitiesContext():

            e_res_result = ForwardArrayDef("e_res", "real", ["[100]"])
            ntd_pdf = NorthernTracksDetectorModel(mode=DistributionMode.PDF)

            with ForLoopContext(1, 100, "i") as i:
                e_res_result[i] << ntd_pdf.energy_resolution(true_energy, e_recos[i])

        code_gen_pdf.generate_single_file()

    return code_gen_rng.filename, code_gen_pdf.filename


def test_rv_generation(output_directory, random_seed):

    # Get Stan files
    rng_file_name, pdf_file_name = generate_rv_test_code(output_directory)

    # Compile Stan code
    stanc_options = {"include-paths": [STAN_PATH]}

    rng_stan_model = CmdStanModel(
        stan_file=rng_file_name,
        stanc_options=stanc_options,
    )

    pdf_stan_model = CmdStanModel(
        stan_file=pdf_file_name,
        stanc_options=stanc_options,
    )

    zenith = np.pi / 2

    # Check rng matches pdf for true energy of 1e4 GeV
    data = {
        "true_energy": 1e4,
        "true_dir": [np.sin(zenith), 0, np.cos(zenith)],
        "e_recos": np.logspace(2, 8, 100),
    }

    output_rng = rng_stan_model.sample(
        data=data,
        iter_sampling=10000,
        chains=1,
        fixed_param=True,
        seed=random_seed,
    )
    output_pdf = pdf_stan_model.sample(
        data=data,
        iter_sampling=1,
        chains=1,
        fixed_param=True,
        seed=random_seed,
    )

    reco_energy_samples = output_rng.stan_variable("rec_energy")
    reco_dir_samples = output_rng.stan_variable("rec_dir")

    reco_energy_pdf = np.exp(output_pdf.stan_variable("e_res")[0])

    E_bins = np.log10(data["e_recos"])

    E_hist, _ = np.histogram(
        reco_energy_samples,
        bins=E_bins,
        density=True,
    )

    reco_zenith = np.degrees(np.arccos(reco_dir_samples[:, 2]))

    assert max(E_hist) == pytest.approx(max(reco_energy_pdf), 0.1)

    assert np.mean(reco_zenith) == pytest.approx(90, 0.1)

    # Check rng matches pdf for true energy of 1e6 GeV
    data = {
        "true_energy": 1e6,
        "true_dir": np.asarray([np.sin(zenith), 0, np.cos(zenith)]).T,
        "e_recos": np.logspace(2, 8, 100),
    }

    output_rng = rng_stan_model.sample(
        data=data,
        iter_sampling=10000,
        chains=1,
        fixed_param=True,
        seed=random_seed,
    )

    output_pdf = pdf_stan_model.sample(
        data=data,
        iter_sampling=1,
        chains=1,
        fixed_param=True,
        seed=random_seed,
    )

    reco_energy_samples = output_rng.stan_variable("rec_energy")
    reco_dir_samples = output_rng.stan_variable("rec_dir")

    reco_energy_pdf = np.exp(output_pdf.stan_variable("e_res")[0])

    E_bins = np.log10(data["e_recos"])

    E_hist, _ = np.histogram(
        reco_energy_samples,
        bins=E_bins,
        density=True,
    )

    reco_zenith = np.degrees(np.arccos(reco_dir_samples[:, 2]))

    assert max(E_hist) == pytest.approx(max(reco_energy_pdf), 0.1)

    assert np.mean(reco_zenith) == pytest.approx(90, 0.1)
