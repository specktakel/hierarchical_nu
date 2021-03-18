import numpy as np
import os
from cmdstanpy import CmdStanModel

from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.backend.stan_generator import (
    StanGenerator,
    GeneratedQuantitiesContext,
    DataContext,
    FunctionsContext,
    Include,
    ForLoopContext,
    StanFileGenerator,
    WhileLoopContext,
    TransformedDataContext,
)
from hierarchical_nu.backend.operations import FunctionCall
from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
)
from hierarchical_nu.backend.parameterizations import LogParameterization
from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.stan_interface import STAN_PATH


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

    stanc_options = {"include_paths": [STAN_PATH]}

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

    assert np.mean(e_res) == -13.9071762520618

    assert np.mean(ang_res) == -43.526770393359996
