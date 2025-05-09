from cmdstanpy import CmdStanModel
import numpy as np
import pytest
from os import path
from pathlib import Path

from hierarchical_nu.stan.interface import STAN_PATH

wd = path.abspath(path.dirname(__file__)) / Path("stan")


def test_interpolation():
    interpolation_model = CmdStanModel(
        stan_file=wd / Path("interpolation_test.stan"),
        stanc_options={"include-paths": [STAN_PATH]},
    )

    data = {
        "x": np.arange(1, 11),
        "y": np.arange(1, 11),
        "test": np.array([0, 1.0, 5.5, 10.0, 11.0]),
        "N": 10,
        "L": 5,
    }
    samples = interpolation_model.sample(
        data=data,
        iter_sampling=1,
        fixed_param=True,
        chains=1,
    )
    # output is truncated at lowest and highest value
    assert samples.stan_variable("interpolated").squeeze() == pytest.approx(
        np.array([1.0, 1.0, 5.5, 10.0, 10.0])
    )
    assert samples.stan_variable("log_interpolated").squeeze() == pytest.approx(
        np.exp(np.array([1.0, 1.0, 5.5, 10.0, 10.0])), rel=1e-3
    )
    assert samples.stan_variable("interpolated_2bins").squeeze() == pytest.approx(
        np.array([1.0, 1.0, 5.5, 10.0, 10.0])
    )


def test_binary_search():

    search_model = CmdStanModel(
        stan_file=wd / Path("binary_test.stan"),
        stanc_options={"include-paths": [STAN_PATH]},
    )

    data = {
        "x": np.arange(1, 11),
        "test": np.array([0.0, 1.0, 9.5, 10.0, 11.0]),
        "N": 10,
        "L": 5,
    }

    samples = search_model.sample(data=data, iter_sampling=1, fixed_param=True, chains=1)
    assert samples.stan_variable("search").squeeze() == pytest.approx(
        np.array([0.0, 1.0, 9.0, 9.0, 11.0])
    )


def test_angles():

    geometry_model = CmdStanModel(
        stan_file=wd / Path("utils_test.stan"),
        stanc_options={"include-paths": [STAN_PATH]},
    )

    data = {
        "N": 4,
        "F": np.arange(1, 5),
        "eps": np.array([1.0, 0.0, 1.0, 1.0]),
        "N_vec": 3,
        "omega": np.array(
            [[1.0, 0.0, 0.0], [1 / np.sqrt(2), 0.0, -1 / np.sqrt(2)], [0.0, 0.0, -1.0]]
        ),
    }
    samples = geometry_model.sample(
        data=data,
        iter_sampling=1,
        fixed_param=True,
        chains=1,
    )

    assert samples.stan_variable("weights").squeeze() == pytest.approx(
        data["F"] * data["eps"] / np.sum(data["F"] * data["eps"]), rel=1e-4
    )
    assert pytest.approx(np.sum(samples.stan_variable("weights"))) == 1.0
    assert samples.stan_variable("Nex").squeeze() == pytest.approx(
        data["F"] * data["eps"], rel=1e-4
    )
    assert samples.stan_variable("dec").squeeze() == pytest.approx(
        np.arcsin(data["omega"][:, 2]), rel=1e-4
    )
    assert samples.stan_variable("zenith").squeeze() == pytest.approx(
        np.arccos(-data["omega"][:, 2]), rel=1e-4
    )
    assert samples.stan_variable("angle").squeeze() == pytest.approx(
        np.array([0.0, np.pi / 4, np.pi / 2]), rel=1e-4
    )
