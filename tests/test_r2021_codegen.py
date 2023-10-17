from ftplib import parse150
import numpy as np
import os
import pytest
from cmdstanpy import CmdStanModel

from icecube_tools.utils.vMF import get_theta_p

from hierarchical_nu.detector.r2021 import IC86_IIDetectorModel
from hierarchical_nu.backend.stan_generator import (
    GeneratedQuantitiesContext,
    DataContext,
    ModelContext,
    FunctionsContext,
    Include,
    ForLoopContext,
    ParametersContext,
    StanFileGenerator,
    FunctionCall,
    TransformedParametersContext,
)
from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ParameterDef,
)
from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.stan.interface import STAN_PATH
from hierarchical_nu.stan.interface import STAN_GEN_PATH

from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.point_source_likelihood.energy_likelihood import (
    MarginalisedIntegratedEnergyLikelihood,
)


class TestR2021:
    @pytest.fixture
    def sim_file(self, output_directory):
        # Generate code s.t. samples can be compared to icecube_tools

        file_name = os.path.join(output_directory, "r2021_sim")

        _ = IC86_IIDetectorModel.generate_code(
            mode=DistributionMode.RNG,
            rewrite=True,
            ereco_cuts=False,
            path=output_directory,
        )

        with StanFileGenerator(file_name) as code_gen:
            with FunctionsContext():
                _ = Include("interpolation.stan")
                _ = Include("utils.stan")
                _ = Include("vMF.stan")
                _ = Include(IC86_IIDetectorModel.RNG_FILENAME)
                rng = IC86_IIDetectorModel(DistributionMode.RNG)
                rng.generate_rng_function_code()

            with DataContext():
                etrue = ForwardVariableDef("true_energy", "real")
                phi = ForwardVariableDef("phi", "real")
                theta = ForwardVariableDef("theta", "real")

            with GeneratedQuantitiesContext():
                rng_return = ForwardVariableDef("rng_return", "vector[5]")
                reco_energy = ForwardVariableDef("reco_energy", "real")
                kappa = ForwardVariableDef("kappa", "real")
                reco_dir = ForwardVariableDef("reco_dir", "vector[3]")

                rng_return << StringExpression(
                    [
                        "IC86_II_rng(true_energy, [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]')"
                    ]
                )
                reco_energy << rng_return[1]
                reco_dir << rng_return[2:4]
                kappa << rng_return[5]

        code_gen.generate_single_file()
        return code_gen.filename

    @pytest.fixture
    def model_file(self, output_directory):
        file_name = os.path.join(output_directory, "r2021_model")

        _ = IC86_IIDetectorModel.generate_code(
            mode=DistributionMode.PDF,
            rewrite=True,
            path=output_directory,
        )

        with StanFileGenerator(file_name) as code_gen:
            with FunctionsContext():
                _ = Include("interpolation.stan")
                _ = Include("utils.stan")
                _ = Include("vMF.stan")
                _ = Include(IC86_IIDetectorModel.PDF_FILENAME)

            with DataContext():
                size = ForwardVariableDef("size", "int")
                ereco = ForwardArrayDef("reco_energy", "real", ["[", size, "]"])
                phi = ForwardVariableDef("phi", "real")
                theta = ForwardVariableDef("theta", "real")

            with ParametersContext():
                true_energy = ParameterDef("true_energy", "real", 2.25, 8.0)

            with TransformedParametersContext():
                lp = ForwardArrayDef("lp", "real", ["[", size, "]"])
                with ForLoopContext(1, size, "i") as i:
                    lp[i] << StringExpression(
                        [
                            "IC86_IIEnergyResolution(true_energy, reco_energy[i], [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]')"
                        ]
                    )

            with ModelContext():
                StringExpression(["target += log_sum_exp(lp)"])

        code_gen.generate_single_file()
        return code_gen.filename

    def test_file_generation_r2021(self, output_directory):
        IC86_IIDetectorModel.generate_code(
            mode=DistributionMode.PDF,
            rewrite=False,
            path=output_directory,
        )

        IC86_IIDetectorModel.generate_code(
            mode=DistributionMode.RNG,
            rewrite=False,
            path=output_directory,
        )

    @pytest.fixture
    def test_samples(self, sim_file, random_seed):
        num_samples = 1000

        irf = R2021IRF.from_period("IC86_II")

        samples = np.zeros(
            (irf.true_energy_values.size, irf.declination_bins.size - 1, num_samples)
        )

        stanc_options = {"include-paths": [STAN_PATH, os.path.dirname(sim_file)]}
        # model_file = os.path.join(model_dir, "r2021")
        # Compile model
        stan_model = CmdStanModel(
            stan_file=sim_file,
            stanc_options=stanc_options,
        )

        phi = 0
        theta = np.array([3 * np.pi / 4, np.pi / 2, np.pi / 4])
        etrue = np.power(10, irf.true_energy_values)

        for c_e, e in enumerate(etrue):
            for c_d, t in enumerate(theta):
                data = {"theta": t, "phi": phi, "true_energy": e}

                output = stan_model.sample(
                    data=data,
                    iter_sampling=num_samples,
                    chains=1,
                    seed=random_seed,
                )

                e_res = np.log10(output.stan_variable("reco_energy"))
                n, bins = np.histogram(
                    e_res, irf.reco_energy_bins[c_e, c_d], density=True
                )

                samples[c_e, c_d, :] = e_res
                kappa = output.stan_variable("kappa")
                p = 0.5
                ang_err = get_theta_p(kappa, p=p)

                assert np.all(ang_err >= 0.2)

                assert np.all(ang_err <= 20.0)

                assert n == pytest.approx(
                    irf.reco_energy[c_e, c_d].pdf(
                        irf.reco_energy_bins[c_e, c_d][:-1] + 0.01
                    ),
                    abs=0.3,
                )

        return samples

    def test_everything(self, test_samples, model_file, random_seed):
        # Generate model for fitting
        stanc_options = {"include-paths": [STAN_PATH, os.path.dirname(model_file)]}
        # model_file = os.path.join(model_dir, "r2021")
        # Compile model
        stan_model = CmdStanModel(
            stan_file=model_file,
            stanc_options=stanc_options,
        )

        irf = R2021IRF.from_period("IC86_II")
        phi = 0
        theta = np.array([3 * np.pi / 4])  # , np.pi/2, np.pi/4])
        etrue = irf.true_energy_values[:-2]
        size = 100
        num_samples = 1000
        for c_e, e in enumerate(etrue[1:-1], 1):
            for c_d, t in enumerate(theta):
                data = {
                    "theta": t,
                    "phi": phi,
                    "reco_energy": np.random.choice(test_samples[c_e, c_d], size),
                    "size": size,
                }

                output = stan_model.sample(
                    data=data,
                    iter_sampling=num_samples,
                    chains=1,
                    seed=random_seed,
                    inits={"true_energy": e},
                )

                true_energy = output.stan_variable("true_energy")
                # Tests, manual, have shown that this sometimes not the case!
                assert true_energy.min() < e

                assert true_energy.max() > e

    def test_ereco_cuts(self, output_directory):
        # Test that the ereco cuts are applied correctly

        file_name = os.path.join(output_directory, "r2021_sim")
        _ = IC86_IIDetectorModel.generate_code(
            mode=DistributionMode.RNG,
            rewrite=True,
            ereco_cuts=True,
            path=output_directory,
        )

        aeff = EffectiveArea.from_dataset("20210126", "IC86_II")
        eres = MarginalisedIntegratedEnergyLikelihood("IC86_II", np.linspace(1, 9, 25))
        cosz_bins = aeff.cos_zenith_bins
        dec = np.sort(np.arcsin((cosz_bins[:-1] + cosz_bins[1:]) / 2))
        theta_vals = np.pi / 2 - dec

        with StanFileGenerator(file_name) as code_gen:
            with FunctionsContext():
                _ = Include("interpolation.stan")
                _ = Include("utils.stan")
                _ = Include("vMF.stan")
                _ = Include(IC86_IIDetectorModel.RNG_FILENAME)
                ic86_rng = IC86_IIDetectorModel(DistributionMode.RNG)
                ic86_rng.generate_rng_function_code()

            with DataContext():
                etrue = ForwardVariableDef("true_energy", "real")
                phi = ForwardVariableDef("phi", "real")
                theta = ForwardVariableDef("theta", "real")

            with GeneratedQuantitiesContext():
                reco_energy = ForwardArrayDef("reco_energy", "real", ["[1000]"])
                with ForLoopContext(1, 1000, "j") as j:
                    reco_energy[j] << StringExpression(
                        [
                            "IC86_IIEnergyResolution_rng(true_energy, [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]')"
                        ]
                    )

        code_gen.generate_single_file()

        stanc_options = {"include-paths": [STAN_PATH, output_directory]}

        model = CmdStanModel(
            stan_file=code_gen.filename,
            stanc_options=stanc_options,
        )

        for c, (t, d) in enumerate(zip(theta_vals, dec)):
            samples = model.sample(
                data={
                    "theta": t,
                    "phi": 0.0,
                    "true_energy": 5.4,
                },
                fixed_param=True,
                chains=1,
                iter_sampling=1,
            )

            ereco = samples.stan_variable("reco_energy")[0]

            assert ereco.min() > eres._ereco_limits[c, 0]
