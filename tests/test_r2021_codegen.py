from ftplib import parse150
import numpy as np
import os
import pytest
from cmdstanpy import CmdStanModel

from hierarchical_nu.detector.r2021 import R2021DetectorModel
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
    ParameterDef
)
from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

from hierarchical_nu.stan.interface import STAN_PATH
from hierarchical_nu.stan.interface import STAN_GEN_PATH

from icecube_tools.detector.r2021 import R2021IRF




class TestR2021():

    @pytest.fixture
    def sim_file(self, output_directory):
        #Generate code s.t. samples can be compared to icecube_tools

        file_name = os.path.join(output_directory, "r2021_sim")

        _ = R2021DetectorModel.generate_code(
            mode=DistributionMode.RNG,
            rewrite=True,
            gen_type="histogram",
            ereco_cuts=False,
            path=output_directory)

        with StanFileGenerator(file_name) as code_gen:

            with FunctionsContext():

                _ = Include("interpolation.stan")
                _ = Include("utils.stan")
                _ = Include("vMF.stan")
                _ = Include(R2021DetectorModel.RNG_FILENAME)

            with DataContext():

                etrue = ForwardVariableDef("true_energy", "real")
                phi = ForwardVariableDef("phi", "real")
                theta = ForwardVariableDef("theta", "real")

            with GeneratedQuantitiesContext():

                reco_energy = ForwardVariableDef("reco_energy", "real")
                reco_energy << StringExpression(["R2021EnergyResolution_rng(true_energy, [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]')"])
                pre_event = ForwardVariableDef("pre_event", "vector[4]")
                pre_event << StringExpression(["R2021AngularResolution_rng(true_energy, reco_energy, [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]')"])
                kappa = ForwardVariableDef("kappa", "real")
                reco_dir = ForwardVariableDef("reco_dir", "vector[3]")
                kappa << StringExpression(["pre_event[4]"])
                reco_dir << StringExpression(["pre_event[1:3]"])
        code_gen.generate_single_file()
        return code_gen.filename

    @pytest.fixture
    def model_file(self, output_directory):
        file_name = os.path.join(output_directory, "r2021_model")

        _ = R2021DetectorModel.generate_code(
            mode=DistributionMode.PDF,
            rewrite=True,
            gen_type="lognorm",
            path=output_directory
        )

        with StanFileGenerator(file_name) as code_gen:

            with FunctionsContext():

                _ = Include("interpolation.stan")
                _ = Include("utils.stan")
                _ = Include("vMF.stan")
                _ = Include(R2021DetectorModel.PDF_FILENAME)

            with DataContext():

                size = ForwardVariableDef("size", "int")
                ereco = ForwardArrayDef("reco_energy", "real", ["[", size, "]"])
                phi = ForwardVariableDef("phi", "real")
                theta = ForwardVariableDef("theta", "real")
                
            with ParametersContext():
                true_energy = ParameterDef("true_energy", "real", 2.25, 8.75)

            with TransformedParametersContext():
                lp = ForwardArrayDef("lp", "real", ["[", size, "]"])
                with ForLoopContext(1, size, "i") as i:
                    lp[i] << StringExpression(["R2021EnergyResolution(true_energy, reco_energy[i], [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]')"])
                
            with ModelContext():
                StringExpression(["target += log_sum_exp(lp)"])

        code_gen.generate_single_file()
        return code_gen.filename

    def test_file_generation_r2021(self, output_directory):

        R2021DetectorModel.generate_code(
            mode=DistributionMode.PDF,
            rewrite=False,
            gen_type="lognorm",
            path=output_directory
        )

        R2021DetectorModel.generate_code(
            mode=DistributionMode.RNG,
            rewrite=False,
            gen_type="histogram",
            path=output_directory
        )

    @pytest.fixture
    def test_samples(self, sim_file, random_seed):
        num_samples = 1000

        irf = R2021IRF.from_period("IC86_II")

        samples = np.zeros((irf.true_energy_values.size, irf.declination_bins.size-1, num_samples))

        stanc_options = {"include-paths": [STAN_PATH, os.path.dirname(sim_file)]}
        #model_file = os.path.join(model_dir, "r2021")
        # Compile model
        stan_model = CmdStanModel(
            stan_file=sim_file,
            stanc_options=stanc_options,
        )
        
        phi = 0
        theta = np.array([3*np.pi/4, np.pi/2, np.pi/4])
        etrue = irf.true_energy_values

        for c_e, e in enumerate(etrue):
            for c_d, t in enumerate(theta):


                data = {
                    "theta": t,
                    "phi": phi,
                    "true_energy": e
                }

                output = stan_model.sample(
                    data=data,
                    iter_sampling=num_samples,
                    chains=1,
                    seed=random_seed,
                )

                e_res = output.stan_variable("reco_energy")
                n, bins = np.histogram(e_res, irf.reco_energy_bins[c_e, c_d], density=True)

                samples[c_e, c_d, :] = e_res
                kappa = output.stan_variable("kappa")
                p = 0.5
                ang_err = np.rad2deg(np.sqrt((-2 / kappa) * np.log(1 - p)))

                assert np.all(ang_err >= 0.2)

                assert np.all(ang_err <= 20.)

                assert n == pytest.approx(irf.reco_energy[c_e, c_d].pdf(irf.reco_energy_bins[c_e, c_d][:-1]+0.01), abs=0.2)

        return samples

    def test_everything(self, test_samples, model_file, random_seed):
        # Generate model for fitting
        stanc_options = {"include-paths": [STAN_PATH, os.path.dirname(model_file)]}
        #model_file = os.path.join(model_dir, "r2021")
        # Compile model
        stan_model = CmdStanModel(
            stan_file=model_file,
            stanc_options=stanc_options,
        )

        irf = R2021IRF.from_period("IC86_II")
        phi = 0
        theta = np.array([3*np.pi/4])#, np.pi/2, np.pi/4])
        etrue = irf.true_energy_values
        size = 100
        num_samples=1000
        for c_e, e in enumerate(etrue[1:-1], 1):
            for c_d, t in enumerate(theta):


                data = {
                    "theta": t,
                    "phi": phi,
                    "reco_energy": np.random.choice(test_samples[c_e, c_d], size),
                    "size": size
                }

                output = stan_model.sample(
                    data=data,
                    iter_sampling=num_samples,
                    chains=1,
                    seed=random_seed,
                    inits={"true_energy": e}
                )

                true_energy = output.stan_variable("true_energy")
                # Tests, manual, have shown that this sometimes not the case!
                assert true_energy.min() < e

                assert true_energy.max() > e
      