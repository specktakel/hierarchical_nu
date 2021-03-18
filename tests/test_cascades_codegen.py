import os

from hierarchical_nu.backend.stan_generator import StanFileGenerator
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.backend.parameterizations import DistributionMode


def test_file_generation_cascades(output_directory):

    file_name = os.path.join(output_directory, "cascades")

    with StanFileGenerator(file_name) as cg:

        _ = CascadesDetectorModel(mode=DistributionMode.PDF)

        _ = CascadesDetectorModel(mode=DistributionMode.RNG)

        cg.generate_files()
