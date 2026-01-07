from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.backend.parameterizations import DistributionMode
from hierarchical_nu.backend import FunctionsContext, StanGenerator
import pytest


@pytest.mark.skip(reason="Detector model no longer maintained")
def test_file_generation_cascades(output_directory):
    _ = CascadesDetectorModel.generate_code(
        mode=DistributionMode.PDF, path=output_directory, rewrite=True
    )

    _ = CascadesDetectorModel.generate_code(
        mode=DistributionMode.RNG, path=output_directory, rewrite=True
    )

    with StanGenerator() as gc:
        with FunctionsContext():
            cas_pdf = CascadesDetectorModel()
            cas_pdf.generate_pdf_function_code()

            cas_rng = CascadesDetectorModel(DistributionMode.RNG)
            cas_rng.generate_rng_function_code()
