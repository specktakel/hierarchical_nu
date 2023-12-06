from pathlib import Path
import pytest

from hierarchical_nu.utils.cache import Cache

# from hierarchical_nu.utils.cache import Cache
# from hierarchical_nu.backend.stan_generator import StanGenerator
# from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
# from hierarchical_nu.detector.cascades import CascadesDetectorModel


@pytest.fixture(scope="session", autouse=True)
def cache_setup():
    cache_dir = Path(__file__).parent.resolve() / "cache_files"
    Cache.set_cache_dir(cache_dir)


@pytest.fixture(scope="session")
def output_directory(tmpdir_factory):
    directory = tmpdir_factory.mktemp("output")

    return directory


@pytest.fixture(scope="session")
def random_seed():
    seed = 100

    return seed
