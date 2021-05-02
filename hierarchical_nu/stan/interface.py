import os
from abc import ABCMeta, abstractmethod

from hierarchical_nu.backend.stan_generator import StanFileGenerator

from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.events import TRACKS, CASCADES


STAN_PATH = os.path.join(os.path.dirname(__file__), "stan")


class StanInterface(object, metaclass=ABCMeta):
    """
    Abstract base class for fleixble interface to
    Stan code generation.
    """

    def __init__(
        self,
        output_file,
        sources,
        includes=["interpolation.stan", "utils.stan"],
    ):
        """
        :param output_file: Name of output Stan file
        :param sources: Sources object
        """

        self._includes = includes

        self._output_file = output_file

        self._sources = sources

        self._get_source_info()

        self._code_gen = StanFileGenerator(output_file)

    def _get_source_info(self):
        """
        Store some useful source info.
        """

        self._ps_spectrum = None

        self._diff_spectrum = None

        if self._sources.point_source:

            self._ps_spectrum = self.sources.point_source_spectrum

        if self._sources.diffuse:

            self._diff_spectrum = self.sources.diffuse_spectrum

    @abstractmethod
    def _functions(self):

        pass

    @abstractmethod
    def _data(self):

        pass

    def _transformed_data(self):

        pass

    def _parameters(self):

        pass

    def _transformed_parameters(self):

        pass

    def _model(self):

        pass

    def _generated_quantities(self):

        pass

    def generate(self):

        with self._code_gen:

            self._functions()

            self._data()

            self._transformed_data()

            self._parameters()

            self._transformed_parameters()

            self._model()

            self._generated_quantities()

        self._code_gen.generate_single_file()

        return self._code_gen.filename

    @property
    def includes(self):

        return self._includes

    @property
    def output_file(self):

        return self._output_file

    @property
    def sources(self):

        return self._sources
