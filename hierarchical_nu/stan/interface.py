import os
from abc import ABCMeta, abstractmethod

from hierarchical_nu.backend.stan_generator import StanFileGenerator

# To includes
STAN_PATH = os.path.dirname(__file__)

# To generated files
STAN_GEN_PATH = os.path.join(os.getcwd(), ".stan_files")


class StanInterface(object, metaclass=ABCMeta):
    """
    Abstract base class for fleixble interface to
    Stan code generation.
    """

    def __init__(
        self,
        output_file,
        sources,
        detector_model_type,
        includes=["interpolation.stan", "utils.stan"],
    ):
        """
        :param output_file: Name of output Stan file
        :param sources: Sources object
        :param event_types: Types of event to simulate
        :includes: Stan includes
        """

        self._includes = includes

        self._output_file = output_file

        self._sources = sources

        self._get_source_info()

        self._detector_model_type = detector_model_type

        self._event_types = self._detector_model_type.event_types

        self._code_gen = StanFileGenerator(output_file)

        self._check_output_dir()

    def _get_source_info(self):
        """
        Store some useful source info.
        """

        self._ps_spectrum = None

        self._diff_spectrum = None

        if self.sources.point_source:

            self._ps_spectrum = self.sources.point_source_spectrum

        if self.sources.diffuse:

            self._diff_spectrum = self.sources.diffuse_spectrum

        if self.sources.atmospheric:

            self._atmo_flux = self.sources.atmospheric_flux

    def _check_output_dir(self):
        """
        Creat Stan code dir if not existing.
        """

        if not os.path.isdir(STAN_GEN_PATH):

            os.makedirs(STAN_GEN_PATH)

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

    @property
    def detector_model_type(self):

        return self._detector_model_type
