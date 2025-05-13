import os
from abc import ABCMeta, abstractmethod

from hierarchical_nu.backend.stan_generator import StanFileGenerator
from ..source.source_info import SourceInfo

# To includes
STAN_PATH = os.path.dirname(__file__)

# To generated files
STAN_GEN_PATH = os.path.join(os.getcwd(), ".stan_files")


class StanInterface(SourceInfo, metaclass=ABCMeta):
    """
    Abstract base class for fleixble interface to
    Stan code generation.
    """

    def __init__(
        self,
        output_file,
        sources,
        event_types,
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

        self._event_types = event_types

        # Store number of event types in self._Net
        self._Net = len(self._event_types)

        self._check_output_dir()

    def _get_source_info(self):
        """
        Store some useful source info.
        """

        super().__init__(self._sources)

        num_params = 0
        num_params += 1 if self._fit_index else 0
        num_params += 1 if self._fit_beta else 0
        num_params += 1 if self._fit_Enorm else 0
        num_params += 1 if self._fit_eta else 0
        if not num_params <= 2:
            raise NotImplementedError("Can only use 2D interpolation")

        self._fit = [self._fit_index, self._fit_beta, self._fit_Enorm, self._fit_eta]

        if self.sources.diffuse:
            self._diff_spectrum = self.sources.diffuse_spectrum
            self._diff_frame = self.sources.diffuse.frame

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
        self._code_gen = StanFileGenerator(self._output_file)

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
    def event_types(self):
        return self._event_types
