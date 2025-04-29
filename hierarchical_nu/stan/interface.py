import os
from abc import ABCMeta, abstractmethod

from hierarchical_nu.backend.stan_generator import StanFileGenerator
from hierarchical_nu.source.parameter import Parameter
from ..source.flux_model import LogParabolaSpectrum, PowerLawSpectrum, PGammaSpectrum, TwiceBrokenPowerLaw
from ..source.seyfert_model import SeyfertNuMuSpectrum

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

        self._ps_spectrum = None

        self._ps_frame = None

        self._diff_spectrum = None

        self._diff_frame = None

        self._shared_luminosity = True

        self._shared_src_index = True

        self._logparabola = False

        self._power_law = True

        self._pgamma = False

        self._fit_index = False
        self._fit_beta = False
        self._fit_Enorm = False

        self._shared_luminosity = False
        self._shared_src_index = False
        self._shared_eta = False
        if self.sources.point_source:
            self._ps_spectrum = self.sources.point_source_spectrum
            self._ps_frame = self.sources.point_source_frame
            self._logparabola = self._ps_spectrum == LogParabolaSpectrum
            self._power_law = (self._ps_spectrum == PowerLawSpectrum or self._ps_spectrum == TwiceBrokenPowerLaw)
            self._pgamma = self._ps_spectrum == PGammaSpectrum
            self._seyfert = self._ps_spectrum == SeyfertNuMuSpectrum

            try:
                Parameter.get_parameter("luminosity")
                self._shared_luminosity = True
            except ValueError:
                self._shared_luminosity = False

            if self._power_law or self._pgamma or self._logparabola:
                # Get source index and check if it is varied
                src_index = self._sources.point_source[0].parameters["index"]
                if not src_index.fixed and src_index.name == "src_index":
                    self._shared_src_index = True
                elif not src_index.fixed:
                    self._shared_src_index = False
            if self._logparabola or self._pgamma:
                beta = self._sources.point_source[0].parameters["beta"]
                E0_src = self._sources.point_source[0].parameters["norm_energy"]
                if not beta.fixed and beta.name == "beta_index":
                    self._shared_src_index = True
                elif not E0_src.fixed and E0_src.name == "E0_src":
                    self._shared_src_index = True
            if self._seyfert:
                eta = self._sources.point_source[0].parameters["eta"]
                # also use self._shared_src_index bc it is shared (lol) among source spectra
                if not eta.fixed and eta.name == "eta":
                    self._shared_src_index = True
                elif not eta.fixed:
                    self._shared_src_index = False
                self._fit_eta = not eta.fixed
            else:
                self._fit_eta = False

            try:
                self._fit_index = (
                    not self._sources.point_source[0]
                    .flux_model.parameters["index"]
                    .fixed
                )
            except KeyError:
                self._fit_index = False
            try:
                self._fit_beta = (
                    not self._sources.point_source[0]
                    .flux_model.parameters["beta"]
                    .fixed
                )
            except KeyError:
                self._fit_beta = False
            try:
                self._fit_Enorm = (
                    not self._sources.point_source[0]
                    .flux_model.parameters["norm_energy"]
                    .fixed
                )
            except KeyError:
                self._fit_Enorm = False

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
