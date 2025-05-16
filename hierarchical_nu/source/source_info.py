from .parameter import Parameter
from .flux_model import (
    PowerLawSpectrum,
    TwiceBrokenPowerLaw,
    PGammaSpectrum,
    LogParabolaSpectrum,
)
from .seyfert_model import SeyfertNuMuSpectrum


class SourceInfo:
    """
    Class to organise all source- and parameter-related attributes for simulations and fits
    """

    def __init__(self, sources):

        self._sources = sources
        self._sources.organise()

        self._bg = False
        if self._sources.background:
            self._bg = True

        self._shared_luminosity = False
        self._shared_src_index = False
        try:
            Parameter.get_parameter("luminosity")
            self._shared_luminosity = True
        except ValueError:
            pass
        try:
            # Hijack shared_luminosity for Seyferts, where the pressure ratio acts as luminosity
            Parameter.get_parameter("pressure_ratio")
            self._shared_luminosity = True
        except:
            pass
        if self._sources.point_source:

            try:
                ang_sys = Parameter.get_parameter("ang_sys_add")
                self._ang_sys = True
                self._fit_ang_sys = not ang_sys.fixed
            except ValueError:
                self._ang_sys = False
                self._fit_ang_sys = False

            self._ps_spectrum = self.sources.point_source_spectrum
            self._ps_frame = self.sources.point_source_frame
            self._logparabola = self._ps_spectrum == LogParabolaSpectrum
            self._power_law = self._sources.point_source_spectrum in [
                PowerLawSpectrum,
                TwiceBrokenPowerLaw,
            ]
            self._pgamma = self._ps_spectrum == PGammaSpectrum
            self._seyfert = self._ps_spectrum == SeyfertNuMuSpectrum

            if self._power_law or self._pgamma or self._logparabola:
                index = self._sources.point_source[0].parameters["index"]
                self._fit_index = not index.fixed
                if not index.fixed and index.name == "src_index":
                    self._shared_src_index = True
                elif not index.fixed:
                    self._shared_src_index = False
            else:
                self._fit_index = False

            if self._logparabola or self._pgamma:
                beta = self._sources.point_source[0].parameters["beta"]
                E0_src = self._sources.point_source[0].parameters["norm_energy"]
                if not beta.fixed and beta.name == "beta_index":
                    self._shared_src_index = True
                elif not E0_src.fixed and E0_src.name == "E0_src":
                    self._shared_src_index = True
                self._fit_beta = not beta.fixed
                self._fit_Enorm = not E0_src.fixed
            else:
                self._fit_beta = False
                self._fit_Enorm = False

            if self._seyfert:
                eta = self._sources.point_source[0].parameters["eta"]
                self._fit_eta = not eta.fixed
                if not eta.fixed and eta.name == "eta":
                    self._shared_src_index = True
            else:
                self._fit_eta = False
            try:
                Nex_src = Parameter.get_parameter("Nex_src")
                self._fit_nex = True
            except ValueError:
                self._fit_nex = False
        else:
            self._shared_src_index = False
            self._fit_index = False
            self._fit_beta = False
            self._fit_Enorm = False
            self._fit_eta = False
            self._power_law = False
            self._logparabola = False
            self._pgamma = False
            self._seyfert = False
            self._ang_sys = False
            self._fit_ang_sys = False
            self._ps_frame = None
            self._ps_spectrum = None
            self._fit_nex = False

        if self.sources.diffuse:
            self._diff_frame = self.sources.diffuse.frame
            self._diff_spectrum = self.sources.diffuse_spectrum
        else:
            self._diff_spectrum = None
            self._diff_frame = None

    @property
    def sources(self):
        return self._sources
