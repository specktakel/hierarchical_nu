from astropy import units as u

from hierarchical_nu.stan.interface import StanInterface

from hierarchical_nu.backend.stan_generator import (
    StanFileGenerator,
    FunctionsContext,
    Include,
    DataContext,
    TransformedDataContext,
    ParametersContext,
    TransformedParametersContext,
    GeneratedQuantitiesContext,
    ForLoopContext,
    IfBlockContext,
    ElseIfBlockContext,
    ElseBlockContext,
    WhileLoopContext,
    ModelContext,
    FunctionCall,
)

from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ParameterDef,
    ParameterVectorDef,
)

from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode


class StanFitInterface(StanInterface):
    """
    For generating Stan fit code.
    """

    def __init__(
        self,
        output_file,
        sources,
        detector_model_type,
        includes=["interpolation.stan", "utils.stan", "vMF.stan"],
        theta_points=30,
    ):

        super().__init__(
            output_file=output_file,
            sources=sources,
            detector_model_type=detector_model_type,
            includes=includes,
        )

        self._theta_points = theta_points

    def _functions(self):

        with FunctionsContext():

            for include_file in self._includes:
                _ = Include(include_file)

            if self.sources.point_source:

                self._src_spectrum_lpdf = self._ps_spectrum.make_stan_lpdf_func(
                    "src_spectrum_logpdf"
                )

                self._flux_conv = self._ps_spectrum.make_stan_flux_conv_func(
                    "flux_conv"
                )

            if self.sources.diffuse:

                self._diff_spectrum_lpdf = self._diff_spectrum.make_stan_lpdf_func(
                    "diff_spectrum_logpdf"
                )

            if self.sources.atmospheric:

                self._atmo_flux_func = self._atmo_flux.make_stan_function(
                    theta_points=self._theta_points
                )

                self._atmo_flux_integral = self._atmo_flux.total_flux_int.to(
                    1 / (u.m ** 2 * u.s)
                ).value

    def _data(self):

        with DataContext():

            self._N = ForwardVariableDef("N", "int")
            self._N_str = ["[", self._N, "]"]
            self._omega_det = ForwardArrayDef(
                "omega_det", "unit_vector[3]", self._N_str
            )
            self._Edet = ForwardVariableDef("Edet", "vector[N]")
            self._event_type = ForwardVariableDef("event_type", "vector[N]")
            self._kappa = ForwardVariableDef("kappa", "vector[N]")
            self._Esrc_min = ForwardVariableDef("Esrc_min", "real")
            self._Esrc_max = ForwardVariableDef("Esrc_max", "real")

            self._Ns = ForwardVariableDef("Ns", "int")
            self._Ns_str = ["[", self._Ns, "]"]
            self._Ns_1p_str = ["[", self._Ns, "+1]"]
            self._Ns_2p_str = ["[", self._Ns, "+2]"]

            self._varpi = ForwardArrayDef("varpi", "unit_vector[3]", self._Ns_str)
            self._D = ForwardVariableDef("D", "vector[Ns]")

            self._Ngrid = ForwardVariableDef("Ngrid", "int")
            self._Eg = ForwardVariableDef("E_grid", "vector[Ngrid]")

            self._T = ForwardVariableDef("T", "real")

            if self.sources.diffuse:

                N_int_str = self._Ns_1p_str
                self._z = ForwardVariableDef("z", "vector[Ns+1]")

            else:

                N_int_str = self._Ns_str
                self._z = ForwardVariableDef("z", "vector[Ns]")

            if self.sources.point_source or self.sources.diffuse:

                if self.sources.point_source:

                    self._src_index_grid = ForwardVariableDef(
                        "src_index_grid", "vector[Ngrid]"
                    )

                if self.sources.diffuse:

                    self._diff_index_grid = ForwardVariableDef(
                        "diff_index_grid", "vector[Ngrid]"
                    )

                if "tracks" in self._event_types:

                    self._integral_grid_t = ForwardArrayDef(
                        "integral_grid_t", "vector[Ngrid]", N_int_str
                    )

                if "cascades" in self._event_types:

                    self._integral_grid_c = ForwardArrayDef(
                        "integral_grid_c", "vector[Ngrid]", N_int_str
                    )

            if self.sources.diffuse and self.sources.atmospheric:

                N_pdet_str = self._Ns_2p_str

            elif self.sources.diffuse or self.sources.atmospheric:

                N_pdet_str = self._Ns_1p_str

            else:

                N_pdet_str = self._Ns_str

            if "tracks" in self._event_types:

                self._Pg_t = ForwardArrayDef("Pdet_grid_t", "vector[Ngrid]", N_pdet_str)

            if "cascades" in self._event_types:

                self._Pg_c = ForwardArrayDef("Pdet_grid_c", "vector[Ngrid]", N_pdet_str)

            if self.sources.point_source:

                self._L_scale = ForwardVariableDef("L_scale", "real")

            if self.sources.diffuse:

                self._F_diff_scale = ForwardVariableDef("F_diff_scale", "real")

            if self.sources.atmospheric:

                self._atmo_integ_val = ForwardVariableDef("atmo_integ_val", "real")
                self._F_atmo_scale = ForwardVariableDef("F_atmo_scale", "real")

            self._F_tot_scale = ForwardVariableDef("F_tot_scale", "real")
