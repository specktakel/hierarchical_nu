import os
from configya import YAMLConfig

from ..stan_interface import STAN_PATH

# Some config defaults
file_config = {}
file_config["atmo_sim_filename"] = os.path.join(STAN_PATH, "atmo_gen.stan")
file_config["main_sim_filename"] = os.path.join(STAN_PATH, "sim_code.stan")
file_config["fit_filename"] = os.path.join(STAN_PATH, "model_code.stan")
file_config["include_paths"] = [
    STAN_PATH,
]

parameter_config = {}
parameter_config["alpha"] = 2.3
parameter_config["alpha_range"] = (1.0, 4.0)
parameter_config["L"] = 5e46  # * u.erg / u.s
parameter_config["L_range"] = (0, 1e60)
parameter_config["Enorm"] = 1e5  # * u.GeV
parameter_config["Emin"] = 5e4  # * u.GeV
parameter_config["Emax"] = 1e8  # * u.GeV
parameter_config["diff_norm"] = 5e-14  # * (1 / (u.GeV * u.m**2 * u.s))
parameter_config["obs_time"] = 10  # * u.year

# If True, use same Emin_det for all
# If False, use separate for tracks and cascades
parameter_config["Emin_det_eq"] = False
parameter_config["Emin_det"] = 1e5  # * u.GeV
parameter_config["Emin_det_tracks"] = 1e5  # * u.GeV
parameter_config["Emin_det_cascades"] = 6e4  # * u.GeV

# Can be "icecube", "northern_tracks" or "cascades"
parameter_config["detector_model_type"] = "icecube"


class FileConfig(YAMLConfig):
    def __init__(self):

        super(FileConfig, self).__init__(
            file_config, "~/.hierarchical_nu", "file_config.yml"
        )


class ParameterConfig(YAMLConfig):
    def __init__(self):

        super(ParameterConfig, self).__init__(
            parameter_config, "~/.hierarchical_nu", "parameter_config.yml"
        )
