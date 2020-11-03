import os
from configya import YAMLConfig

cwd = os.getcwd()

# Some config defaults
file_config = {}
file_config["atmo_sim_filename"] = os.path.join(cwd, "stan_files/atmo_gen.stan")
file_config["main_sim_filename"] = os.path.join(cwd, "stan_files/sim_code.stan")
file_config["fit_filename"] = os.path.join(cwd, "stan_files/model_code.stan")
file_config["include_paths"] = [
    os.path.join(cwd, "stan_files"),
]

parameter_config = {}
parameter_config["alpha"] = 2.3
parameter_config["alpha_range"] = (1.0, 4.0)
parameter_config["L"] = 1e47  # * u.erg / u.s
parameter_config["L_range"] = (0, 1e60)
parameter_config["Enorm"] = 1e5  # * u.GeV
parameter_config["Emin"] = 1e5  # * u.GeV
parameter_config["Emax"] = 1e8  # * u.GeV
parameter_config["Emin_det"] = 1e5  # * u.GeV
parameter_config["diff_norm"] = 1.44e-14  # * (1 / (u.GeV * u.m**2 * u.s))
parameter_config["obs_time"] = 10  # * u.year


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
