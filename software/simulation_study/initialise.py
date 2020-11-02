from astropy import units as u
import sys

sys.path.append("../")

from model_check import ModelCheck

ModelCheck.initialise_env(
    Emin=1e5 * u.GeV,
    Emax=1e8 * u.GeV,
    atmo_sim_name="stan_files/atmo_gen",
    main_sim_name="stan_files/sim_code",
    fit_filename="stan_files/model_code",
)
