from astropy import units as u
import sys

sys.path.append("../")

from python.model_check import ModelCheck

ModelCheck.initialise_env(
    Emin=1e5 * u.GeV,
    Emax=1e8 * u.GeV,
    output_dir="output",
)
