import numpy as np
from astropy import units as u
import sys

sys.path.append("../")

from python.model_check import ModelCheck

ModelCheck.initialise_env(
    Emin=1e5 * u.GeV,
    Emax=1e8 * u.GeV,
    output_dir="output",
)

# Make seeds.txt file depending on number of parallel srun tasks
n_tasks = int(sys.argv[-1])
seed_list = np.linspace(10, 99, n_tasks).astype(int) * 100
np.savetxt("seeds.txt", seed_list, fmt="%i")
