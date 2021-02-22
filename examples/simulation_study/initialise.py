import numpy as np
import sys

from hierarchical_nu.model_check import ModelCheck

ModelCheck.initialise_env(output_dir="output")

# Make seeds.txt file depending on number of parallel srun tasks
n_tasks = int(sys.argv[-1])
seed_list = np.linspace(10, 99, n_tasks).astype(int) * 100
np.savetxt("seeds.txt", seed_list, fmt="%i")
