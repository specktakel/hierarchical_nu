import sys
import os

sys.path.append("../")
from python.model_check import ModelCheck

"""
Script to run fits to simulations for 
a selection of known input parameters.

The configuration is specified in the YAMLConfig 
files as detailed in python.config 
"""

n_jobs = int(sys.argv[1])
n_subjobs = int(sys.argv[2])
seed = int(sys.argv[3])

cwd = os.getcwd()
output_file = os.path.join(cwd, f"output/fit_sim_numu_{seed}.h5")

model_check = ModelCheck()
model_check.parallel_run(n_jobs=n_jobs, n_subjobs=n_subjobs, seed=seed)
model_check.save(output_file)
