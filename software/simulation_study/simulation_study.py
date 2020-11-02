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

cwd = os.getcwd()
# TODO add job ID to filename
output_file = os.path.join(cwd, "output/sim_study_test.h5")

model_check = ModelCheck()
model_check.parallel_run(n_jobs=4)
model_check.save(output_file)
