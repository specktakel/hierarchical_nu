#!/usr/bin/env python

from draw_inelasticity import *
from interaction_info import interaction_source_files

# example usage
# warning: this will be painfully slow during the first run
# subsequent executions will be much faster, since the CDFs can be loaded from ./pickle/*

pickledir = './pickle/'
source = interaction_source_files()

sampler = inelasticity_sampler(source, pickledir)

interaction_type = 'nubar_CC' # nubar_NC, nu_CC, nu_NC
log_energy = 5.6
nsamples = 1000

print sampler.sample_y(log_energy, interaction_type, nsamples)
