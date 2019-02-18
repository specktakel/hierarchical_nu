#!/usr/bin/env python

import matplotlib
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

import pandas as pd
import numpy as np

neutrino_interaction = 'CC'
neutrino_flavor = 'nue'
nsamples = int(1.e1)

lEmin = 3.0
lEmax = 7.0
gamma = 1.0

sim_component = neutrino_flavor + '_' + neutrino_interaction

if 'bar' in neutrino_flavor:
    ytype = 'nubar_'+neutrino_interaction
else:
    ytype = 'nu_'+neutrino_interaction


outfile = './output/'+sim_component+'_simulation.h5'

# sample neutrino energies from powerlaw
from pareto_sampler.pareto import truncated_pareto
pareto = truncated_pareto(gamma, 10**lEmin, 10**lEmax)
neutrino_energies = pareto.samples(nsamples)
neutrino_energies_log = np.log10(neutrino_energies)
print neutrino_energies_log

# sample neutrino directions uniformly across sky
from recodir_sampler.uniform_sampler import uniform_from_sphere, transform_to_cartesian, transform_to_spherical
neutrino_directions = uniform_from_sphere(nsamples)
neutrino_directions_cartesian = transform_to_cartesian(neutrino_directions)
neutrino_cos_thetas = np.asarray(neutrino_directions[:,0])
print neutrino_cos_thetas


# setup y sampler and create y_samples
from inelasticity_sampler.draw_inelasticity import inelasticity_sampler
from inelasticity_sampler.interaction_info import interaction_source_files
source = interaction_source_files()
pickledir = './inelasticity_sampler/pickle/'

log_central_energy = 5.
kinematics_sampler = inelasticity_sampler(source, pickledir)
y_samples =  kinematics_sampler.sample_y(log_central_energy, ytype, nsamples)
print y_samples


# calculate weights
from aeff_calculator.aeff_calculator import effective_area
aeff = effective_area(sim_component) 

effective_areas = np.power(10, np.asarray([aeff.eval(lE, ct) for lE,ct in zip(neutrino_energies_log, neutrino_cos_thetas)]))
effective_areas = effective_areas.flatten()
print effective_areas

weights = 1./nsamples * effective_areas / pareto.pdf(neutrino_energies) * 4. * np.pi * 10.**4
print weights


# calculate energy deposit as function of interaction type and inelasticity
# this needs to be a function of the interaction type (NC, CC, hadronic light scale etc)
energy_deposits = neutrino_energies

# dummy energy resolution
# relative resolution needs to be a function of energy deposit
relative_resolution = 0.1
energy_resolutions = energy_deposits * relative_resolution
reconstructed_energies = np.random.normal(energy_deposits, energy_resolutions)
print reconstructed_energies


# dummy angular resolution
# concentration as function of energy deposit
from recodir_sampler.vMF_sampler import VonMisesFisher
err = 10. / 180. * np.pi
var = err**2
kappa = 2.3 / var 
reconstructed_directions = []
vMF = VonMisesFisher()
for mu in neutrino_directions_cartesian: 
    tdis = vMF.randomize_about_point(np.asarray([mu]), kappa=kappa, num_samples=1)
    reconstructed_directions.append(transform_to_spherical(tdis)[0])
print reconstructed_directions

    





