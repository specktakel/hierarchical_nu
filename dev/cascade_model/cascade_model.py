#!/usr/bin/env python

import matplotlib
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

import pandas as pd
import numpy as np


# cusomt imports

# sample neutrino energies from powerlaw
from pareto_sampler.pareto import truncated_pareto
# sample neutrino directions uniformly across sky
from recodir_sampler.uniform_sampler import uniform_from_sphere, transform_to_cartesian, transform_to_spherical
# setup y sampler and create y_samples
from inelasticity_sampler.draw_inelasticity import inelasticity_sampler
from inelasticity_sampler.interaction_info import interaction_source_files
# calculate weights
from aeff_calculator.aeff_calculator import effective_area
# dummy angular resolution
# concentration as function of energy deposit
from recodir_sampler.vMF_sampler import VonMisesFisher

class cascade_sim(object):
    def __init__(self, pars):
		self.lEmin = pars['lEmin']
		self.lEmax = pars['lEmax']
		self.gamma_inj = pars['gamma']
		self.neutrino_type = pars['neutrino_type'] # string
		self.interaction_type = pars['interaction_type'] # string

		self.sim_component = self.neutrino_type + '_' + self.interaction_type
		
		if 'bar' in self.neutrino_type:
		    ytype = 'nubar_'+self.interaction_type
		else:
		    ytype = 'nu_'+self.interaction_type

		self.ytable = ytype

		print "... setting up all samplers"
		# internal variables
		print "... setting up powerlaw sampler"
		self.pareto = truncated_pareto(self.gamma_inj, 10**self.lEmin, 10**self.lEmax)
		print "done."
		
		print "... setting up effective area"
		self.aeff = effective_area(self.sim_component)
		print "done."

		print "... setting up inelasticity sampler"
		pickledir = './inelasticity_sampler/pickle/'
		source = interaction_source_files()
		self.kinematics_sampler = inelasticity_sampler(source, pickledir)
		print "done"

		print "... setting up directional sampler"
		self.vMF = VonMisesFisher()

		print "all done!"


    def run_simulation(self, nsamples, outfile):
		print "... starting simulation for", nsamples, "events"
        # sample neutrino energies from powerlaw 
		neutrino_energies = self.pareto.samples(nsamples)
		neutrino_energies_log = np.log10(neutrino_energies)
        
        # sample neutrino directions
		neutrino_directions = uniform_from_sphere(nsamples)
		neutrino_directions_cartesian = np.asarray(transform_to_cartesian(neutrino_directions))
		neutrino_cos_thetas = np.asarray(neutrino_directions[:,0])

        # calculate weights
		effective_areas = np.power(10, np.asarray([self.aeff.eval(lE, ct) for lE,ct in zip(neutrino_energies_log, neutrino_cos_thetas)]))
		effective_areas = effective_areas.flatten()
		weights = 1./nsamples * effective_areas / self.pareto.pdf(neutrino_energies) * 4. * np.pi * 10.**4
        # get deposited energy
        

		#bjorken_y = np.ones(nsamples)
		bjorken_y = np.asarray([self.kinematics_sampler.sample_y(lE, self.ytable, 1) for lE in neutrino_energies_log]).flatten()
		e_had = bjorken_y * neutrino_energies
		e_em = (1.-bjorken_y) * neutrino_energies
		e_dep = e_em + self.__get_hadronic_lightscale__(e_had) * e_had 

        # get reconstructed energies
		reconstructed_energies = np.random.normal(e_dep, self.__get_energy_resolution__(e_dep))

        # get reconstructed directions
		kappas = self.__get_kappa__(e_dep)
		reconstructed_directions = []
		for obj in zip(neutrino_directions_cartesian.tolist(), kappas.tolist()):
			mu, kappa = obj
			tdis = self.vMF.randomize_about_point(np.asarray([mu]), kappa=kappa, num_samples=1)
			reconstructed_directions.append(transform_to_spherical(tdis)[0])
		dirs = np.vstack(reconstructed_directions)

		print "... writing output to file"
		to_store={}
		to_store["prim_energy"]=np.asarray(neutrino_energies)
		to_store["prim_coszenith"]=np.asarray(neutrino_directions[:,0])
		to_store["prim_azimuth"]=np.asarray(neutrino_directions[:,1])
		to_store["rec_energy"]=np.asarray(reconstructed_energies)
		to_store["rec_coszenith"]=np.asarray(dirs[:,0])
		to_store["rec_azimuth"]=np.asarray(dirs[:,1])
		to_store["generation_weight"]=np.asarray(weights)
		to_store["dep_energy"]=np.asarray(e_dep)
		to_store["bjorken_y"]=np.asarray(bjorken_y)

		names = to_store.keys()
		X = np.hstack(tuple([to_store[key].reshape(to_store[key].shape[0], -1) for key in names]))
		df = pd.DataFrame(X, columns=names)
		store = pd.HDFStore(outfile)
		store[self.sim_component]=df
		store.close()
		print "done!"
		return 

    def __get_energy_resolution__(self, edep):
        # energy resolution as fuction of energy deposit
        # deterministic
		relative_resolution = 0.1 # need Erec paper or JvS thesis
		energy_resolution = edep * relative_resolution
		return energy_resolution

    def __get_kappa__(self, edep):
        # angular error as function of energy deposit
        # deterministic
		err = 10.*np.ones(len(edep)) / 180. * np.pi # need Erec paper or JvS thesis
		var = err**2
		kappa = 2.3 / var
		return kappa

    def __get_hadronic_lightscale__(self, ehad):
        # reduce visible energy in hadronic component
        # (stochastic)
		xi = 1.0
		return xi # from EM thesis



'''
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
'''

    





