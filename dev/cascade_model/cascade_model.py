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
from inelasticity_sampler.energy_dependencies import \
        get_eres_exinterpolator, get_angres_exinterpolator, get_hadronic_scaling_factor

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

        self.eres_interp = get_eres_exinterpolator()
        self.angularres_interp = get_angres_exinterpolator()
        self.hadr_casc_factor = get_hadronic_scaling_factor()

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

        bjorken_y = []
        reconstructed_energies = []
        deposited_energies = []
        for obj in zip(neutrino_energies, neutrino_energies_log):
            enu, log_enu = obj

            # start rejection sampling to take into account efficiency
            nmax = 100 # maximal number of rejection sampling trials
            accepted = False
            e_dep_min = 10.

            N_not_accepted = 0
            for i in range(nmax):
                by = self.kinematics_sampler.sample_y(log_enu, self.ytable, 1)[0]
                 
                e_had = by * enu
                if self.interaction_type == 'CC':
                    e_em = (1.-by) * enu # only true for nue/nuebar events.
                    # needs to become more realistic for nutau and numu

                else:
                    e_em = 0.0

                light_scale = self.__get_hadronic_lightscale__(e_had)
                e_dep = e_em + light_scale * e_had 
                

                # now rejection sample according to efficiency
                alpha = self.__get_selection_eff__(e_dep)
                if np.random.uniform(0,1) > alpha:
                    # reject event
                    continue

                else:
                    accepted = True
                    break

            if not accepted:
                # the cases where no accepted event generated within nmax trials
                N_not_accepted+=1
                if N_not_accepted%50==0:
                    print "did not accept", N_not_accepted, "events. meh!"

                e_dep = e_dep_min
                by = e_dep / enu 

            bjorken_y.append(by)

            if type(e_dep)==np.array: 
                deposited_energies.append(e_dep[0])
            else:
                deposited_energies.append(e_dep)

            e_rec = -1.0
            while e_rec <= 0.0:
                res = self.__get_energy_resolution__(e_dep)
                if type(res)==np.array:
                    e_rec = np.random.normal(e_dep, res[0])
                else:
                    e_rec = np.random.normal(e_dep, res)

            if type(e_rec)==np.array:
                reconstructed_energies.append(e_rec[0])
            else:
                reconstructed_energies.append(e_rec)

        # get reconstructed directions
        kappas_old = self.__get_kappa__(deposited_energies)
        kappas = []
        for tk in kappas_old:
            if type(tk)==np.array:
                kappas.append(tk[0])
            else:
                kappas.append(tk)

        reconstructed_directions = []
        for obj in zip(neutrino_directions_cartesian.tolist(), kappas):
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
        to_store["dep_energy"]=np.asarray(deposited_energies)
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
        relative_resolution = self.eres_interp(edep)
        energy_resolution = edep * relative_resolution
        return energy_resolution

    def __get_kappa__(self, edep):
        # angular error as function of energy deposit
        # deterministic
        err = self.angularres_interp(edep) / 180. * np.pi
        #err = 20. / 180. * np.pi * np.ones(len(edep))
        var = err**2
        kappa = 2.3 / var
        return kappa

    def __get_hadronic_lightscale__(self, ehad):
        # reduce visible energy in hadronic component
        # (stochastic)
        xi = self.hadr_casc_factor.gen_value(ehad)
        return xi

    def __get_selection_eff__(self, edep):
        # use to do rejection sampling based on edep
        # to take into account some selection efficiency
        # model as sigmoid
        return (1 + np.exp(-4.0*(np.log10(edep)-3.5)))**(-1)



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

    





