#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from rootpy.plotting import Hist
from numpy.random import uniform
import os
import cPickle as pickle

class energy_slice(object):
    def __init__(self, data, pickledir, idx):
        self.data = data
        self.transformed_data = np.sqrt(self.data)

        self.bw = 0.08
        self.kde_nbins = 2000
        self.range_low = 0.0
        self.range_high = 1.0
        self.hist_nbins = 30
        
        self.x_vals = np.linspace(self.range_low, self.range_high, self.kde_nbins)

        # need to re-calculate inverse cdf - or can we just load it?
        self.picklefile = pickledir+"bin_"+str(idx)+"_invcdf_vals.pickle"

        if not os.path.isfile(self.picklefile):
            from kde import meerkat_input, meerkat_kde
            
            # build histograms
            edges = np.linspace(self.range_low, self.range_high, self.hist_nbins+1)
            self.edges = edges

            self.hist_data = Hist(edges)
            self.hist_data.fill_array(self.data)
            self.hist_data.scale(1./self.hist_data.Integral("width"))

            self.hist_transformed_data = Hist(edges)
            self.hist_transformed_data.fill_array(self.transformed_data)
            self.hist_transformed_data.scale(1./self.hist_transformed_data.Integral("width"))
            
 
            # build kde
            args = {'name':'sqrt_inelasticity', 'values':self.transformed_data, 'bandwidth': self.bw, 'nbins':self.kde_nbins, 'range':[self.range_low, self.range_high]}
            m_input = meerkat_input([args], np.ones(len(self.transformed_data)), mc_conv=len(self.transformed_data))
            self.kde = meerkat_kde(m_input)

            # normalize with quad, since we will use quad later
            pdf = lambda x: self.kde.eval_point([x])
            self.kde_norm = quad(pdf, self.range_low, self.range_high)[0]

            # build inverse CDF        
            def cdf(x):
                pdf = lambda x: self.kde.eval_point([x]) / self.kde_norm
                val = quad(pdf, self.range_low, x)[0]

                if val < 0.0:
                    val = 0.0
                if val > 1.0:
                    val = 1.0

                return val
   
            y_vals_cdf = np.array([cdf(tx) for tx in self.x_vals])
            y_vals_cdf[0] = 0.0 # guard against numeric problems
            y_vals_cdf[-1] = 1.0

            tf = open(self.picklefile, "wb")
            pickle.dump(y_vals_cdf, tf)
            tf.close()

        else:
            
            tf = open(self.picklefile, "rb")
            y_vals_cdf = pickle.load(tf)
            

        self.y_vals_cdf = y_vals_cdf            

        # have to take care of the regions with 0 probability 
        nonzero = np.nonzero(y_vals_cdf)
        idx_first = int(nonzero[0][0])-1
        self.inv_cdf = interp1d(y_vals_cdf[idx_first:], self.x_vals[idx_first:],  kind=5) 
    
    def sample_y(self, nsamples=1):
        u = uniform(self.range_low, self.range_high, nsamples)
        return self.inv_cdf(u)

        
class inelasticity_sampler_interaction(object):
    def __init__(self, infile_xy, pickledir): 
        
        with open(infile_xy) as f:
            fline = f.readline()
        
        fline = [int(s) for s in fline.split(" ")]
        nsamples, blank, nbins, lEmin, lEmax = fline
        
        self.nsamples = nsamples
        self.nbins = nbins
        self.lEmin = lEmin
        self.lEmax = lEmax
        edges_lE = np.linspace(lEmin, lEmax, nbins+1)
        
        self.energy_hist = Hist(edges_lE)
        data = np.loadtxt(infile_xy, skiprows=1)

        self.y_table = []
        for i in range(nbins):
            tdat = data[self.nsamples*i : self.nsamples*(i+1),1] 
            self.y_table.append(tdat)

        self.y_table = np.vstack(self.y_table)
		
        self.samplers = [energy_slice(self.y_table[i], pickledir, i) for i in range(len(self.y_table))]

    def get_index(self, lE):
        return self.energy_hist.find_bin(lE)

    def sample_y(self, lE, nsamples=1): 
        random_sqrt_y = self.sample_sqrt_y(lE, nsamples)
        return random_sqrt_y**2

    def sample_sqrt_y(self, lE, nsamples=1):
        idx = self.get_index(lE) 
        random_sqrt_y = self.samplers[idx].sample_y(nsamples=nsamples)
        return random_sqrt_y 


class inelasticity_sampler(object):
    def __init__(self, input_xy, pickledir):
        self.sampler_type = {}
        for key in input_xy.itype.keys():
            self.sampler_type[key] = inelasticity_sampler_interaction(input_xy.itype[key], pickledir+key+"/")

        #key = 'nubar_CC'
        #self.sampler_type[key] = inelasticity_sampler_interaction(input_xy.itype[key], pickledir+key+"/")

    def sample_y(self, lE, interaction_type, nsamples=1):
        return self.sampler_type[interaction_type].sample_y(lE, nsamples)
        
    def sample_sqrt_y(self, lE, interaction_type, nsamples=1):
        return self.sampler_type[interaction_type].sample_sqrt_y(lE, nsamples) 
