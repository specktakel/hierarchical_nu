#!/usr/bin/env python

from draw_inelasticity import *

import rootpy.plotting.root2matplotlib as rplt
import matplotlib.pyplot as plt
from rootpy.plotting import Hist
import numpy as np

for idx in range(111):
#for idx in range(1):
    sampler = inelasticity_sampler("./xsec/xy_nubar_CC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat", idx)
    log_central_energy = sampler.energy_hist[idx+1].x.center
    central_energy = 10**log_central_energy

    resamples = sampler.sample_y(0, nsamples=int(1.e6)) # interface will change
    
    hist = sampler.samplers[0].hist_data
    hist_sqrt = sampler.samplers[0].hist_transformed_data
    
    hist_new = Hist(sampler.samplers[0].edges)
    hist_sqrt_new = Hist(sampler.samplers[0].edges)
    
    hist_new.fill_array(resamples)
    hist_new.scale(1./hist_new.Integral("width"))
    hist_sqrt_new.fill_array(np.sqrt(resamples))
    hist_sqrt_new.scale(1./hist_sqrt_new.Integral("width"))
    
    plt.style.use('ggplot')
    #fig, ((ax1), (ax2)) = plt.subplots(nrows=1, ncols=2)
    
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    #ratio = 0.7
    #fig = plt.figure()
    #ax1 = fig.add_subplot(2,1,1)
    #ax2 = fig.add_subplot(2,2,1)
    hist_sqrt.title="original"
    rplt.step(hist_sqrt, color='b', axes=ax1)
    hist_sqrt_new.title="resample"
    rplt.step(hist_sqrt_new, color='r', axes=ax1)
    hist.title="original"
    rplt.step(hist, color='b', axes=ax2)
    hist_new.title="resample"
    rplt.step(hist_new, color='r', axes=ax2)
    ax1.set_xlabel('sqrt(y)', fontsize=18)
    ax1.set_ylabel('pdf', fontsize=18)
    ax1.set_title('energy: %.2e GeV (log_energy: %.2f)' %(central_energy, log_central_energy))
    ax2.set_xlabel('y', fontsize=18)
    ax2.set_ylabel('pdf', fontsize=18)

    ax1.legend()
    ax2.legend()

    ax1.set_xlim([0.0, 1.0])
    ax2.set_xlim([0.0, 1.0])
    
    #plt.subplots_adjust(wspace=0.5)
    #ax1.set_aspect(0.3)
    #ax2.set_aspect(0.3)
    
    plt.tight_layout()
    plt.savefig("./plots/bin_"+str(idx)+"_sqrt_y.pdf")









