from draw_inelasticity import *

import rootpy.plotting.root2matplotlib as rplt
import matplotlib.pyplot as plt
from rootpy.plotting import Hist
import numpy as np

from interaction_info import interaction_source_files


interaction_type = 'nubar_CC'
xsec_data = '../csms_input/xy_nubar_CC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat' # original data for nubar_CC case

#interaction_type = 'nubar_NC'
#xsec_data = '../csms_input/xy_nubar_NC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat' # original data for nubar_CC case

#interaction_type = 'nu_NC'
#xsec_data = '../csms_input/xy_nu_NC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat' # original data for nubar_CC case

#interaction_type = 'nu_CC'
#xsec_data = '../csms_input/xy_nu_CC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat' # original data for nubar_CC case





pickledir = './pickle/' # location of pre computed CDF tables

source = interaction_source_files()

the_sampler = inelasticity_sampler(source, pickledir)

# get the nubar_CC sampler
#sampler = inelasticity_sampler(xsec_data, pickledir)

##########################################################################
# the_sampler.sample_y(logE, interaction_type, nsamples) does the trick. #
# Here we check the correctness                                          #
##########################################################################

### check input/output consistency
### by means of plotting both

with open(xsec_data) as f:
    fline = f.readline()

fline = [int(s) for s in fline.split(" ")]
nsamples, blank, nbins, lEmin, lEmax = fline

edges_lE = np.linspace(lEmin, lEmax, nbins+1)

energy_hist = Hist(edges_lE)
data = np.loadtxt(xsec_data, skiprows=1)

y_table = []
for i in range(nbins):
    tdat = data[nsamples*i : nsamples*(i+1),1] 
    y_table.append(tdat)

y_table = np.vstack(y_table)


for idx in range(len(y_table)):
    data = y_table[idx]
    transformed_data = np.sqrt(data)

    current_sampler = the_sampler.sampler_type[interaction_type]
    log_central_energy = current_sampler.energy_hist[idx].x.center    
    central_energy = 10**log_central_energy

    # request resampled y and sqrt(y) in this energy bin
    y_samples = the_sampler.sample_y(log_central_energy, interaction_type, int(1.e6))
    sqrt_y_samples = np.sqrt(y_samples)

    current_sampler_in_energy_bin = current_sampler.samplers[idx]
	# get original binning
    edges = np.linspace(current_sampler_in_energy_bin.range_low, current_sampler_in_energy_bin.range_high, current_sampler_in_energy_bin.hist_nbins+1)

    hist = Hist(edges)
    hist_sqrt = Hist(edges)

    hist.fill_array(data)
    hist.scale(1./hist.Integral("width"))
    hist_sqrt.fill_array(transformed_data)
    hist_sqrt.scale(1./hist_sqrt.Integral("width"))
 
    hist_new = Hist(edges)
    hist_sqrt_new = Hist(edges)
    
    hist_new.fill_array(y_samples)
    hist_new.scale(1./hist_new.Integral("width"))
    hist_sqrt_new.fill_array(sqrt_y_samples)
    hist_sqrt_new.scale(1./hist_sqrt_new.Integral("width"))
    
    plt.style.use('ggplot') 
    
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
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
        
    plt.tight_layout()
    plt.savefig("./pdf/bin_"+str(idx)+"_sqrt_y.pdf")
    plt.close()
