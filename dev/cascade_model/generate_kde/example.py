#!/usr/bin/env python
import itertools
import matplotlib as mpl
mpl.use('TkAgg')
import pylab as plt
import time

import ROOT
from kde import *
from numpy.lib.recfunctions import append_fields

#from fancy_plot import *

#nsamples = int(1.e5)
nsamples = -1

mc=np.load('dataset_8yr_fit_IC86_2015_16_mc_2017_11_21.npy')[:nsamples]
mc = mc[(mc['trueDec']>np.radians(-5)) & (mc['trueE']<10**(7.0))]

mc_conv = len(mc)

def GreatCircleDistance(ra_1, dec_1, ra_2, dec_2):
    '''Compute the great circle distance between two events'''
    '''All coordinates must be given in radians'''
    delta_dec = np.abs(dec_1 - dec_2)
    delta_ra = np.abs(ra_1 - ra_2)
    x = (np.sin(delta_dec / 2.))**2. + np.cos(dec_1) *\
        np.cos(dec_2) * (np.sin(delta_ra / 2.))**2.
    return 2. * np.arcsin(np.sqrt(x))

psi = GreatCircleDistance(mc['trueRa'], mc['trueDec'], mc['ra'], mc['dec'])
mc = append_fields(mc, ['psi', 'logEt'], [psi, np.log10(mc['trueE'])])


from rootpy.plotting import Hist, Hist2D, HistStack, Legend, Canvas
from rootpy.plotting.style import get_style, set_style
from rootpy.plotting.utils import get_limits

import rootpy.plotting.root2matplotlib as rplt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
set_style('ATLAS', mpl=True)

def powerlaw(Et):
    gamma=1
    return (Et/1.e5)**(-gamma)

plaw = np.vectorize(powerlaw)
    
# create binned pdf
weights = mc['orig_OW']*plaw(mc['trueE'])

psi = np.log(mc['psi'])
psi_range = [-12,np.log10(np.pi)]

dec = mc['trueDec']
dec_range = [np.sin(np.radians(-5)), 1]

Et = np.log10(mc['trueE'])
Et_range = [2.0, 7.0]

Er = mc['logE']
print "Er_max=",int(np.amax(Er))+1.0
Er_range = [2.0, int(np.amax(Er))+1.0]

silverman = 0.3

args1 = {'name':'logEt', 'values': Et, 'bandwidth': 0.9 * silverman, 'nbins':40, 'range': Et_range}
args2 = {'name':'trueDec', 'values': dec, 'bandwidth': 0.5 * silverman, 'nbins':100, 'range': dec_range}
args3 = {'name':'logPsi', 'values': psi, 'bandwidth': 0.5 * silverman, 'nbins':100, 'range': psi_range}
args4 = {'name':'logEr', 'values': Er, 'bandwidth': 0.8 * silverman, 'nbins':40, 'range': Er_range}

m_input = meerkat_input([args1, args2, args3, args4], weights, mc_conv=mc_conv)
m_kde4d = meerkat_kde(m_input)

nbins_x = 50
nbins_y = 50
erbins = np.linspace(2.0, 7.0, nbins_x)
etbins = np.linspace(2.0, 7.0, nbins_y)
hist = Hist2D(etbins, erbins)
for tbin in hist:
        tbin.value = m_kde4d.eval_point([tbin.x.center, 0.2, -3.5, tbin.y.center])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
fig.subplots_adjust(right=0.8)
cmap = plt.get_cmap('viridis')
lin = rplt.hist2d(hist, axes=ax, cmap = cmap, norm=mpl.colors.LogNorm(vmin=1.e-4, vmax=2.))
#lin = rplt.hist2d(hist, axes=ax, cmap = cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=1.5))
cbar_ax1 = fig.add_axes([0.83, 0.16, 0.07, 0.79])
cbar = fig.colorbar(lin[3], cbar_ax1)
plt.show()



nbins = 50
nbins_x, nbins_y, nbins_z, nbins_w = nbins, nbins, nbins, nbins
bins_Et = np.linspace(2.0, 7.0, nbins_x)
bins_dec = np.linspace(np.radians(-5), 1, nbins_y)
bins_psi = np.linspace(-12, np.log10(np.pi), nbins_z)
bins_Er = np.linspace(2.0, 7.0, nbins_w)

coords = np.array(list(itertools.product(bins_Et, bins_dec, bins_psi, bins_Er)))
pdf_vals = np.asarray([m_kde4d.eval_point(coord) for coord in coords])
pdf_vals = pdf_vals.reshape(nbins_x,nbins_y,nbins_z, nbins_w)

import cPickle as pickle
with open("kde_4D.dat", 'wb') as fp:
	pickle.dump(dict({'vars':['log10Et', 'SinDect', 'log10Psi', 'log10Er'], 'bins':[bins_Et, bins_dec, bins_psi, bins_Er], 'coords':coords, 'pdf_vals':pdf_vals}), fp)

silverman = 0.2

args1 = {'name':'logEt', 'values': Et, 'bandwidth': 0.9 * silverman, 'nbins':200, 'range': Et_range}
args2 = {'name':'trueDec', 'values': dec, 'bandwidth': 0.3 * silverman, 'nbins':200, 'range': dec_range}
args3 = {'name':'logPsi', 'values': psi, 'bandwidth': 0.8 * silverman, 'nbins':200, 'range': psi_range}

m_input = meerkat_input([args1, args2, args3], weights, mc_conv=mc_conv)
m_kde3d = meerkat_kde(m_input)

erbins = np.linspace(2.0, 7.0, nbins_x)
etbins = np.linspace(dec_range[0], 1.0, nbins_y)
hist = Hist2D(etbins, erbins)
for tbin in hist:
        tbin.value = m_kde3d.eval_point([tbin.x.center, tbin.y.center, -3.5])

'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
fig.subplots_adjust(right=0.8)
cmap = plt.get_cmap('viridis')
lin = rplt.hist2d(hist, axes=ax, cmap = cmap, norm=mpl.colors.LogNorm(vmin=1.e-4, vmax=2.))
#lin = rplt.hist2d(hist, axes=ax, cmap = cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=1.5))
cbar_ax1 = fig.add_axes([0.83, 0.16, 0.07, 0.79])
cbar = fig.colorbar(lin[3], cbar_ax1)
plt.show()
'''



nbins_x, nbins_y, nbins_z, nbins_w = nbins, nbins, nbins, nbins
bins_Et = np.linspace(2.0, 7.0, nbins_x)
bins_dec = np.linspace(np.radians(-5), 1, nbins_y)
bins_psi = np.linspace(-12, np.log10(np.pi), nbins_z)

coords = np.array(list(itertools.product(bins_Et, bins_dec, bins_psi)))
pdf_vals = np.asarray([m_kde3d.eval_point(coord) for coord in coords])
pdf_vals = pdf_vals.reshape(nbins_x,nbins_y,nbins_z)

import cPickle as pickle
with open("kde_3D_EtDecPsi.dat", 'wb') as fp:
        pickle.dump(dict({'vars':['log10Et', 'SinDect', 'log10Psi'], 'bins':[bins_Et, bins_dec, bins_psi], 'coords':coords, 'pdf_vals':pdf_vals}), fp)


'''
erbins = np.linspace(2.0, 7.0, nbins_x)
etbins = np.linspace(2.0, 7.0, nbins_y)
hist = Hist2D(etbins, erbins)
for tbin in hist:
	tbin.value = m_kde4d.eval_point([tbin.x.center, 0.2, -3.5, tbin.y.center])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
fig.subplots_adjust(right=0.8)
cmap = plt.get_cmap('viridis')
lin = rplt.hist2d(hist, axes=ax, cmap = cmap, norm=mpl.colors.LogNorm(vmin=1.e-4, vmax=2.))
#lin = rplt.hist2d(hist, axes=ax, cmap = cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=1.5))
cbar_ax1 = fig.add_axes([0.83, 0.16, 0.07, 0.79])
cbar = fig.colorbar(lin[3], cbar_ax1)
plt.show()
'''










	



	
	
	
	







 





