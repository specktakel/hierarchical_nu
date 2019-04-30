#!/usr/bin/env python

"""
Plot effective area tables supplied with the article.
Jakob van Santen <jvansanten@icecube.wisc.edu>, 2014-08-01
"""

import numpy
import pylab
import os

def load_aeff(flavor):
	"""Load an effective area table"""
	fname = os.path.join(os.path.dirname(__file__), 'effective_area.nu_%s.txt' % flavor)
	names = ('energy_lo', 'energy_hi', 'ct_lo', 'ct_hi', 'aeff', 'error')
	dt = numpy.dtype([(n, float) for n in names])
	table = numpy.loadtxt(fname, dtype=dt)
	shape = tuple(numpy.unique(table[field]).size for field in ('ct_lo', 'energy_lo'))
	table = table.reshape(shape)
	return table

def stepped_path(left_edges, right_edges, bins):
	"""Convert left/right edges and bin center values to a stepped path"""
	edges = numpy.concatenate((left_edges, [right_edges[-1]]))
	x = numpy.zeros((2*len(edges)))
	y = numpy.zeros((2*len(edges)))
	
	x[0::2], x[1::2] = edges, edges
	y[1:-1:2], y[2::2] = bins, bins
	
	return x, y

for flavor  in 'e', 'mu', 'tau':
	table = load_aeff(flavor)

	pylab.figure()
	ax = pylab.gca()
	for i in range(0, table.shape[0], 2):
		# a chunk of constant zenith angle
		chunk = table[i,:]
		mid = stepped_path(chunk['energy_lo'], chunk['energy_hi'], chunk['aeff'])
		lo = stepped_path(chunk['energy_lo'], chunk['energy_hi'], chunk['aeff']-chunk['error'])
		hi = stepped_path(chunk['energy_lo'], chunk['energy_hi'], chunk['aeff']+chunk['error'])
		# central value
		pylab.plot(*mid, label='%.1f - %.1f' % (chunk['ct_lo'][0], chunk['ct_hi'][0]))
		# represent error range with a fill
		pylab.fill_between(lo[0], lo[1], hi[1], color=ax.lines[-1].get_color(), alpha=0.5)

	pylab.legend(loc='upper left', title=r'$\cos\theta$')
	pylab.ylim((1e-4, 1e3))
	pylab.loglog(nonposy='clip')
	pylab.title(r'Effective Area for $\nu_{%s}$' % ([flavor, '\\'+flavor][len(flavor)>1]))
	pylab.ylabel(r'$A_{\rm eff} \,\, [m^2]$')
	pylab.xlabel('Neutrino Energy [GeV]')
	pylab.grid()
pylab.show()
