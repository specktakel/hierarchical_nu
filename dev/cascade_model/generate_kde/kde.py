#!/usr/bin/env python

import numpy as np
import os, sys

import ROOT
os.environ["ROOT_INCLUDE_PATH"] = os.pathsep + "/home/hans/software/Meerkat/Meerkat/inc/"
from ROOT import gSystem, gStyle, RooRealVar
from ROOT import std, Double

gSystem.Load("/home/hans/software/Meerkat/Meerkat/lib/libMeerkat.so")

from ROOT import OneDimPhaseSpace, CombinedPhaseSpace, BinnedKernelDensity, FactorisedDensity, FormulaDensity, ParametricPhaseSpace,\
    AdaptiveKernelDensity, KernelDensity

from root_numpy import array2tree

class meerkat_variable(object):
	"""
	brainless storage
	"""
	def __init__(self, args):
		self.variable_name = args["name"]
		self.values = args["values"]
		self.bw = args["bandwidth"]
		self.nbins = args["nbins"]
		self.range = args["range"]

class meerkat_input(object):
	"""
	var_args: list of python dictionaries with information of each variable
	weights: numpy.array of weights
	pdf_seed: meerkat_kde object (optional if adaptive = False)
	adaptive: whether to use adaptive binwidth
	"""
	
	def __init__(self, var_args, weights, pdf_seed=None, adaptive=False, mc_conv=0):
			
		# get input
		self.pdf_seed = pdf_seed
		self.adaptive = adaptive
		self.weights = weights
		self.variables = [meerkat_variable(arg) for arg in var_args]
		self.mc_conv = 0

		# create info
		self.ndim = len(self.variables)
		self.var_names = []
		self.data = []
		self.spaces = []
		self.space = None
		self.bws = []
		self.nbins = []

		for var in self.variables:
			self.var_names.append(var.variable_name)
			self.data.append(var.values)
			self.spaces.append(OneDimPhaseSpace(var.variable_name, var.range[0], var.range[1]))
			self.bws.append(var.bw)
			self.nbins.append(var.nbins)

		# create combined phase space
		if self.ndim == 1:
			self.space = self.spaces[0]
		elif self.ndim == 2:
			self.space = CombinedPhaseSpace("PhspCombined", self.spaces[0], self.spaces[1])
		elif self.ndim == 3:
			self.space = CombinedPhaseSpace("PhspCombined", self.spaces[0], self.spaces[1], self.spaces[2])
		elif self.ndim == 4:
			self.space = CombinedPhaseSpace("PhspCombined", self.spaces[0], self.spaces[1], self.spaces[2], self.spaces[3])
		else:
			print "FATAL: ndim > 4 not implemented"
			sys.exit(1)

		# create input data
		self.tree = array2tree( np.array(self.data[0], dtype=[(self.var_names[0], np.float32)]) )
		for i in range(self.ndim-1):
			array2tree( np.array(self.data[i+1], dtype=[(self.var_names[i+1], np.float32)]), tree=self.tree)
		array2tree( np.array(self.weights, dtype=[("weight", np.float32)]), tree=self.tree)
			
			

		
class meerkat_kde(object):
	"""
	interface to meerkat kde functions
	args: meerkat_input object
	"""
	
	def __init__(self, args):

		self.kde_norm = 1.0
		self.kde = None
		self.args = args

                self.approx_pdf = 0
                if args.pdf_seed:
                        self.approx_pdf = args.pdf_seed


		for i in range(args.ndim):
			self.kde_norm *= args.variables[i].range[1] - args.variables[i].range[0]
		
		self.kde_norm = self.kde_norm ** (-1)

		# fixed bw kde
		if not args.adaptive:
			if args.ndim == 1:
				print ""
				print "... constructing 1 dim. KDE (fixed bandwith)"
				print ""
				kde = BinnedKernelDensity("KernelPDF",
                        	      args.space, # Phase space
                        	      args.tree,  # Input ntuple
                        	      args.var_names[0], # Variables to use
                        	      "weight",      # weights
                        	      args.nbins[0],  # Numbers of bins
                        	      args.bws[0],  # Kernel widths
                        	      self.approx_pdf,  # Approximation PDF (0 for flat approximation)
                        	      args.mc_conv  # Sample size for MC convolution (0 for binned convolution)
                        	      )
				
				self.kde = kde

			elif args.ndim == 2:
                                print ""
                                print "... constructing 2 dim. KDE (fixed bandwith)"
                                print ""

                        	kde = BinnedKernelDensity("KernelPDF",
                        	      args.space, # Phase space
                        	      args.tree,  # Input ntuple
                        	      args.var_names[0], args.var_names[1], # Variables to use
                        	      "weight",      # weights
                        	      args.nbins[0], args.nbins[1], # Numbers of bins
                        	      args.bws[0], args.bws[1],  # Kernel widths
                        	      self.approx_pdf,  # Approximation PDF (0 for flat approximation)
                        	      args.mc_conv  # Sample size for MC convolution (0 for binned convolution)
                        	      )

				self.kde = kde

			elif args.ndim == 3:
                                print ""
                                print "... constructing 3 dim. KDE (fixed bandwith)"
                                print ""


                                kde = BinnedKernelDensity("KernelPDF",
                                      args.space, # Phase space
                                      args.tree,  # Input ntuple
                                      args.var_names[0], args.var_names[1], args.var_names[2], # Variables to use
                                      "weight",      # weights
                                      args.nbins[0], args.nbins[1], args.nbins[2], # Numbers of bins
                                      args.bws[0], args.bws[1], args.bws[2],  # Kernel widths
                                      self.approx_pdf,  # Approximation PDF (0 for flat approximation)
                                      args.mc_conv  # Sample size for MC convolution (0 for binned convolution)
                                      )
	
				self.kde = kde
			
			elif args.ndim == 4:
                                print ""
                                print "... constructing 4 dim. KDE (fixed bandwith)"
                                print ""


                                kde = BinnedKernelDensity("KernelPDF",
                                      args.space, # Phase space
                                      args.tree,  # Input ntuple
                                      args.var_names[0], args.var_names[1], args.var_names[2], args.var_names[3], # Variables to use
                                      "weight",      # weights
                                      args.nbins[0], args.nbins[1], args.nbins[2], args.nbins[3], # Numbers of bins
                                      args.bws[0], args.bws[1],  args.bws[2], args.bws[3], # Kernel widths
                                      self.approx_pdf,  # Approximation PDF (0 for flat approximation)
                                      args.mc_conv  # Sample size for MC convolution (0 for binned convolution)
                                      )

				self.kde = kde

		# adaptive bw kde
		else:

			if not self.approx_pdf:
				print "please provide a seed for the KDE. exiting..."
				sys.exit(1)

			if args.ndim == 1:
                                print ""
                                print "... constructing 1 dim. KDE (adaptive bandwith)"
                                print ""

    				kde_adapt = AdaptiveKernelDensity("KernelPDF",
                                      args.space,  # Phase space
                                      args.tree,  # Input ntuple
                                      args.var_names[0],  # Variables to use
                                      "w",  # weights
                                      args.nbins[0],  # Numbers of bins
                                      args.bws[0],  # Kernel widths
                                      self.approx_pdf,  # PDF for width scaling
                                      0,  # Approximation PDF (0 for flat approximation)
                                      args.mc_conv  # Sample size for MC convolution (0 for binned convolution)
                                      )

				self.kde = kde_adapt

                        elif args.ndim == 2:
                                print ""
                                print "... constructing 2 dim. KDE (adaptive bandwith)"
                                print ""

                                kde_adapt = AdaptiveKernelDensity("KernelPDF",
                                      args.space,  # Phase space
				      args.tree,
                                      args.var_names[0], args.var_names[1], # Variables to use
                                      "weight",      # weights
                                      args.nbins[0], args.nbins[1], # Numbers of bins
                                      args.bws[0], args.bws[1],  # Kernel widths
                                      self.approx_pdf,  # PDF for width scaling
                                      0,  # Approximation PDF (0 for flat approximation)
                                      args.mc_conv  # Sample size for MC convolution (0 for binned convolution)
                                      )

				self.kde = kde_adapt

                        elif args.ndim == 3:
                                print ""
                                print "... constructing 3 dim. KDE (adaptive bandwith)"
                                print ""

                                kde_adapt = AdaptiveKernelDensity("KernelPDF",
                                      args.space,  # Phase space
                                      args.tree,  # Input ntuple
                                      args.var_names[0], args.var_names[1], args.var_names[2], # Variables to use
                                      "weight",      # weights
                                      args.nbins[0], args.nbins[1], args.nbins[2], # Numbers of bins
                                      args.bws[0], args.bws[1], args.bws[2],  # Kernel widths

                                      self.approx_pdf,  # PDF for width scaling
                                      0,  # Approximation PDF (0 for flat approximation)
                                      args.mc_conv  # Sample size for MC convolution (0 for binned convolution)
                                      )

				self.kde = kde_adapt

                        elif args.ndim == 4:
                                print ""
                                print "... constructing 4 dim. KDE (adaptive bandwith)"
                                print ""

                                kde_adapt = AdaptiveKernelDensity("KernelPDF",
                                      args.space,  # Phase space
                                      args.tree,  # Input ntuple
                                      args.var_names[0], args.var_names[1], args.var_names[2], args.var_names[3], # Variables to use
                                      "weight",      # weights
                                      args.nbins[0], args.nbins[1], args.nbins[2], args.nbins[3], # Numbers of bins
                                      args.bws[0], args.bws[1],  args.bws[2], args.bws[3], # Kernel widths 
                                      self.approx_pdf,  # PDF for width scaling
                                      0,  # Approximation PDF (0 for flat approximation)
                                      args.mc_conv  # Sample size for MC convolution (0 for binned convolution)
                                      )

				self.kde = kde_adapt





	def eval_point(self, point):
		l = self.args.ndim
		#l = len(point)
		#if not l==self.args.ndim:
		#	print "dim(point) != dim(phasespace) !!"
		#	sys.exit(1)

		v = std.vector(Double)(l)
		for i in range(l):
			v[i] = point[i]	
			
		return self.kde.density(v) * self.kde_norm		

def factorize_kdes(phsp1, kde1, phsp2, kde2):
	phsp = CombinedPhaseSpace("PhspCombined", phsp1, phsp2)
	factorized_kde = FactorisedDensity("FactPDF", phsp, kde1, kde2)
	return factorized_kde
