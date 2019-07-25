#!/usr/bin/env python

from .aeff_info import aeff_info
from scipy.interpolate import RectBivariateSpline 
from scipy import ndimage
import numpy as np
import pandas as pd
import h5py

class effective_area_cascades(object):
    def __init__(self, interaction):
        '''
        interaction: "flavor_channel", e.g. "nue_nc" or "numu_cc"
        '''

        d = aeff_info()
        self.infile = d.infiles[interaction]
        self.smoothing = d.smoothing[interaction]
        self.interaction = interaction

        # define lower bound on effective area
        self.lower_bound_lE = -5.0
        self.lower_bound_hE = -3.0

        # will set effective area to zero outside (negative infinity)
        self.lE_limit_low = 3.0
        self.lE_limit_high = 7.0

        self.spline = None

        self.lE_vals = None
        self.ct_vals = None
        self.aeff_vals = None

        self.__create_spline__()

    def eval(self, lE, cos_zenith):
        '''
        return log10(Aeff) in m
        '''
        if not isinstance(lE, np.ndarray):
            if lE < self.lE_limit_low or lE > self.lE_limit_high:
                print( "energy", lE, "outside bounds. returning neg. infty.")
                return np.NINF

            elif cos_zenith < -1.0 or cos_zenith > 1.0:
                print ("cos zenith", cos_zenith, "outside bounds. returning neg. infty.")
                return np.NINF

            else:
                return self.spline(lE, cos_zenith)


        else:
            # evaluate on grid. don't check bounds.
            return self.spline(lE, cos_zenith)


    def __create_spline__(self):

        table = np.loadtxt(self.infile, skiprows=2)

        enu_low = np.log10(table[:,0])
        enu_high = np.log10(table[:,1])

        ctnu_low = table[:,2]
        ctnu_high = table[:,3]

        self.ctnu_low = ctnu_low
        self.ctnu_high = ctnu_high
        
        log_enu_binc = 0.5*(enu_low + enu_high)
        log_enu_binc = log_enu_binc.reshape((len(log_enu_binc), 1))

        ctnu_binc = 0.5*(ctnu_low + ctnu_high)
        ctnu_binc = ctnu_binc.reshape((len(ctnu_binc), 1))
        
        table = np.append(table, log_enu_binc, axis=1)
        table = np.append(table, ctnu_binc, axis=1)

        aeff_arr = []
        logE_arr = []
        ct_arr = []
				
		# enforce bounds
        xvals = np.unique(table[:,-2])
        idx = xvals < self.lE_limit_low
        k = len(xvals[idx])-2
        xvals = xvals[k:]
        idx = xvals < self.lE_limit_high
        k = len(xvals[idx])+2
        xvals = xvals[:k] 
        yvals = np.unique(table[:,-1])
		

		# collect effective area from all observable bins
        for logE in xvals:
            for ct in yvals:
                idx = np.logical_and(table[:,-1]==ct, table[:,-2]==logE)
                asum = np.sum(table[idx][:,-4])
                if asum:
                    aeff = np.log10(np.sum(table[idx][:,-4]))
                else:
                    aeff = np.NINF
                
                if np.isinf(aeff):
                    if logE < 5.0:
                        aeff=self.lower_bound_lE
                    else:
                        aeff=self.lower_bound_hE
            
               
                aeff_arr.append(aeff)
                logE_arr.append(logE)
                ct_arr.append(ct)

        self.lE_vals = np.array(logE_arr)
        self.ct_vals = np.array(ct_arr)
        self.aeff_arr = np.array(aeff_arr)

        self.aeff_vals = self.aeff_arr
       


		# now we average over bins. i.e. map 2 bins into 1
        xvals_avg = (0.5*(xvals[1:]+xvals[:-1]))[::2] # only every second
        aeff_smooth = np.ones(len(xvals_avg)*len(yvals))
        aeff_smooth = aeff_smooth.reshape((len(xvals_avg), len(yvals)))
	
        
		# average along energy axis
        aeff_arr_shaped = np.asarray(aeff_arr).reshape((len(xvals), len(yvals)))
        for column_idx in range(len(yvals)):
            aeff_avg = aeff_arr_shaped[:, column_idx]
            aeff_avg = (0.5*(aeff_avg[1:]+aeff_avg[:-1]))[::2]
            aeff_smooth[:, column_idx]=np.array(aeff_avg)
        
        self.spline = RectBivariateSpline(xvals_avg, yvals, aeff_smooth, s=self.smoothing) 
        print( "... creating effective area spline for interaction", self.interaction, ", done!")


class effective_area_tracks(object):

    def __init__(self):

        # Data release suggested by Christian and Lisa
        
        with h5py.File("aeff_input_tracks/effective_area.h5", 'r') as f:
            area10 = f['2010/nu_mu/area'][()]
            self.lE_bin_edges = np.log10(f['2010/nu_mu/bin_edges_0'][()]) # Energy [GeV]
            self.cosz_bin_edges = f['2010/nu_mu/bin_edges_1'][()] # cos(zenith)

        self.aeff_vals = np.sum(area10, axis=2)
        self.lE_bin_cen = (self.lE_bin_edges[:-1] + self.lE_bin_edges[1:])/2
        self.cosz_bin_cen = (self.cosz_bin_edges[:-1] + self.cosz_bin_edges[1:])/2

        self.lE_limit_low = self.lE_bin_edges[0]
        self.lE_limit_high = self.lE_bin_edges[-1]

        self.cosz_limit_low = self.cosz_bin_edges[0]
        self.cosz_limit_high = self.cosz_bin_edges[-1]

        # More recent Aeff info

        """
        filename = 'aeff_input_tracks/IC79-2010-TabulatedAeff.txt'
        filelayout = ['Emin', 'Emax', 'cos(z)min', 'cos(z)max', 'Aeff']
        output = pd.read_csv(filename, comment = '#',
                     delim_whitespace = True,
                     names = filelayout)

        output_dict = output.to_dict()
        Emin = list(output_dict['Emin'].values())
        Emax = list(output_dict['Emax'].values())
        coszmin = list(output_dict['cos(z)min'].values())
        coszmax = list(output_dict['cos(z)max'].values())
        aeff = list(output_dict['Aeff'].values())
        self.aeff_vals = np.reshape(aeff, (70, 200))
        
        # find bin centres
        Emin = np.sort(list(set(Emin)))
        Emax = np.sort(list(set(Emax)))
        self.lE_bin_cen = np.log10((Emin + Emax)/2)

        coszmin = np.sort(list(set(coszmin)))
        coszmax = np.sort(list(set(coszmax)))
        self.cosz_bin_cen = (coszmin + coszmax)/2

        # find min/max
        self.lE_limit_low = np.log10(min(Emin))
        self.lE_limit_high = np.log10(max(Emax))
        self.cosz_limit_low = min(coszmin)
        self.cosz_limit_high = max(coszmax)
        """
        
        self.__create_spline__()
        
    def eval(self, lE, cos_zenith):
        '''
        return log10(Aeff) in m
        '''

        if not isinstance(lE, np.ndarray):

            if lE < self.lE_limit_low or lE > self.lE_limit_high:
                return [[np.NINF]]
            elif cos_zenith < self.cosz_limit_low or cos_zenith > self.cosz_limit_high:
                return [[np.NINF]]
            else:
                return self.spline(lE, cos_zenith)

        else:
            return self.spline(lE, cos_zenith)

		
    def __create_spline__(self):

        # Smooth
        sigma = [0.8, 0.5]
        self.aeff_vals_nonzero = self.aeff_vals
        self.aeff_vals_nonzero[self.aeff_vals_nonzero == 0] = 1e-10
        self.log10_aeff_smooth = ndimage.filters.gaussian_filter(np.log10(self.aeff_vals_nonzero), sigma, mode='constant')
        
        # Make spline
        self.spline = RectBivariateSpline(self.lE_bin_cen, self.cosz_bin_cen,
                                          self.log10_aeff_smooth, s=0.0)
        # find maximum
        x = np.linspace(self.lE_limit_low, 7.0, 500)
        y = np.linspace(self.cosz_limit_low, self.cosz_limit_high, 500)
        xx, yy = np.meshgrid(x, y)
        xx = xx.T
        yy = yy.T
        z = self.eval(x, y)

        self.log10_aeff_max = np.max(z)
        
        


        

       
