import numpy as np
from scipy.interpolate import RectBivariateSpline


class EnergyConditional(object):
    """
    Simple class to load files output from
    cascade_model code.

    @author Francesca Capel
    @date April 2019
    """

    def __init__(self, filename):

        dat = np.loadtxt(filename)

        log_enu = dat[:,0]
        log_edep = dat[:,1]
        
        self.log_enu_axis = sorted(np.unique(log_enu))
        self.log_edep_axis = sorted(np.unique(log_edep))

        orig_cond_pdf = dat[:,2]
        cond_pdf = []
        for val in orig_cond_pdf:
            if val:
                cond_pdf.append(np.log10(val))
            else:
                cond_pdf.append(-20.)
        
        cond_pdf = np.asarray(cond_pdf)
        cond_pdf = cond_pdf.reshape(len(self.log_enu_axis), len(self.log_edep_axis))
        self.cond_pdf = cond_pdf.T

        self.spline = RectBivariateSpline(self.log_enu_axis, self.log_edep_axis, self.cond_pdf, 
                                         s=0, kx=1, ky=1) # spline degree = 1!
    
