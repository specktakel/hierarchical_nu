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
    
    def get_norm_spline_Edet(self):
        """
        Return a spline which is normalised along the Edet axis.
        Also no longer logged.
        """

        self.log_enu_axis = np.linspace(3.0, 7.0, 500)
        self.log_edep_axis = np.linspace(1.0, 7.0, 500)

        # Calculated normalised distribution
        self.dist_val = np.zeros((len(self.log_enu_axis), len(self.log_edep_axis)))
        for j, edep in enumerate(self.log_edep_axis):

            # Normalise for this Edep
            prob = []
            for _ in self.log_enu_axis:
                prob.append(10**self.spline(_, edep)[0][0])
            norm = np.trapz(prob, self.log_enu_axis)

            # Get dist value
            for i, enu in enumerate(self.log_enu_axis):
                self.dist_val[i][j] = 10**self.spline(enu, edep)[0][0] / norm

            # Fit a spline
            self.norm_spline = RectBivariateSpline(self.log_enu_axis, self.log_edep_axis, self.dist_val,
                                                   s=0, kx=1, ky=1) # spline degree = 1!
            
                
            
