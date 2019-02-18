#!/usr/bin/env python

class aeff_info(object):
    def __init__(self): 

        self.nue_cc = './aeff_input/effective_area.per_bin.nu_e.cc.cascade.txt'
        self.nue_cc_smoothing = 1

        self.nue_nc = './aeff_input/effective_area.per_bin.nu_e.nc.cascade.txt'
        self.nue_nc_smoothing = 5

        self.nue_gr = './aeff_input/effective_area.per_bin.nu_e_bar.gr.cascade.txt'
        self.nue_gr_smoothing = 15

        self.numu_cc = './aeff_input/effective_area.per_bin.nu_mu.cc.cascade.txt'
        self.numu_cc_smoothing = 150

        self.numu_nc = './aeff_input/effective_area.per_bin.nu_mu.nc.cascade.txt'
        self.numu_nc_smoothing = 300

        self.nutau_cc = './aeff_input/effective_area.per_bin.nu_tau.cc.cascade.txt'
        self.nutau_cc_smoothing = 5

        self.nutau_nc = './aeff_input/effective_area.per_bin.nu_tau.nc.cascade.txt'
        self.nutau_nc_smoothing = 30 

        '''
        self.nue_cc = '../aeff_input/effective_area.per_bin.nu_e.cc.cascade.txt'
        self.nue_cc_smoothing = 1

        self.nue_nc = '../aeff_input/effective_area.per_bin.nu_e.nc.cascade.txt'
        self.nue_nc_smoothing = 5

        self.nue_gr = '../aeff_input/effective_area.per_bin.nu_e_bar.gr.cascade.txt'
        self.nue_gr_smoothing = 15

        self.numu_cc = '../aeff_input/effective_area.per_bin.nu_mu.cc.cascade.txt'
        self.numu_cc_smoothing = 150

        self.numu_nc = '../aeff_input/effective_area.per_bin.nu_mu.nc.cascade.txt'
        self.numu_nc_smoothing = 300

        self.nutau_cc = '../aeff_input/effective_area.per_bin.nu_tau.cc.cascade.txt'
        self.nutau_cc_smoothing = 5

        self.nutau_nc = '../aeff_input/effective_area.per_bin.nu_tau.nc.cascade.txt'
        self.nutau_nc_smoothing = 30
        '''


        self.infiles = {'nue_NC': self.nue_nc, 'nue_CC': self.nue_cc, 'numu_CC': self.numu_nc, 'numu_CC': self.numu_cc, 'nutau_CC': self.nutau_nc, 'nutau_CC': self.nutau_cc, 'nue_GR': self.nue_gr}
        self.smoothing = {'nue_NC': self.nue_nc_smoothing, 'nue_CC': self.nue_cc_smoothing, 'numu_NC': self.numu_nc_smoothing, 'numu_CC': self.numu_cc_smoothing, 'nutau_NC': self.nutau_nc_smoothing, 'nutau_CC': self.nue_cc_smoothing, 'nue_GR': self.nue_gr_smoothing }
