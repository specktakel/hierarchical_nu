#!/usr/bin/env python

class interaction_source_files(object):
    def __init__(self):
        self.indir = "./csms_input/"
        self.nu_NC_infile = self.indir+"xy_nu_NC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat"
        self.nu_CC_infile = self.indir+"xy_nu_CC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat"
        self.anti_nu_NC_infile = self.indir+"xy_nubar_NC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat"
        self.anti_nu_CC_infile = self.indir+"xy_nubar_CC_iso_NLO_HERAPDF1.5NLO_EIG_xy.dat"
        
        self.itype={'nubar_NC': self.anti_nu_NC_infile, 'nubar_CC': self.anti_nu_CC_infile, 'nu_CC': self.nu_CC_infile, 'nu_NC': self.nu_NC_infile}
