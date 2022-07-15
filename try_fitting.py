#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May  6 15:33:27 2022


@author: boeglinw
"""

import numpy as np
import matplotlib.pyplot as pl
from analysis_modules import channel_data_class as cdc
from analysis_modules import peak_sampling_class as PS
from analysis_modules import raw_fitting_class as RFC


import LT.box as B

us = 1e6


#%%
cc = cdc.channel_data(29880, 2, 'New_MainDB1.db', version = 0)
cc.read_database_par()
cc.load_data()
"""
cc.load_npz_data(file_name='DAQ_190813-112521_filtered.npz')
cc.td *= cdc.us
cc.dt *= cdc.us
"""
#%% plot data range

cc.plot_raw()


#%% get sample peaks

ps = PS.peak_sampling(cc, plot_single_peaks = False, plot_common_peak = True)
ps.find_good_peaks()

ps.fit_peaks(save_fit = True)
ps.save_parameters(cc.db_file)

#%% ready for raw fitting

rf = RFC.raw_fitting(cc, refine_positions=True)
rf.fit_progress = 1000
rf.find_peaks(N_close = 2)
rf.check_cov = False

#%%
rf.use_refined = True

rf.setup_fit_groups()
rf.init_fitting()
print(f'Created {rf.fg.shape[0]} fit groups')
#%%

ng = 1000
rf.plot_fit_group(ng, shifted = False, warnings = True)
rf.plot_fit_group(ng, shifted = True, warnings = True)

#%%
rf.fit_data()

#%%
rf.save_fit(new_row = True)
