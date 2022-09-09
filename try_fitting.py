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
cdc.db.DATA_BASE_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/MAST_data/'
dbfile = 'full_shot_listDB.db'

#%% data range
cdc.db.DATA_BASE_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/DataAnalysis/'
dbfile = 'New_MainDB1.db'


#%% duplicate rows for further analysis
"""
shot = 30124; channel = 1

q_where = f'Shot = {shot} AND Channel = {channel}'

cdc.db.duplicate_row(dbfile, 'Peak_Sampling', q_where )
cdc.db.duplicate_row(dbfile, 'Rate_Analysis', q_where)
cdc.db.duplicate_row(dbfile, 'Raw_Fitting', q_where)
"""

#%% normal loading data for analysis  1st pass

shot = 30124; channel = 1; version = 0
cc = cdc.channel_data(shot, channel, dbfile, file_type='raw', version = version)
cc.read_database_par()
cc.load_data()


#%% second pass

shot = 30124; channel = 1; version = 1; data_version = 0; iteration = 0

cc = cdc.channel_data(shot, channel, dbfile, file_type='corrected', version = version, data_version = data_version, iteration = iteration)
cc.read_database_par()
cc.load_data()

#%% filtered data
"""
cc.load_npz_data(file_name='DAQ_190813-112521_filtered.npz')
cc.td *= cdc.us
cc.dt *= cdc.us
"""
#%% plot data range

cc.plot_raw(ls = '-')


#%% get sample peaks

ps = PS.peak_sampling(cc, plot_single_peaks = False, plot_common_peak = True)
ps.find_good_peaks()

ps.fit_peaks(save_fit = True)
ps.save_parameters(cc.db_file)


#%% ready for raw fitting

rf = RFC.raw_fitting(cc, refine_positions=True, use_refined = True, correct_data = True, fit_progress = 1000)
rf.find_peaks()     # N_close = 2 is default


#%%
rf.setup_fit_groups()
rf.init_fitting()
print(f'Created {rf.fg.shape[0]} fit groups')
#%%
B.pl.figure()

# find a fit group for a certain time in us
ng = rf.find_fitgroup(342400)

rf.plot_fit_group(ng, shifted = False, warnings = True)
rf.plot_fit_group(ng, shifted = True, warnings = True)

#%%
rf.fit_data()

#%%
rf.save_fit()

#%% 

rf.save_corr(keep_iteration=False)