#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May  6 15:33:27 2022

@author: boeglinw
"""

import numpy as np
import matplotlib.pyplot as pl
from analysis_modules import channel_data_class as cdc
from analysis_modules import rate_analysis_class as rac

from analysis_modules import database_operations as db

import LT.box as B

us=1.e6 #conversion to microseconds constant

#%% data range
cdc.db.DATA_BASE_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/MAST_data/'
dbfile = 'full_shot_listDB.db'


#%% data range
cdc.db.DATA_BASE_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/DataAnalysis/'
dbfile = 'New_MainDB1.db'

#%%
shot = 30121
channel = 3
version = 0

"""
# for channel 2 comparisons
data_dir = f'./Analysis_Results/{shot}/Raw_Fitting/'
file_name = 'fit_results_30121_2_0_0.000_0.500_21_07_2022_12_50_14.npz'
file_name = 'fit_results_30121_2_0_0.000_0.500_21_07_2022_12_55_11.npz'

ra = rac.rate_analysis(dbfile, shot, channel, Afit_file=data_dir + file_name)
"""


"""
# for channel 1 comparisons
data_dir = f'./Analysis_Results/{shot}/Raw_Fitting/'
file_name = 'fit_results_30121_1_0_0.000_0.500_21_07_2022_13_14_28.npz'
#file_name = 'fit_results_30121_1_0_0.000_0.500_21_07_2022_13_17_07.npz'
#file_name = 'fit_results_30121_1_0_0.000_0.500_21_07_2022_13_19_25.npz'

ra = rac.rate_analysis(dbfile, shot, channel, Afit_file=data_dir + file_name)
"""

ra = rac.rate_analysis(dbfile, shot, channel)

ra.time_slice_data()
ra.make_2d_histo()

#%%

B.pl.figure()
ra.plot_results()

#%%
B.pl.figure(figsize = (16,4))
# raw data
# use keywords vmin, vmax to set z-range
ra.plot_2d(raw = True)


# fitted data
B.pl.figure(figsize = (16,4))
# use keywords vmin, vmax to set z-range
ra.plot_2d(raw = False)

#%% save rate results

ra.save_rates()

#%% save the entire object (only for testing)

ra.save_myself()