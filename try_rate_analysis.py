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

ra = rac.rate_analysis(dbfile, shot, channel)
ra.time_slice_data()
ra.make_2d_histo()

#%%

B.pl.figure()
ra.plot_results()

#%%
B.pl.figure(figsize = (16,4))

# use keywords vmin, vmax to set z-range
ra.plot_2d(raw = True)

#%% save rate results

ra.save_rates()

#%% save the entire object (only for testing)

ra.save_myself()