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

dbfile = 'New_MainDB1.db'
shot = 29880
channel = 2

ra = rac.rate_analysis(dbfile, shot, channel)
ra.time_slice_data()
ra.make_2d_histo()

B.pl.figure()
ra.plot_results()

#%%
B.pl.figure(figsize = (16,4))

# use keywords vmin, vmax to set z-range
ra.plot_2d(raw = True)

#%% save rate results

ra.save_rates()