#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May  6 15:33:27 2022

example of evaluating shots for good data

@author: boeglinw
"""

import numpy as np
import matplotlib.pyplot as pl
import sys

from analysis_modules import database_operations as db

# colored trajectories
import matplotlib.colors as COL
# colors list
color_table = COL.CSS4_COLORS
color_names = list(color_table.keys())

color_base = ['r', 'g', 'b', 'm', 'y', 'c', 'k']


import LT.box as B

us=1.e6 #conversion to microseconds constant

#%%

def get_csv_values(a, data_type = int):
    return np.array(a.split(',')).astype(data_type)

def get_CV(dbfile, q_where, latest = True):  
    # get channel and version  information (use latest version)
    # check if data exist
    db.check_data(dbfile, 'Combined_Rates',  q_where)
    data = db.retrieve(dbfile, 'Channels, Versions', 'Combined_Rates', where = q_where)
    if latest:
        return [get_csv_values(x) for x in data[-1]]
    else:
        return [ [get_csv_values(xx[0]), get_csv_values(xx[1])] for xx in data]
    
def get_file_name(dbfile, shot, channel, version):
    q_where = f'Shot = {shot} AND Channel = {channel} AND Version = {version}'
    db.check_data(dbfile, 'Rate_Analysis',  q_where)
    fn, = db.retrieve(dbfile, 'Result_File_Name', 'Rate_Analysis', where = q_where)[-1]
    return fn

def get_limits(dbfile, q_where, latest = True):
    db.check_data(dbfile, 'Combined_Rates',  q_where)
    data = db.retrieve(dbfile, 't_min, t_max, A_min, A_max', 'Combined_Rates', where = q_where)
    if latest:
        return np.array(data[-1])
    else:
        return np.array(data)
    

#%% data base information
db.DATA_BASE_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/MAST_data/'
dbfile = 'full_shot_listDB.db'


#%% read the data

use_limits = True
shot = 30124

q_where = f'Shot = {shot}'

channels, versions = get_CV(dbfile, q_where)

if channels.shape[0] != versions.shape[0]:
    sys.exit(f'Different numbers of channels {channels.shape[0]} and versions {versions.shape[0]} !')

# get plotting parameters

t_min, t_max, A_min, A_max = get_limits(dbfile, q_where)


# load the rate data

data = []

for i, cn in enumerate(channels):
    vn = versions[i]
    fn = get_file_name(dbfile, shot, cn, vn)
    # read the data
    data.append(np.load(fn))
# all data sets loaded

#%% plot the data
fig = B.pl.figure(figsize = (16,8))

# create a grid

gs = fig.add_gridspec(3, hspace=0)   # 3 rows no height space

axs = gs.subplots(sharex=True, sharey=True)  # share both axes
fig.suptitle(f'Combined Rates for Shot {shot}')


# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

plots = []

for ia, ax in enumerate(axs):
    d = data[ia]
    plots.append(B.plot_exp(d['t'], d['Ap'], d['dAp'], axes = ax, color = color_base[ia], label = f'Channel {channels[ia]}'))
    ax.plot(d['t'], d['Ap'], color = color_base[ia])
    ax.legend()
    
if use_limits:
    B.pl.xlim((t_min , t_max))
    B.pl.ylim((A_min, A_max))
