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


db_file = 'full_shot_listDB_nyoko2.db'
db_dir = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/MAST_data/'

cdc.db.DATA_BASE_DIR = db_dir

db = cdc.db


#%% scanning a single channel for data only
shot = 30114; channel = 0

cc = cdc.channel_data(shot, channel, db_file, scan_only=True, Vscan_s = .1, Vscan_th = .3)
cc.read_database_par()
cc.load_data()
# scanning for data only
rf = RFC.raw_fitting(cc, scan_only = True)
print(f'Shot {shot} has potentially useful data in channel {channel} : {rf.has_data}')

cc.plot_raw(ls = '-')


#%% get the entire shot list
shot_list = db.get_shot_list(db_file)

#%%  Check all shots if there potentially contain useful data and store 
#    the channel list in the comment section of the shot list
 
for shot, n_chan in shot_list[-4:]:
    print(f'-------------------------- Shot = {shot} -----------------')
    ok_chan = []
    for channel in  range(n_chan):
        # load data file
        cc = cdc.channel_data(shot, channel, db_file, scan_only=True, Vscan_s = .1, Vscan_th = .3)
        cc.read_database_par()
        cc.load_data()
        # scanning for data only
        rf = RFC.raw_fitting(cc, scan_only = True)
        if rf.has_data:
            ok_chan.append(channel)
        print(f'Shot {shot} has potentially useful data in channel {channel} : {rf.has_data}')
    q_what = f'Comment = "{str(ok_chan)}"'
    q_table = 'Shot_List'
    q_where = f'Shot = {shot}'
    db.writetodb(db_file, q_what, q_table, q_where)
        



#%% Change the directory structure (mwhen moving from one computer to another)

shot_list = db.get_shot_list(db_file)

new_data_dir = '/data2/plasma.1/'

for shot, n_chan in shot_list[:]:
    current = db.get_folder(db_file, shot)
    fields  = current.split('/')
    rel_dir = '/'.join(fields[-3:])
    new_folder = new_data_dir + rel_dir
    db.set_folder(db_file, shot, new_folder)

    
    
    


