#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 08:49:55 2022

do a quick analysis of a new shot


The database directory can be changed as follows:
    
OA.cdc.db.DATA_BASE_DIR = 'new diretory/'

# to change the default db

OA.dbfile = 'new_dbfile.db'


@author: boeglinw
"""
import datetime as DT
import online_analysis as OA

# get current date
today = DT.datetime.today().strftime('%b-%d-%Y')




#%% add a new shot
OA.add_shot(OA.dbfile, 
            30114 , 
            'DAQ_090913-131415.hws', 
            rp_pos = 1.82, 
            rp_setpoint=478.0, 
            t_offset = 0., 
            n_chan = 4, 
            comment = '[0,1,2,3]', 
            folder = 'MAST/090913/', 
            date='Sep-09-2013')

#%%a nalyze the new shot

# setup for new shot
AS = OA.analyze_shot(30114)
# perform analysis
AS.analyze_all()


