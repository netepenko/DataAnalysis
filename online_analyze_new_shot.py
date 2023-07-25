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
from analysis_modules import online_analysis as OA

# get current date
today = DT.datetime.today().strftime('%b-%d-%Y')


#%%


OA.cdc.db.DATA_BASE_DIR = './'

OA.dbfile = 'new_online_dbfile.db'

#%% create database if not existing already

OA.create_db(OA.dbfile, force_creation=True)

#%% add a new shot to have something in the new DB
OA.add_shot(OA.dbfile, 
            30114 , 
            'DAQ_090913-131415.hws', 
            rp_pos = 1.82, 
            rp_setpoint=478.0, 
            t_offset = 0., 
            channels = [0,1,2,3], 
            comment = '[0,1,2,3]', 
            folder = 'MAST/090913/', 
            date='Sep-09-2013')



#%% add a new shot
OA.add_shot(OA.dbfile, 
            29912 , 
            'DAQ_200813-121826.hws', 
            rp_pos = 1.82, 
            rp_setpoint=479.0, 
            t_offset = 0., 
            channels = [1,2,3], 
            comment = '[1,2,3]', 
            folder = 'MAST/082013/', 
            date='Aug-20-2013')

#%%a nalyze the new shot

# setup for new shot
# AS = OA.analyze_shot(30114)
AS = OA.analyze_shot(29912, channels = [1,2,3])
# perform analysis
AS.analyze_all()


