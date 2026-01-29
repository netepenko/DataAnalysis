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
# do this for testing only
OA.create_db(OA.dbfile, force_creation=True)

# this will use the existing DB if it exists
OA.create_db(OA.dbfile, force_creation=False)

#%% add a new shot and its information


# set the time offset default parameter
OA.def_pars.comment='New t_offset value'
OA.def_pars.t_offset = -0.1

#%% important parameter defauls are on OA.def_pars they can be show using OA.def_pars.show_all()


OA.add_shot(OA.dbfile, 
            52433 , 
            'DAQ_52433_250805_133845.hws', 
            rp_pos = 1.75, 
            rp_setpoint=-80., 
            comment = 'Testing analysis', 
            folder = '../MAST-U_data/raw_data_2025/', 
            date='Aug-05-2025')

#%% delete this shot and all its table entries
OA.delete_shot(OA.dbfile, 52433,include_shotlist = True)  


#%%a nalyze the new shot

# setup for new shot
# AS = OA.analyze_shot(30114)
AS = OA.analyze_shot(52433, channels = [0,1,2,3,4,5])
# perform analysis
AS.analyze_all()


