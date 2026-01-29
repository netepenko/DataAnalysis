#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 12:13:44 2026

teest db operations

@author: boeglinw
"""

import numpy as np
import importlib as IL
from analysis_modules import database_operations as DO

import os

#%% reload DO
IL.reload(DO)



#%%
db_file = 'New_MainDB1.db'

#%% delete DB file if needed
try:
    os.remove(db_file)
except:
    print(f'{db_file} not found, already deleted ?')

#%%
DO.create(db_file)


#%% query and substitution commands  for tables without versions

where_db_cp = ('Shot = 99999')
sub_db_cp_new = ('Shot = 30102, Signal_channels = "[10,20]" ')

DO.copy_row(db_file, 'Shot_List', where_db_cp, sub_db_cp_new)

#%% query and substitution commands  for tables with versions

where_db_cp = ('Shot = 99999 AND Channel = 0')
sub_db_cp_new_version = ('Shot = 29978, Channel = 3, Version = 5')

where_db_cp_new = ('Shot = 29978 AND Channel = 3')
sub_db_cp_new = ('Shot = 29978, Channel = 3')

#%%
DO.find_version(where_db_cp_new, False)


#%%
DO.copy_row(db_file, 'Peak_Sampling', where_db_cp, sub_db_cp_new)
#%%
DO.copy_row(db_file, 'Peak_Sampling', where_db_cp, sub_db_cp_new)

#%%
DO.copy_row(db_file, 'Peak_Sampling', where_db_cp, sub_db_cp_new_version)

#%% 
DO.copy_row(db_file, 'Peak_Sampling', where_db_cp_new, sub_db_cp_new)

#%% create entry for a number of channels
DO.copy_row(db_file, 'Peak_Sampling', 'Shot = 99999 AND Channel = 0 AND Version = 0', 'Shot = 55555,Version = 0, Channel = 0')
DO.copy_row(db_file, 'Peak_Sampling', 'Shot = 99999 AND Channel = 0 AND Version = 0', 'Shot = 55555,Version = 0, Channel = 1')
DO.copy_row(db_file, 'Peak_Sampling', 'Shot = 99999 AND Channel = 0 AND Version = 0', 'Shot = 55555,Version = 0, Channel = 2')
DO.copy_row(db_file, 'Peak_Sampling', 'Shot = 99999 AND Channel = 0 AND Version = 0', 'Shot = 55555,Version = 0, Channel = 3')
DO.copy_row(db_file, 'Peak_Sampling', 'Shot = 99999 AND Channel = 0 AND Version = 0', 'Shot = 55555,Version = 0, Channel = 4')
DO.copy_row(db_file, 'Peak_Sampling', 'Shot = 99999 AND Channel = 0 AND Version = 0', 'Shot = 55555,Version = 0, Channel = 5')