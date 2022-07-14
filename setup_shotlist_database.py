#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:40:26 2022

Setup a data base with a complete show list

@author: boeglinw
"""
import numpy as np
from analysis_modules import database_operations as db

import LT.box as B
import dateutil as DU
import os

#%%
def get_date(l):
    s = l['file_name']
    # extract the date
    date_str = l['file_name'].split('_')[1].split('-')[0]
    return DU.parser.parse(date_str).strftime('%b-%d-%Y')

def get_dir(l):
    s = l['digitizer_file_name']
    return os.path.split(s)[0] + '/'

#%%
database_dir = '/data2/plasma.1/analysis/'

database_name = 'full_shot_listDB.db'

db_file_name = database_dir + database_name

# create initial database
db.create(db_file_name)

# load shot list

data_dir = '/data2/plasma.1/MAST/'

shot_list = 'updated_shot_list.data'

ds = B.get_file(data_dir + shot_list)

#%% loop over the data and setup the queries

shot_list_fields = ['Shot',         'Date', 'File_Name',' Folder', 'RP_position', 'RP_setpoint', 't_offset', 'N_chan', 'Comment']
shot_list_types =  ['INT not NULL', 'TEXT', 'TEXT',      'TEXT',   'REAL',        'REAL',         'REAL', 'INT',    'TEXT']
shot_list_values = []



for l in ds:
    # create table data
    shot = l['shot_number']
    date_str = "'"+get_date(l) + "'"
    file_name = "'" + l['file_name']+ "'"
    file_dir = "'" + get_dir(l) + "'"
    rp_pos = l['RP_pos']
    rp_setp = l['setpoint']
    
    shot_list_values.append([shot, date_str,  file_name, file_dir, rp_pos, rp_setp, '0', '4', "''"])
    
    
shot_list = db.db_table_data('Shot_List', shot_list_fields, shot_list_types, shot_list_values)
conn = db.lite.connect(db_file_name)
with conn:
    shot_list.insert_into(conn)
conn.close()


