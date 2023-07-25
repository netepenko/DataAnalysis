#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:16:50 2023

Script to wait for a file to be available. Then read it, print a command and delete it and continue waiting.

@author: boeglinw
"""

import os.path
import sys
import time


file_path = 'acq_data.data'

while True:
    if  not os.path.exists(file_path):
        print(file_path + ' not yet found waiting !')
        time.sleep(1)
    
    else:
        if os.path.isfile(file_path):
            # read file
            with open(file_path,'r') as fi:
                d = fi.readlines()
                ff = d[0].split()
                cmd = ff[0]; file = ff[1]
                if cmd == 'quit':
                    sys.exit(0)
                print(f'Executing: {cmd} {file}')
                print(f'deleting {file_path}')
                os.remove(file_path)        
        else:
            raise ValueError("%s isn't a file!" % file_path)
