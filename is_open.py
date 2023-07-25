#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:44:51 2023

@author: boeglinw
"""

import numpy as np
import psutil
import os


def is_open(f_name, user_name='boeglinw', process_name = 'python3.10', debug = False):

    open_files = []
    open_dirs = []
    
    
    for proc in psutil.process_iter():
        if (proc.username() == user_name) and (proc.name() == process_name):
            try:
                for of in proc.open_files():
                    dir_name, file_name = os.path.split(of.path)
                    open_files.append(file_name)
                    open_dirs.append(dir_name)
            except Exception as err:
                print(f'---> Cannot get openfiles  reason: {err}  <----')

    if debug:        
        print(70*'-')        
        print(f'open files for {process_name} ')
        print(70*'-')
        for i,of in enumerate(open_files):
            print(f'{open_dirs[i]}/{of}')
        print(70*'-')

    return f_name in open_files
    
