#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 07:44:02 2024

Script to set link for data analysis

@author: boeglinw
"""

import os
import sys
import errno


# helper function to create symbolic link, do not touch!

def make_link(src,dst, overwrite = False):
    # create symbolic links
    try:
        os.symlink(src,dst)
        print("created symbolic link for : ", src)
        return None
    except Exception as e:
        if e.errno == errno.EEXIST:
            if not overwrite:
                # link exists, do not overwrite
                print(f'{dst} exists, will use it')
                return None
            else:
                # remove old link
                print('Will replace old link with new one')
                os.remove(dst)
                # create new link
                os.symlink(src,dst)
                print("created symbolic link for : ", src)
                return None
        else:
            print(f'Cannot create link: {e}')
            return e
    


#%% edit here and set to your location
    
analysis_modules_dir = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/DataAnalysis/'


#%%
# make link to analysis_modules
make_link(analysis_modules_dir+'analysis_modules', './analysis_modules')

# make link to db_edit (editor for data base)
make_link(analysis_modules_dir+'db_edit', './db_edit')