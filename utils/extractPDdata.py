#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:42:21 2026

Extract raw digitizer data from UDA

@author: aaboutal
"""
import numpy as np
import pyuda
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path 
from numpy import savetxt


# Digitizer signal channel numbers taken for 'single', 'duo', 'quad', and 'octal' modes
channels  = [[0] ,  [0,1],  [0,1,2,3],  [0,1,2,3,4,5,6,7]] 

# Digitizer Mode:
Dig_mode = {"single":0, "dual":1, "quad":2, "octal":3}


#%% default values
# setup destination directory
dest_dir_default = '/home/bwerner/MAST-U_data/MU05/'
# shot_num = 54316

#%% funceion to load data from UDA

def get_PDdata(shot_num, mode = "quad", dest_dir = dest_dir_default):
    """
    Extract proton detector (PD) channel data and time data from UDA
    The data for each channel are stored in compressed npz files.
    
    Example: - destination directory (relative to home): ./MAST-U_data/MU05/
             - script location (relative to home): ./DataAnalysis/utils/
             - shot number : 54316
             
    Python console: current working directory is home:
        
             >>> from DataAnalysis.utils import extractPDdata as EPD
             >>> EPD.get_PDdata(54316, mode = "quad", dest_dir = "./MAST-U_data/MU05/")


    for Shot_List entires into the analysis data base use the following form to
    generate the proper file names when analyzing: PD_channel{}_{}.npz
    
    

    Parameters
    ----------
    shot_num : int
        shot number.
    mode : string, optional
        digitizer mode. The default is "quad". Possible values are: single, dual, quad, octal
    dest_dir : string, optional
        destination directory. The default is dest_dir_default, dedined above

    Returns
    -------
    None.

    """
    # check if directory exists if not create it
    Path.mkdir(dest_dir, parents = True, exist_ok = True)
    # get the data
    print(f"setup UDA client for shot : {shot_num}")
    # setup pyuda client
    client = pyuda.Client()
    PD_xpd = client.get('/XPD/DATA', shot_num) 
    print(f"get time data from UDA for shot: {shot_num}")
    tp = PD_xpd.time.data
    # prepare to load the data
    idx = Dig_mode[mode]      # pick which channel group
    print(f"Digitizer-Mode: {mode}")
    
    for ch in channels[idx]:
        Vp = PD_xpd.data[ch]
        print("Channel:", ch)
        #print(Vp)
        output_file = dest_dir + f'PD_channel{ch:d}_{shot_num:d}.npz'
        print(f'---> saving channel {ch} in file {output_file} <---')
        # write compressed npz file
        np.savez_compressed(output_file,time=tp,signal= Vp)

  




