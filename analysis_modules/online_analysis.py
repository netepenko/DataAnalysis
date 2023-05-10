#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May  6 15:33:27 2022


Script for thort online analysis between shots

@author: boeglinw
"""

import numpy as np
import matplotlib.pyplot as pl
from analysis_modules import channel_data_class as cdc
from analysis_modules import rate_analysis_class as rac
from analysis_modules import peak_sampling_class as PS
from analysis_modules import raw_fitting_class as RFC


import LT.box as B


us = 1e6


#%% This needs to be set globally
cdc.db.DATA_BASE_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/MAST_data/'
dbfile = 'online_DB.db'


#%% add shot to online data base

def add_shot(dbfile, shot, filename, rp_pos = 2., rp_setpoint = 0, t_offset = 0., channels = [0,1,2,3] , first_chan = 0, comment = '[0,1,2,3]', 
             folder = 'MAST/090913/', date = 'Jan-01-2023'):
     # add table entries in data base for new shot
     # add Shot_list entry
     n_chan = len(channels)
     parameters = [f'Shot = {shot}',
                   f'RP_position = {rp_pos}',
                   f'RP_setpoint = {rp_setpoint}',
                   f'File_Name = "{filename}"',
                   f'N_chan = {n_chan}',
                   f'comment = "{comment}"',
                   f't_offset = {t_offset}',
                   f'Date = "{date}"',
                   f'Folder = "{folder}"'
                   ]
     new_par = ','.join(parameters)
     cdc.db.copyrow(dbfile, 'Shot_list', 'Shot = 99999', new_par)
     # add  corresponding entires to
     for i in channels:         
         cdc.db.copyrow(dbfile, 'Peak_Sampling', f'Shot = 99999 AND Channel = {i}', f'Shot = {shot}' )
         cdc.db.copyrow(dbfile, 'Rate_Analysis', f'Shot = 99999 AND Channel = {i}', f'Shot = {shot}' )
         cdc.db.copyrow(dbfile, 'Raw_Fitting', f'Shot = 99999 AND Channel = {i}', f'Shot = {shot}' )
     return


def delete_shot(dbfile, shot):
    # deletes all channel entries for this shot number
    cdc.db.delete_row(dbfile, 'Peak_Sampling', f'Shot = {shot}' )
    cdc.db.delete_row(dbfile, 'Raw_Fitting', f'Shot = {shot}' )
    cdc.db.delete_row(dbfile, 'Rate_Analysis', f'Shot = {shot}' )
    return

#%%
def make_2d_histo(rf,tmin = 0.*cdc.us, 
                  tmax = 1*cdc.us, 
                  dt = 3e-3*cdc.us,
                  hy_min = 0.,
                  hy_max = 2.,
                  hy_bins = 50):


    # 2d histogram setup
    
    # x - axis
    if tmin is None:
        tmin = rf.tp.min()
    if tmax is None:
        tmax = rf.tp.max()
    hx_bins = int((tmax - tmin)/dt) + 1
    
    h_title = f'shot {rf.channel_data.shot}, channel: {rf.channel_data.channel}' 
    
    h2p = B.histo2d(rf.tp, rf.Vp, range = [[tmin,tmax],[hy_min,hy_max]], bins = [hx_bins, hy_bins],
                         title = h_title, xlabel = r't[$\mu$ s]', ylabel = 'raw PH [V]')
    
    return h2p
#%%

def calc_rates(h, Vmin, Vmax):
    rr = [np.array(h.project_y(bins = [i]).sum(Vmin, Vmax))/(h.x_bin_width)*cdc.us  for i in  range(h.nbins_x)]
    return np.vstack( (h.x_bin_center, np.array(rr).T) )
    





#%% normal loading data for analysis  1st pass

# this is for a quick analysis of the data, no fitting is performed. This is usefule to have a look of the data outised of
# using digiplot. Is also makes it possible to add the shot data to a sqlite data base

class analyze_shot:
    def __init__(self, shot, channels = [0,1,2,3], version = 0, dbfile = 'online_DB.db'):
        
        self.shot = shot
        self.channels = channels
        self.version = version
        self.clear_all_data()
        
    def analyze_all(self):
        self.clear_all_data()
        self.find_peaks()
        self.make_histos()
        self.plot_histos()
        self.plot_rates()


    def clear_all_data(self):        
        self.rf_a = []
        self.ra_a = []
        self.h2_a = []
        self.rates_a = []
        
    def find_peaks(self):
        self.rf_a = []
        channels =  self.channels
        for ch in channels:
            cc = cdc.channel_data(self.shot, ch, dbfile, file_type='raw', version = self.version)
            cc.read_database_par()
            cc.load_data()
            rf = RFC.raw_fitting(cc, refine_positions=False, use_refined = False, correct_data = False, fit_progress = 1000)
            rf.find_peaks()
            self.rf_a.append(rf)
    
    def make_histos(self):
        self.h2_a = []
        self.ra_a = []
        self.rates_a = []
        channels =  self.channels
        if self.rf_a == []:
            print('No peak data, need to run find_peaks first !')
            self. find_peaks()
        for i,ch in enumerate(channels):
            rf = self.rf_a[i]                        
            ra = rac.rate_analysis(dbfile, self.shot, ch, version = self.version) # needed for the parameters       
            h2 = make_2d_histo(rf, tmin = None, tmax = None, 
                               hy_min = ra.par['h_min'], 
                               hy_max = ra.par['h_max'],
                               hy_bins = ra.par['h_bins'], 
                               dt = ra.par['time_slice_width']*cdc.us)
            R = calc_rates(h2, ra.par['p_min'], ra.par['p_max'])           
            self.ra_a.append(ra)    
            self.h2_a.append(h2)
            self.rates_a.append(R)
     
    def plot_histos(self):
            
        # plot all 2d histos
        
        fig_2d = B.pl.figure(figsize=(8, 14), constrained_layout=False)
        grid = fig_2d.add_gridspec(len(self.h2_a), 1, wspace=0, hspace=.35)
        axs = grid.subplots(sharex = True)
        
        for i, ax in enumerate(axs):
            no_label = i < len(self.h2_a)-1
            self.h2_a[i].plot(axes = ax, skip_x_label = no_label)
        
        fig_2d.subplots_adjust(top=0.95, bottom = 0.06)

    def plot_rates(self):        
        # plot all rates
        fig_r = B.pl.figure(figsize=(8, 14), constrained_layout=False)
        grid = fig_r.add_gridspec(len(self.rates_a), 1, wspace=0, hspace=.35)
        axs = grid.subplots(sharex = True)
        
        for i, ax in enumerate(axs):
        
            R = self.rates_a[i]
            rf = self.rf_a[i]
            p_title = f'shot {rf.channel_data.shot}, channel: {rf.channel_data.channel}' 
            ax.plot(R[0], R[1])
            ax.fill_between(R[0], R[1] - R[2], R[1]+R[2], alpha = 0.5)
            ax.set_title(p_title)
        ax.set_xlabel(r't ($\mu$s)')
        fig_r.subplots_adjust(top=0.95, bottom = 0.06)
        

        
    #%%
