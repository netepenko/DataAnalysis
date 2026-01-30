#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May  6 15:33:27 2022


Collection of tools for automated shot analysis, typically for online analysis. Used in combination with the online_analyze_new_shot.py

@author: boeglinw
"""

import numpy as np
import matplotlib.pyplot as pl
from analysis_modules import channel_data_class as cdc
from analysis_modules import rate_analysis_class as rac
from analysis_modules import peak_sampling_class as PS
from analysis_modules import raw_fitting_class as RFC

import os

import LT.box as B


us = 1e6


#%% make a string representation of a list

def list_to_str(l):
    return '['+','.join([f'{v}' for v in l])+']'


#%% This needs to be set globally, in the script using these tools

cdc.db.DATA_BASE_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/MAST_data/'
dbfile = 'online_DB.db'

def create_db(dbfile, force_creation = False):
    # create DB is it has not already been created
    db_path = cdc.db.DATA_BASE_DIR + dbfile
    db_exists = os.path.exists(db_path)
    if db_exists and force_creation:
        print(f're-create: {db_path}, was forced to do this ')
        try:
            os.remove(db_path)
        except Exception as err:
            print(f'cannot remove {db_path} , already deleted ? Error message : {err}')
        cdc.db.create(db_path)
    elif db_exists:
        print(f'{db_path} exists already, use force_creation=True to force overwrite')
    else:
        cdc.db.create(db_path)

#%%
class default_parameters:
    
    def __init__(self, reference_shot = 99999,
                 rp_pos = 2., 
                 rp_setpoint = 0, 
                 t_offset = 0., 
                 first_chan = 0, 
                 signal_channels = [0,1,2,3,4,5],
                 noise_channels = [6,7],
                 detector_numbers = [1,2,3,4,5,6],
                 digitizer_channels = [1,2,3,4,5,6,7,8],
                 bias_voltages = [45,45,45,45,45,45],
                 comment ='No comment',
                 folder = 'MAST/090913/', 
                 date = 'Jan-01-2023',
                 # V_step for peak sampling
                 V_step_ps =  0.02,
                 # Threshold for peak sampling
                 V_th_ps = 0.05,
                 #time range for peak sampling
                 t_min_ps = 0.1,
                 t_max_ps = 0.25,
                 # V_step for peak finding
                 V_step = 0.05,
                 # theshold for peak finding
                 V_th = 0.05, 
                 # time range for peak finding
                 t_min = 0.1,
                 t_max = 1.0,
                 # proton signal selection
                 p_min = 0.1,
                 p_max = 0.5):
        
        """
        default parameters for database tables used in analysis. The following tables
        are using these:
            
            Shot_List
            Peak_Sampling
            Raw_Fitting
            Rate_Analysis
    
        Parameters
        ----------

        reference_shot : int, optional
            shot number of prototype shot used in generating table entries. 
        rp_pos : float, optional
            RP postion . 
        rp_setpoint : float, optional
            RP position set point. 
        t_offset : float, optional
            time offset in s. 
        first_chan : int, optional
            first channel number. 
        signal_channels : list, optional
            list of signal channels numbers.
        noise_channels : list, optional
            list of noise channel numbers. 
        detector_numbers : list, optional
            list of detector numbers. 
        digitizer_channels : list, optional
            list of digitizer channels (as labelled on the device). 
        bias_voltages : list, optional
            list of detector bias voltages used. 
        comment : str, optional
            comments for this entry. 
        folder : str, optional
            location of data file directory. 
        date : str, optional
            date of data or entry. 
        V_step_ps : float, optional
            voltage step size for peak finding in peak sampling
        V_th_ps : float, optional
            threshold for peak sampling
        t_min_ps : float, optional
            start time for peak finding in peak sampling
        t_max_ps : float, optional
            end time for peak finding in peak sampling
        V_step : float, optional
            voltage step size for peak finding in raw fitting
        V_th : float, optional
            threshold for raw fitting
        t_min : float, optional
            start time for raw fitting
        t_max : float, optional
            end time for raw fitting
        p_min : float, optional
            lower limit for proton signals used in rate analysis
        p_max : float, optional     
            upper limit for proton signals used in rate analysis
        """
        
        self.reference_shot = reference_shot
        self.rp_pos = rp_pos
        self.rp_setpoint = rp_setpoint 
        self.t_offset = t_offset 
        self.first_chan = first_chan 
        self.signal_channels = signal_channels
        self.noise_channels = noise_channels
        self.detector_numbers = detector_numbers
        self.digitizer_channels = digitizer_channels
        self.bias_voltages = bias_voltages
        self.comment = comment
        self.folder = folder 
        self.date = date
        
        self.V_step = V_step
        self.V_th = V_th 
        self.t_min =t_min
        self.t_max = t_max
        
        self.V_step_ps = V_step_ps
        self.V_th_ps = V_th_ps 
        self.t_min_ps =t_min_ps
        self.t_max_ps = t_max_ps
        
        self.p_min = p_min
        self.p_max = p_max
        
    def show_all(self):
        line = 70*'-'
        print(line)
        print("Current Default Values")
        print(line)
        for k, v in self.__dict__.items():
            print(f'{k}: {v}')   
        print(line)

#%%
# create instance

def_pars =  default_parameters() 

#%% add shot to online data base
"""
defaults = def_pars,
             reference_shot = def_pars.reference_shot,
             rp_pos = def_pars.rp_pos, 
             rp_setpoint = def_pars.rp_setpoint, 
             t_offset = def_pars.t_offset, 
             first_chan = def_pars.first_chan, 
             signal_channels = def_pars.signal_channels,
             noise_channels = def_pars.noise_channels,
             detector_numbers = def_pars.detector_numbers,
             digitizer_channels = def_pars.digitizer_channels,
             bias_voltages = def_pars.bias_voltages,
             comment = def_pars.comment,
             folder = def_pars.folder, 
             date = def_pars.date
"""



def add_shot(dbfile, shot, filename, defaults = def_pars, **kwargs):
    """
    add a shot to the data base and set its default parameters in tables for analysis. The following tables
    are updated:
        
        Shot_List
        Peak_Sampling
        Raw_Fitting
        Rate_Analysis

    Parameters
    ----------
    dbfile : TYPE str
        data base file path
    shot : int
        shot number
    filename : str
        digitizer data file name
    defaults : call default_parameters instance, optional
        class instance containing the default parameters. The default is def_pars.
    reference_shot : int, optional
        shot number of prototype shot used in generating table entries. The default is def_pars.reference_shot.
    rp_pos : float, optional
        RP postion . The default is def_pars.rp_pos.
    rp_setpoint : float, optional
        RP position set point. The default is def_pars.rp_setpoint.
    t_offset : float, optional
        time offset in s. The default is def_pars.t_offset.
    first_chan : int, optional
        first channel number. The default is def_pars.first_chan.
    signal_channels : list, optional
        list of signal channels numbers. The default is def_pars.signal_channels.
    noise_channels : list, optional
        list of noise channel numbers. The default is def_pars.noise_channels.
    detector_numbers : list, optional
        list of detector numbers. The default is def_pars.detector_numbers.
    digitizer_channels : list, optional
        list of digitizer channels (as labelled on the device). The default is def_pars.digitizer_channels.
    bias_voltages : list, optional
        list of detector bias voltages used. The default is def_pars.bias_voltages.
    comment : str, optional
        comments for this entry. The default is def_pars.comment.
    folder : str, optional
        location of data file directory. The default is def_pars.folder.
    date : str, optional
        date of data or entry. The default is def_pars.date.

    Returns
    -------
    None.

    """

    # add table entries in data base for new shot
    defaults.show_all()
    
    # initialize kwarguments using default values
    KW_args = {}
    KW_args['reference_shot'] = defaults.reference_shot
    KW_args['rp_pos'] = defaults.rp_pos
    KW_args['rp_setpoint'] = defaults.rp_setpoint
    KW_args['t_offset'] = defaults.t_offset
    KW_args['first_chan'] = defaults.first_chan
    KW_args['signal_channels'] = defaults.signal_channels
    KW_args['noise_channels'] = defaults.noise_channels
    KW_args['detector_numbers'] = defaults.detector_numbers
    KW_args['digitizer_channels'] = defaults.digitizer_channels
    KW_args['bias_voltages'] = defaults.bias_voltages
    KW_args['comment'] = defaults.comment
    KW_args['folder'] = defaults.folder
    KW_args['date'] = defaults.date
    
    
    # update with keywords given in argument list
    
    for k in kwargs.keys():
       KW_args[k] = kwargs[k]
    
    
    # add Shot_list entry
    n_chan = len(KW_args['signal_channels'])
    parameters = [f'Shot = {shot}',
                  'RP_position = '+f'{KW_args["rp_pos"]}',
                  f'RP_setpoint = {KW_args["rp_setpoint"]}',
                  f'File_Name = "{filename}"',
                  f'N_chan = {n_chan}',
                  f't_offset = {KW_args["t_offset"]}',
                  f'Date = "{KW_args["date"]}"',
                  f'Folder = "{KW_args["folder"]}"',
                  f'Signal_channels = "{list_to_str(KW_args["signal_channels"])}"',
                  f'Noise_channels = "{list_to_str(KW_args["noise_channels"])}"',
                  f'Detector_numbers = "{list_to_str(KW_args["detector_numbers"])}"',
                  f'Bias_voltages = "{list_to_str(KW_args["bias_voltages"])}"',
                  f'Comment = "{KW_args["comment"]}"'
                  ]
    new_par = ','.join(parameters)
    # check iof shot exists
    if cdc.db.check_condition(dbfile, 'Shot_List', f'Shot = {shot}'):
        print(f'===>> {shot} already exists in Shot_List, nothing added')
    else:
        cdc.db.copy_row(dbfile, 'Shot_list', f'Shot = {KW_args["reference_shot"]}', new_par)
    # add  corresponding default entires into data base
    for i in KW_args['signal_channels']:   
        where_cp =  f'Shot = {KW_args["reference_shot"]} AND Channel = 0 AND Version = 0'
        sub_cp_ps = f'Shot = {shot}, Channel = {i}, Version = 0, Vstep = {defaults.V_step_ps}, Vth = {defaults.V_th_ps}, tmin = {defaults.t_min_ps}, tmax = {defaults.t_max_ps}'
        sub_cp_rf = f'Shot = {shot}, Channel = {i}, Version = 0, Vstep = {defaults.V_step}, Vth = {defaults.V_th}, dtmin = {defaults.t_min}, dtmax = {defaults.t_max}' 
        sub_cp_ra = f'Shot = {shot}, Channel = {i}, Version = 0, p_min = {defaults.p_min}, p_max = {defaults.p_max}'
        
        cdc.db.copy_row(dbfile, 'Peak_Sampling', where_cp, sub_cp_ps )
        cdc.db.copy_row(dbfile, 'Rate_Analysis', where_cp, sub_cp_ra )
        cdc.db.copy_row(dbfile, 'Raw_Fitting'  , where_cp, sub_cp_rf )
    # remove unneeded reference shots    
    clear_reference_shots(dbfile, KW_args["reference_shot"])

def clear_reference_shots(dbfile, reference_shot):
    cdc.db.delete_row(dbfile, 'Peak_Sampling', f'Shot = {reference_shot} AND Version > 0' )
    cdc.db.delete_row(dbfile, 'Raw_Fitting', f'Shot = {reference_shot} AND Version > 0' )
    cdc.db.delete_row(dbfile, 'Rate_Analysis', f'Shot = {reference_shot} AND Version > 0' )
    

def delete_shot(dbfile, shot, include_shotlist = False):
    # deletes all channel entries for this shot number
    cdc.db.delete_row(dbfile, 'Peak_Sampling', f'Shot = {shot}' )
    cdc.db.delete_row(dbfile, 'Raw_Fitting', f'Shot = {shot}' )
    cdc.db.delete_row(dbfile, 'Rate_Analysis', f'Shot = {shot}' )
    if include_shotlist:
        # also delete shot list entry
        cdc.db.delete_row(dbfile, 'Shot_List', f'Shot = {shot}' )
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
