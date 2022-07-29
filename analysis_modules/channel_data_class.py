#
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Code to analyze PD experimental raw data
# Author: Werner Boeglin, December 2014
# Modified: Alexander Netepenko, September 2016
# -------------------------------------------------------------------------
# Modules: raw_fitting.py, database_operations.py, peak_sampling.py
#
# This version uses database MainDB for stored paremeters
#
# This version also:
#     - Uses the fortran fitting module lfitm1, which has a variable order
#       background polynomial
#     - Improves fitting to get a better signal sample as it allows for a
#       DC offset
#     - Re-scales the time axis to microseconds, which is needed for
#       numerical accuracy
#     - Sets times relative to the beginning of a fit slice (or the middle)
#
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

import numpy as np
import h5py  # HDF file handling
import os
import sys
from . import database_operations as db
from . import utilities as UT
from . import data_plotting

# convert needs to be set true only for data wich was written in int16 format
# while declared as float (mistake in LabView acquisition code lead to this confusion)
convert = True
# --------------------------------
"""
Main class to load data. If a version number is not given the latest version of parameters from 
raw_fitting etc. are bing used

scan_only = True is used to scan the data and see if there are porntially valid data in this channel
no parameters are loaded from the other database tables 
"""
# --------------------------------
# conversion to microseconds constant
us = 1.e6

to_bool = {'True':True, 'False':False}

# sllowable file types

class channel_data():

    # initialize the class instance
    def __init__(self, shot, channel, db_file, 
                 version = None,
                 data_version = None,
                 file_type = 'raw',
                 iteration = None,
                 result_root = None, 
                 scan_only = False, 
                 Vscan_s = 0.1, 
                 Vscan_th = 0.15):
        """
        load data for a digitizer channel and setup the necessary parameters for further analysis

        Parameters
        ----------
        shot : TYPE
            Shot Number
        channel : TYPE
            DESCRIPTION.
        db_file : database file with the parameters
            DESCRIPTION.
        version : TYPE, optional
            analysis version number. The default is None.
        data_version : TYPE, optional
            analysis version number for the corrected data file to be used. The default is None.
        file_type : TYPE, optional
            channel data file type: raw, corrected, filtered . The default is 'raw'.
        iteration : TYPE, optional
            iteration number of corrected data file. The default is None.
        result_root : TYPE, optional
            rood directory for the analysis results. The default is None.
        scan_only : TYPE, optional
            on scan the data no real analysis. The default is False.
        Vscan_s : TYPE, optional
            V_step for scanning peak finding. The default is 0.1.
        Vscan_th : TYPE, optional
            V thershold for scanning peak finding. The default is 0.15.

        Returns
        -------
        Object

        """
        # dictionary of load data functions
        self.load_data_dict = {'raw':self.load_raw_data,
                          'corrected':self.load_hdf_data,
                          'filtered':self.load_npz_data}
        file_types = list(self.load_data_dict.keys())
        if not file_type in file_types:
            print(f'Unknown file type {file_type} allowable values are {file_types}')
            return None
        else:
            self.file_type = file_type
        
        # if version is not specified take the highest one
        self.par = {}  # parameters dictionary initialization
        self.var = {}  # class variables dictionarry
        if scan_only:
            version = 0
            self.par['Vstep'] = Vscan_s
            self.par['Vth'] = Vscan_th
            self.par['max_neg_V'] = -0.3
            self.par['min_delta_t'] = 3e-7
        elif version is None:
            wheredb =  f'Shot = {shot} AND Channel = {channel}'
            e_msg = f'No data for shot {shot} and  channel {channel} in Rate Analysis'
            db.check_data(db_file, 'Raw_Fitting', wheredb, e_msg)
            (version, ) = db.retrieve(db_file, 'Version','Raw_Fitting', wheredb)[-1]   
        if data_version is None:
            self.data_version = version
        else:
            self.data_version = data_version
        self.shot = shot
        self.scan_only = scan_only
        self.shot_str = f'{self.shot:d}'
        self.channel = channel
        self.db_file = db_file
        self.version = version
        self.iteration = iteration
        self.wheredb = f'Shot = {shot:d} AND Channel = {channel:d}'
        self.wheredb_version = self.wheredb + f' AND Version = {self.version:d}'
        self.wheredb_data_version = self.wheredb + f' AND Version = {self.data_version:d}'
        # frequently used string to retrieve data from database that indicates shot and chennel

        self.par['shot'] = shot
        self.par['channel'] = channel
        self.par['version'] = version
        self.psize = 5
        self.N_pulser_events = 0
        if result_root is None:
            self.par['result_root'] = './Analysis_Results/'
        else:
            self.par['result_root'] = result_root

    def load_data(self, **kwargs):
        self.load_data_dict[self.file_type](**kwargs)

    def read_database_par(self):
        # read from database Raw_Fitting table interval limits for analysis
        # if DB doesnt contain parameters for selected shot and channel copy
        # them from another channel or even another shot
        shot = self.shot_str
        wheredb = self.wheredb
        wheredb_version = self.wheredb_version
        wheredb_data_version = self.wheredb_data_version
        dbfile = self.db_file
        which_shot = f'Shot = {shot}'
        # extract data from Shot_list
        
        e_msg = f'table Shot_List does not contain data for {wheredb}'
        db.check_data(dbfile, 'Shot_List' , which_shot, e_msg)

        (self.par['root_dir'],) = db.retrieve(dbfile,  'Root_Folder', 'Common_Parameters')[0]
        self.par['exp_dir'], self.par['exp_file'] = db.retrieve(dbfile,  'Folder, File_Name', 'Shot_List', which_shot)[0]
        (t_offset,) = db.retrieve(dbfile,  't_offset', 'Shot_List', which_shot)[0]
        self.par['t_offset'] = t_offset*us
        

        # for scanning only no other parameters are needed
        if self.scan_only:
            return

        # extract data from Raw_Fitting'
        
        db.check_data(dbfile,'Raw_Fitting' , wheredb_version)
        if not db.check_condition(dbfile, 'Raw_Fitting' , wheredb_version):
            print(f'table Raw_Fitting does not contain data for {wheredb_version}')
            return -1       
        
        # update the file name if corrected data are to be used
        if self.file_type == 'corrected':
            (file_name,) = db.retrieve(dbfile, 'Corrected_Data_File_Name', 'Raw_Fitting', wheredb_data_version)[0]
            if file_name is None:
                sys.exit(f'No corrected file name found in Raw_Fitting for {wheredb_data_version}')
            f_dir, f_name = os.path.split(file_name)
            self.par['exp_file'] = f_name
            # remove root_dir from file name
            self.par['exp_dir'] = f_dir.replace(self.par['root_dir'], '') + '/'
        
        
        self.par['dtmin'], self.par['dtmax'] = np.asarray(db.retrieve(dbfile, 'dtmin, dtmax', 'Raw_Fitting', wheredb_version)[0])*us
        # read other parameters

        self.par['poly_order'], self.par['n_peaks_to_fit'] = db.retrieve(dbfile, 'poly_order, n_peaks_to_fit', 'Raw_Fitting', wheredb_version)[0]
        add_pulser, self.par['pulser_rate'], self.par['P_amp'] = db.retrieve(dbfile,  'add_pulser, pulser_rate, P_amp', 'Raw_Fitting', wheredb_version)[0]
        self.par['add_pulser'] = to_bool[add_pulser]
        use_threshold, self.par['Vth'], self.par['Vstep'] = db.retrieve(dbfile,  'use_threshold, Vth, Vstep', 'Raw_Fitting', wheredb_version)[0]
        self.par['use_threshold'] = to_bool[use_threshold]
        self.par['min_delta_t'], self.par['max_neg_V'] = db.retrieve(dbfile,  'min_delta_t, max_neg_V', 'Raw_Fitting', wheredb_version)[0]

        # laod parameters for finding peaks
        self.par['n_sig_low'], self.par['n_sig_high'], self.par['n_sig_boundary'] = db.retrieve(dbfile, 'n_sig_low, n_sig_high, n_sig_boundary', 'Raw_Fitting', wheredb_version)[0]

        # n_sig_high not used anywhere
        sig, = db.retrieve(dbfile, 'sig', 'Raw_Fitting', wheredb_version)[0]
        self.par['sig'] = sig*us

        # read peak shape parameters from database Peak_Sampling table
        
        e_msg = f'table Peak_Sampling does not contain data for {wheredb_version}'
        db.check_data(dbfile, 'Peak_Sampling' , wheredb_version, e_msg)
        
        decay_time, rise_time, n_samp, position = db.retrieve(dbfile, 'decay_time, rise_time, n_samp, position', 'Peak_Sampling', wheredb_version)[0]
        self.par['decay_time'] = decay_time*us  # converted to microseconds
        self.par['rise_time'] = rise_time*us  # converted to microseconds
        self.par['n_samp'] = n_samp
        self.par['position'] = position*us
        
        (Vstep, Vth, Chi2, tmin, tmax) = db.retrieve(dbfile, 'Vstep, Vth, Chi2, tmin, tmax', 'Peak_Sampling', wheredb_version)[0]
        
        self.par['ps_Vstep'] = Vstep
        self.par['ps_Vth'] = Vth
        self.par['Chi2'] = Chi2
        self.par['ps_tmin'] = tmin*us
        self.par['ps_tmax'] = tmax*us
        
        # order for background fit and set vary codes variables
        self.var['vary_codes_bkg'] = (self.par['poly_order'] + 1)*[1]
        self.var['bkg_len'] = len(self.var['vary_codes_bkg'])
        self.var['peak_num'] = 1  # running number for peak selection
        self.var['data_plot'] = None

        # assign directories for results (edit later for good tree structure!!)
        self.var['res_dir'] = self.par['result_root'] + shot + '/Raw_Fitting/'
        print('Analysis results will be placed in: ', self.var['res_dir'])
        return 0

    
    def load_raw_data(self, **kwargs):
        # --------------------------------
        # ######## Load raw data #########
        self.data_filename =  self.par['root_dir'] + self.par['exp_dir'] + self.par['exp_file']
        f = h5py.File(self.data_filename, 'r')
        # setup reading the data
        data_root = 'wfm_group0/traces/trace' + str(self.par['channel']) + '/'

        print("-----------------------Getting data------------------------")

        # load time information
        t0 = f[data_root + 'x-axis'].attrs['start']*us + self.par['t_offset']
        dt = f[data_root + 'x-axis'].attrs['increment']*us
        # load scale coeeff and scale dataset
        scale = f[data_root + 'y-axis/scale_coef'][()]

        # get the y dataset length
        nall = f[data_root + 'y-axis/data_vector/data'].shape[0]

        # make time array based on number of points in y data
        tall = t0 + dt*np.arange(nall, dtype=float)

        # data window for analysis (indices in all data array)
        #tds = fu.get_window_slice(self.par['dtmin'], tall, self.par['dtmax'])

        # get the y dataset (measured data)
        if convert:
            ydata = f[data_root + 'y-axis/data_vector/data'][()].astype('int16')
        else:
            ydata = f[data_root + 'y-axis/data_vector/data'][()]

        # calculate voltage for dataset
        V = scale[0] + scale[1]*ydata
        print("-----------------------Data loaded-------------------------")

        # save data for future use
        self.td = tall  # time data (microseconds) in analysis interval
        self.Vps = V  # voltage data
        self.dt = dt  # time step (microseconds)

        # testing of fitting pulser
#        self.td = self.td[0:1000000]
#        self.Vps = np.zeros_like(self.td)
        if self.scan_only:
            self.par['dtmin'] = self.td.min()
            self.par['dtmax'] = self.td.max()
            return
        # add pulser to data if add_pulser parameter set to True
        if self.par['add_pulser']:
            self.add_pulser()

#   plotting of raw data without overloading the figure
# (skipping some data points according to maximum allowed points on plot)

    def load_npz_data(self, file_name = None):
        # --------------------------------
        # ######## Load raw data #########
        if file_name is None:
            self.data_filename =  self.par['root_dir'] + self.par['exp_dir'] + self.par['exp_file']
        else:
            self.data_filename = file_name
        self.f = np.load(self.data_filename)
        d = self.f
        print("--------------------- Get npz data ------------------------")
        self.td = d['time']*us
        self.Vps = d['signal']
        self.dt = d['time'][1] - d['time'][0]
        print("-----------------------Data loaded-------------------------")
        # add pulser to data if add_pulser parameter set to True
        if self.par['add_pulser']:
            self.add_pulser()

    def load_hdf_data(self, file_name = None):
        # --------------------------------
        # ######## Load raw data #########
        if file_name is None:
            self.data_filename =  self.par['root_dir'] + self.par['exp_dir'] + self.par['exp_file']
        else:
            self.data_filename = file_name
        print("--------------------- Get hdf data ------------------------")
        f = h5py.File(self.data_filename, 'r')
        # load the data set for corrected data
        Vc = f['V_corr']
        scale = Vc.attrs['V_corr_scale']
        Vd = Vc[:]/scale
        dt = Vc.attrs['dt']
        t0 = Vc.attrs['t0']
        td = t0 + dt*np.arange(Vd.shape[0], dtype=float)
        self.Vps = Vd
        self.dt = dt
        self.td = td
        print("-----------------------Data loaded-------------------------")
        # add pulser to data if add_pulser parameter set to True
        if self.par['add_pulser']:
            self.add_pulser()

#   plotting of raw data without overloading the figure
# (skipping some data points according to maximum allowed points on plot)
    def plot_raw(self, xmin=None, xmax=None, **kwargs):

        V = self.Vps
        t = self.td

        if xmin and xmax:
            interval = np.where((xmin < t) & (t < xmax))
            t = t[interval]
            V = V[interval]
        else:
            interval = np.where((self.par['dtmin'] < t) & (t < self.par['dtmax']))
            t = t[interval]
            V = V[interval]            
        data_plotting.plot_data(t, V, **kwargs)

# add pulser signals to check fit performance
    def add_pulser(self):
        dtmax = self.par['dtmax']
        dtmin = self.par['dtmin']
        n_samp = self.par['n_samp']
        V = self.Vps
        td = self.td
        ts = td[0:n_samp]-td[0]

        # number of pulser events
        Delta_t = (dtmax - dtmin)
        N_events = int(self.par['pulser_rate'] * Delta_t/us)
        self.N_pulser_events = N_events

        try:
            Vtotal = self.Vtotal
        except:
            Vtotal = UT.peak(ts - self.par['position'],
                             1/self.par['decay_time'],
                             1/self.par['rise_time'])[1]
        # Vtotal array of averaged and sampled signals
        # generate random times according to the rate and the time interval
        t_pulse = np.random.uniform(size=N_events)*Delta_t
        t_pulse.sort()   # sort the pulse times
        print(f'---> Added {t_pulse.size} random simulated pulses <---')
        # add the pulses to the signals
        for t_pu in t_pulse:
            Vtotal = UT.peak(ts - self.par['position'],
                             1/self.par['decay_time'],
                             1/self.par['rise_time'])[1]
            # get the slice of t corresponding to current pulse
            sl = UT.get_window_slice(t_pu, td, t_pu + ts[-1])
            # add signals
            if (V[sl].shape[0] < Vtotal.shape[0]):
                V[sl] += Vtotal[:V[sl].shape[0]]*self.par['P_amp']
            elif (V[sl].shape[0] > Vtotal.shape[0]):
                sl1 = slice(sl.start, sl.start + Vtotal.shape[0])
                V[sl1] += Vtotal*self.par['P_amp']
            else:
                V[sl] += Vtotal*self.par['P_amp']
        self.Vps = V
        # db.writetodb('add_pulser = "True"', 'Raw_Fitting',
        # 'Shot = '+str(self.par['shot'])+' AND Channel = '
        # +str(self.par['channel']))
        print('Pulser signal added to data.')


