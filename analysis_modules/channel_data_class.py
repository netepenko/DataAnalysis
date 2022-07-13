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
from . import database_operations as db
from . import utilities as UT
from . import data_plotting

# convert needs to be set true only for data wich was written in int16 format
# while declared as float (mistake in LabView acquisition code lead to this confusion)
convert = True
# --------------------------------
# --------------------------------
# conversion to microseconds constant
us = 1.e6

class channel_data():


    # initialize the class instance
    def __init__(self, shot, channel, db_file, Version = 0):
        self.shot = shot
        self.shot_str = f'{self.shot:d}'
        self.channel = channel
        self.db_file = db_file
        self.Version = Version
        self.wheredb = f'Shot = {shot:d} AND Channel = {channel:d}'
        self.wheredb_version = self.wheredb + f' AND Version = {self.Version:d}'
        # frequently used string to retrieve data from database that indicates shot and chennel
        self.par = {}  # parameters dictionary initialization
        self.var = {}  # class variables dictionarry
        self.par['shot'] = shot
        self.par['channel'] = channel
        self.psize = 5


    def read_database_par(self):
        # read from database Raw_Fitting table interval limits for analysis
        # if DB doesnt contain parameters for selected shot and channel copy
        # them from another channel or even another shot
        shot = self.shot_str
        wheredb = self.wheredb
        wheredb_version = self.wheredb_version
        dbfile = self.db_file
        which_shot = f'Shot = {shot}'
        # extract data from Shot_list
        
        if not db.check_condition(dbfile, 'Shot_List' , which_shot):
            print(f'table Shot_List does not contain data for {wheredb}')
            return -1
        self.par['exp_dir'], self.par['exp_file'] = db.retrieve(dbfile,  'Folder, File_Name', 'Shot_List', which_shot)[0]

        # extract data from Raw_Fitting'
        
        if not db.check_condition(dbfile, 'Raw_Fitting' , wheredb_version):
            print(f'table Raw_Fitting does not contain data for {wheredb_version}')
            return -1       

        self.par['dtmin'], self.par['dtmax'] = np.asarray(db.retrieve(dbfile, 'dtmin, dtmax', 'Raw_Fitting', wheredb_version)[0])*us
        # read other parameters

        self.par['poly_order'], self.par['n_peaks_to_fit'] = db.retrieve(dbfile, 'poly_order, n_peaks_to_fit', 'Raw_Fitting', wheredb_version)[0]
        self.par['add_pulser'], self.par['pulser_rate'], self.par['P_amp'] = db.retrieve(dbfile,  'add_pulser, pulser_rate, P_amp', 'Raw_Fitting', wheredb_version)[0]
        self.par['use_threshold'], self.par['Vstep'], self.par['Vth'] = db.retrieve(dbfile,  'use_threshold, Vth, Vstep', 'Raw_Fitting', wheredb_version)[0]
        self.par['min_delta_t'], self.par['max_neg_V'] = db.retrieve(dbfile,  'min_delta_t, max_neg_V', 'Raw_Fitting', wheredb_version)[0]

        # laod parameters for finding peaks
        self.par['n_sig_low'], self.par['n_sig_high'], self.par['n_sig_boundary'] = db.retrieve(dbfile, 'n_sig_low, n_sig_high, n_sig_boundary', 'Raw_Fitting', wheredb_version)[0]

        # n_sig_high not used anywhere
        sig, = db.retrieve(dbfile, 'sig', 'Raw_Fitting', wheredb_version)[0]
        self.par['sig'] = sig*us

        # read peak shape parameters from database Peak_Sampling table
        if not db.check_condition(dbfile, 'Peak_Sampling' , wheredb_version):
            print(f'table Peak_Sampling does not contain data for {wheredb_version}')
            return -1     
        
        decay_time, rise_time = db.retrieve(dbfile, 'decay_time, rise_time', 'Peak_Sampling', wheredb_version)[0]
        self.par['decay_time'] = decay_time*us  # converted to microseconds
        self.par['rise_time'] = rise_time*us  # converted to microseconds
        
        # order for background fit and set vary codes variables
        self.var['vary_codes_bkg'] = (self.par['poly_order'] + 1)*[1]
        self.var['bkg_len'] = len(self.var['vary_codes_bkg'])
        self.var['peak_num'] = 1  # running number for peak selection
        self.var['data_plot'] = None

        # assign directories for results (edit later for good tree structure!!)
        self.var['res_dir'] = './Analysis_Results/' + shot + '/Raw_Fitting/'
        print('Analysis results will be placed in: ', self.var['res_dir'])
        return 0

    def load_data(self):
        # --------------------------------
        # ######## Load raw data #########
        self.data_filename =  self.par['exp_dir'] + self.par['exp_file']
        f = h5py.File(self.data_filename, 'r')
        # setup reading the data
        data_root = 'wfm_group0/traces/trace' + str(self.par['channel']) + '/'

        print("-----------------------Getting data------------------------")

        # load time information
        t0 = f[data_root + 'x-axis'].attrs['start']*us
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

        # add pulser to data if add_pulser parameter set to True
        if self.par['add_pulser'] == 'True':
            self.add_pulser()

#   plotting of raw data without overloading the figure
# (skipping some data points according to maximum allowed points on plot)

    def load_npz_data(self, file_name = None):
        # --------------------------------
        # ######## Load raw data #########
        if file_name is None:
            self.data_filename =  self.par['exp_dir'] + self.par['exp_file']
        else:
            self.data_filename = file_name
        self.f = np.load(self.data_filename)
        d = self.f
        print("--------------------- Get npz data ------------------------")
        self.td = d['time']
        self.Vps = d['signal']
        self.dt = d['time'][1] - d['time'][0]
        print("-----------------------Data loaded-------------------------")
        # add pulser to data if add_pulser parameter set to True
        if self.par['add_pulser'] == 'True':
            self.add_pulser()

#   plotting of raw data without overloading the figure
# (skipping some data points according to maximum allowed points on plot)
    def plot_raw(self, xmin=None, xmax=None):

        V = self.Vps
        t = self.td

        if xmin and xmax:
            interval = np.where((xmin < t) & (t < xmax))
            t = t[interval]
            V = V[interval]

        data_plotting.plot_data(t, V, '.', color='blue')

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


