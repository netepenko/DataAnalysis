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
import database_operations as db
import static_functions as fu
import raw_fitting
import peak_sampling
from my_plot import my_plot

convert = True
# --------------------------------
# --------------------------------


class channel_data():
    # conversion to microseconds constant
    global us
    us = 1.e6

    # initialize the class instance
    def __init__(self, shot, channel):
        def read_database_par():
            # read from database Raw_Fitting table interval limits for analysis
            # if DB doesnt contain parameters for selected shot and channel copy
            # them from another channel or even another shot
    
            
            (self.par['dtmin'], self.par['dtmax']) = np.asarray(
                    db.retrieve('dtmin, dtmax', 'Raw_Fitting', wheredb))*us
            # read other parameters
            
            (self.par['poly_order'], self.par['n_peaks_to_fit']) = db.retrieve(
                    'poly_order, n_peaks_to_fit', 'Raw_Fitting', wheredb)
            (self.par['add_pulser'], self.par['pulser_rate'],
             self.par['P_amp']) = db.retrieve(
                     'add_pulser, pulser_rate, P_amp', 'Raw_Fitting', wheredb)
            (self.par['use_threshold'], self.par['Vstep'],
             self.par['Vth']) = db.retrieve(
                     'use_threshold, Vth, Vstep', 'Raw_Fitting', wheredb)
    
            # laod parameters for finding peaks
            (self.par['n_sig_low'], self.par['n_sig_high'],
             self.par['n_sig_boundary']) = db.retrieve(
                     'n_sig_low, n_sig_high, n_sig_boundary',
                     'Raw_Fitting', wheredb)
    
            # n_sig_high not used anywhere
            self.par['sig'] = db.retrieve('sig', 'Raw_Fitting', wheredb)[0]*us
    
            # read peak shape parameters from database Peak_Sampling table
            (decay_time, rise_time) = db.retrieve(
                    'decay_time, rise_time', 'Peak_Sampling', wheredb)
            self.par['decay_time'] = decay_time*us  # converted to microseconds
            self.par['rise_time'] = rise_time*us  # converted to microseconds
            #self.par['position'] = position*us  # converted to microseconds
    
           
        # -------- load parameters-------
        self.par = {}  # parameters dictionarry initialization
        self.var = {}  # class variables dictionarry

        self.par['shot'] = shot
        self.par['channel'] = channel

        

        wheredb = 'Shot = ' + str(shot) + ' AND Channel = ' + str(channel)
        # frequently used string to retrieve data from database
        # that indicates shot and chennel
        
        try:
            (self.par['exp_dir'], self.par['exp_file']) = db.retrieve(
                    'Folder, File_Name', 'Shot_List', 'Shot = ' + str(shot))
        except:
            print "Given shot is not found in the shot list or some other problem occured!"
            return

        
        try:
            read_database_par()

        except:
            print "Couldn't find parameters for Channel ", channel
            shot_cp = self.par['shot']   
            ch_cp = self.par['channel'] - 1  # copy param from prev channel
            
            wheredb_cp = ('Shot = ' + str(shot_cp) +
                          ' AND Channel = ' + str(ch_cp))
            
            try:
                
                db.copyrow('Raw_Fitting', wheredb_cp, 'Shot = ' + str(shot) +
                           ', Channel = ' + str(channel))
                db.copyrow('Peak_Sampling', wheredb_cp, 'Shot = ' + str(shot) +
                           ', Channel = ' + str(channel))
                db.copyrow('Rates_Plotting', wheredb_cp,
                           'Shot = ' + str(shot) +
                           ', Channel = ' + str(channel))
                read_database_par()
                print 'Coppied parameters from previous channel'
            except:
                print "Couldn't copy paramateres from previous channel, will try the previous shot!"
                #try to copy from previous shot in shotlist table
                try:
                    shot_cp = db.prevshot(self.par['shot'])   # copy param from previous shot
                    ch_cp = self.par['channel']  # copy param from prev channel
                    wheredb_cp = ('Shot = ' + str(shot_cp) +
                          ' AND Channel = ' + str(ch_cp))
                    db.copyrow('Raw_Fitting', wheredb_cp, 'Shot = ' + str(shot) +
                           ', Channel = ' + str(channel))
                    db.copyrow('Peak_Sampling', wheredb_cp, 'Shot = ' + str(shot) +
                           ', Channel = ' + str(channel))
                    db.copyrow('Rates_Plotting', wheredb_cp,
                               'Shot = ' + str(shot) +
                               ', Channel = ' + str(channel))
                    read_database_par()
                    print 'Coppied parameters from previous shot.'
                except:
                    print "Couldn't copy parameters from previous shot. Input parameters manually in DB"
                    return

        
        #  ------------assign class variables -------------
        # order for background fit and set vary codes variables
        self.var['vary_codes_bkg'] = (self.par['poly_order'] + 1)*[1]
        self.var['bkg_len'] = len(self.var['vary_codes_bkg'])
        self.var['peak_num'] = 1  # running number for peak selection
        self.var['data_plot'] = None
        # assign directories for results (edit later for good tree structure!!)
        self.var['res_dir'] = ('../Analysis_Results/' +
                               str(shot) + '/Raw_Fitting/')
        print 'Analysis results will be placed in: ', self.var['res_dir']

        # --------------------------------
        # --------------------------------
        # ######## Load raw data #########

        f = h5py.File(self.par['exp_dir'] + self.par['exp_file'], 'r')
        data_root = 'wfm_group0/traces/trace' + str(self.par['channel']) + '/'
        print "-----------------------Getting data------------------------"

        # load time information
        t0 = f[data_root + 'x-axis'].attrs['start']*us
        dt = f[data_root + 'x-axis'].attrs['increment']*us
        # load scale coeeff and scale dataset
        scale = f[data_root + 'y-axis/scale_coef'].value

        # get the y dataset length
        nall = f[data_root + 'y-axis/data_vector/data'].shape[0]
        # make time array based on number of points in y data
        tall = t0 + dt*np.arange(nall, dtype=float)

#        # data window for analysis (indices in all data array)
#        tds = fu.get_window_slice(self.par['dtmin'], tall, self.par['dtmax'])

        # get the y dataset (measured data)
        if convert:
            ydata = f[data_root + 'y-axis/data_vector/data'].value.astype('int16')
        else:
            ydata = f[data_root + 'y-axis/data_vector/data'].value

        # calculate voltage for dataset
        V = scale[0] + scale[1]*ydata
        print "-----------------------Data loaded-------------------------"

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
    

    def plot_raw(self, xmin=None, xmax=None):

        V = self.Vps
        t = self.td
        
        if xmin and xmax:
            interval = np.where((xmin < t) & (t < xmax))
            t = t[interval]
            V = V[interval]

        my_plot(t, V, '.', color='blue')

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
            Vtotal = fu.peak(ts - self.par['position'],
                             1/self.par['decay_time'],
                             1/self.par['rise_time'])[1]
        # Vtotal array of averaged and sampled signals
        # generate random times according to the rate and the time interval
        t_pulse = np.random.uniform(size=N_events)*Delta_t
        t_pulse.sort()   # sort the pulse times
        # add the pulses to the signals
        for t_pu in t_pulse:
            Vtotal = fu.peak(ts - self.par['position'],
                             1/self.par['decay_time'],
                             1/self.par['rise_time'])[1]
            # get the slice of t corresponding to current pulse
            sl = fu.get_window_slice(t_pu, td, t_pu + ts[-1])
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
        print 'Pulser signal added to data.'

# pulling methods from separate scripts, to make files shorter and easier to edit
channel_data.find_good_peaks = peak_sampling.find_good_peaks
channel_data.fit_interval = raw_fitting.fit_interval
channel_data.load_peaks = peak_sampling.load_peaks
channel_data.fit_shape = peak_sampling.fit_shape

# for testing purpose if runing this file separately
if __name__ == "__main__":
    ch1 = channel_data(29975, 1)  # for testing purpose
