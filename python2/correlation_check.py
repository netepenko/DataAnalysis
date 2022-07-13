# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:54:47 2017
Correlation check
@author: Alex
"""

import numpy as np
import h5py
import static_functions as fu
from matplotlib import pyplot as pl

class channel_data():
    #conversion to microseconds constant
    global us
    us = 1.e6

    #initialize the class instance    
    def __init__(self, shot, channel):
                #-------- load parameters-------
        self.par = {} #parameters dictionarry initialization
        self.var = {} #class variables dictionarry
        
        self.par['shot'] = shot
        self.par['channel'] = channel
        
     
            
        (self.par['dtmin'],self.par['dtmax']) = (11000., 125100)
        
       
        
        
        
        
        #--------------------------------                        
        #-------------------------------- 
        ######### Load raw data #########
        
        f = h5py.File('../Raw_Data/29975_DAQ_220813-141746.hws', 'r')
        data_root = 'wfm_group0/traces/trace' + str(self.par['channel']) + '/'
        print "-----------------------Getting data------------------------"
        
        #load time information
        t0 = f[data_root + 'x-axis'].attrs['start']*us
        dt = f[data_root + 'x-axis'].attrs['increment']*us
        
        # load scale coeeff and scale dataset
        scale =f[data_root + 'y-axis/scale_coef'].value
        
        # get the y dataset length
        nall = f[data_root + 'y-axis/data_vector/data'].shape[0]
        
        # make time array based on number of points in y data
        tall = t0 + dt*np.arange(nall, dtype = float) 
        
        # data window for analysis (indices in all data array)
        tds = fu.get_window_slice(self.par['dtmin'], tall, self.par['dtmax'])
        
        # get the y dataset (measured data)
        ydata = f[data_root + 'y-axis/data_vector/data'].value.astype('int16')[tds]
        
        # calculate voltage for dataset
        V = scale[0] + scale[1]*ydata
        print "-----------------------Data loaded-------------------------"
        
        # save data for future use
        self.td = tall[tds]    #time data (microseconds) in analysis interval
        self.Vps = V # voltage data
        self.dt= dt # time step (microseconds)
        
       
        #add pulser to data if add_pulser parameter set to True

if __name__=="__main__":
    ch1=channel_data(29975,0) #for testing purpose
    #ch2=channel_data(29975,1)
    
    #np.place(ch1.Vps, ch1.Vps<0.07, 0)
    #np.place(ch2.Vps, ch2.Vps<0.07, 0)
    
    #pl.plot(ch1.Vps, ch2.Vps, '.')
    #pl.plot(ch1.td, ch1.Vps, '.')
    #pl.plot(ch2.td, ch2.Vps, '.', color='red')
    #c=np.correlate(ch1.Vps, ch2.Vps, 'same')
    #cr=np.correlate(ch1.Vps, np.random.random(len(ch1.Vps)), 'same')
    #print len(c)
    #pl.plot(c)
    #pl.figure()
    #pl.plot(cr)