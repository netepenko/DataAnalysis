#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:45:48 2023

Adapt mesured nosise to data and subtract from signal data

all times are in us

@author: boeglinw
"""

import numpy as np

from . import correlation_lags as CL

from scipy import signal


def shift_slice(sl, shift):
    return slice(sl.start+shift, sl.stop+shift) 

def is_even(x):
    return x%2 == 0


def moving_average(x, m):
    # calculate the moving average for M points, based on convolution
    # setup the

    if is_even(m) :
        m +=1
        print (f'you need an odd number of kernel points, new value m = {m}')
    kern = np.ones(m)/m
    xc = np.convolve(x, kern)[:x.shape[0]]
    return np.roll(xc, -m//2 + 1)




h_line = 70*'-'
us = 1e6


class bkg:
    
    def __init__(self,
                 moving_average = 7, \
                 niter = 1,\
                 time_window = 200.,\
                     ):
        """
        setup background data for subtraction

        Parameters
        ----------
        moving_average: int
            window size in data points for moving average (odd number), The default is 21
        niter: int
            number of moving average iterations. The default is 5
        time_window: float
            time window width (in us) used to perform the correction. The default is 200

        Returns
        -------
        None.

        """
        self.moving_avg = moving_average
        self.niter = niter
        self.delta_t = time_window
        
        
    def moving_average(self, dd):
        """
        calculate the moving average and replace the original data with the averaged one. This is performed for the data as
        weel as the background data

        Parameters
        ----------
        dd : channel data class instance
            current data to be averaged.

        Returns
        -------
        None.

        """
        if dd.Vps is None:
            print('moving average: no data loaded, nothing to do !')
            return
        if dd.Vps_bkg is None:
            print('moving average:  no data bkg loaded, nothing to do !')
            return
        # calculate the moving averages for data and bkg
        print(h_line)
        print('Start moving average for data:')
        for i in range(self.niter):
            dd.Vps = moving_average(dd.Vps, self.moving_avg)
        print('Finished moving average for data:')
        print('Start moving average for bkg:')
        for i in range(self.niter):
            dd.Vps_bkg = moving_average(dd.Vps_bkg, self.moving_avg)
        print('Finished moving average for bkg:')

    def correct(self, dd):
        """
        Correct the data using the bkg data. Determined a series of slices and determine the phse-shoft 
        between data and background. Shift the background data and determine the overall scale factor. Then subtrace the 
        scaled bkg. data from the original data

        Parameters
        ----------
        dd : Channel data class instance.
            data set for the selected channel.

        Returns
        -------
        None.

        """
        if (dd.Vps is None):
            print(' no data loaded, nothing to do !')
            return        
        if (dd.Vps_bkg is None):
            print(' no bkg data loaded, nothing to do !')
            return
        #loop over all data and subtract noise
        #dn = self.d_bkg
        # create an arrau of slices
        n_slice = int(self.delta_t//dd.dt)
        
        i_start = np.arange(0, dd.td.shape[0], n_slice)  # starting index
        i_end = np.roll(i_start, -1)                        # stopping index
        
        # create the slice array
        slices = [slice(ss,ee) for ss, ee in zip(i_start, i_end)][:-1]  # skip the last slice
        
        V_corr = np.zeros_like(dd.Vps)
        
        n_slice = len(slices)
        print(f'Analyzing {n_slice} slices ')
        
        for i,sel in enumerate(slices):
         
            if not i%100:
                print(f'working on slice {i} {(i/n_slice*100):.1f} %')
            Vps_loc = dd.Vps[sel]
            Vn_loc = dd.Vps_bkg[sel]
            corr = signal.correlate(Vps_loc, Vn_loc, mode = 'full')
            lags = CL.correlation_lags(Vps_loc.size, Vn_loc.size, mode="full")
            
            lag = lags[np.argmax(corr)]    
            sel_r = slice(sel.start - lag, sel.stop - lag)
            # shift noise to align with data
            #Vnr = np.roll(dn.Vps, lag)
            Vnr = dd.Vps_bkg[sel_r]
            # calculate optimal scaling factor
            #a = np.sum(dd.Vps[sel]*Vnr[sel])/np.sum(Vnr[sel]**2)
            a = np.sum(dd.Vps[sel]*Vnr)/np.sum(Vnr**2)
            
            #V_sig[sel] = dd.Vps[sel] - a*Vnr[sel]
            V_corr[sel] = dd.Vps[sel] - a*Vnr
        print('Correction completed !')
        dd.Vps = V_corr
