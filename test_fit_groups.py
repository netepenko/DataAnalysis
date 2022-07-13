#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:22:14 2022

@author: boeglinw
"""
import numpy as np

def get_fit_groups_old_t(num_peaks, imax, t_off, t):
    # t time of each digitizer point
    # t_off time offset
    # np number of peaks to be fitted (on average)
    # imax indices of peak positions into the t array
 
    # total number of peaks   
    n_peaks = imax.shape[0]
    # total time interval to be fitted
    t_tot = t[-1] - t[0]
    t_res = t[1] - t[0]
    # average time between peaks
    delta_t = t_tot/n_peaks
    # fit time window
    fit_window = delta_t*num_peaks
    # group peak positions along fit windows
    fg = []
    fg_start = []
    fg_stop = []
    new_group = True
    same_group = False
    i_offset = int(t_off/t_res)
    i_window = int(fit_window/t_res)
    i_group = ( (imax - i_offset)/i_window ).astype(int)
    init_first_group = True 
    for i, ig in enumerate(i_group):
        # skip negative indices
        if ig < 0.:
            continue
        if init_first_group:
            i_current_group = ig
            init_first_group = False
            new_group = False
            fg_start.append(i)            
        same_group =  (ig == i_current_group)
        if same_group:
            continue
        else:
            fg_stop.append(i)
            new_group = True
        pass
        if new_group:
            fg_start.append(i)
            new_group = False
            same_group = True
            i_current_group = ig
        pass
    pass
    if (len(fg_stop) == len(fg_start) -1):
        fg_stop.append(fg_start[-1])
    
    fg = np.transpose(np.array([fg_start, fg_stop]))
    return fg, fit_window


def get_fit_groups_new_t(num_peaks, imax, t_off, t):
        # setup peak fitting groups
        # t     :   time of each digitizer point
        # t_off :   time offset, this is used to create shifted fitting groups
        # num_peaks    :  number of peaks to be fitted (on average) in one group
        # imax         : indices of peak positions into the t array
        #
        # returns: fg: fit group array containins start index into imax and end index of each fit group
        #          fit_window: average width of fit range in t
    
        # total number of peaks   
        n_peaks = imax.shape[0]
        # total time interval to be fitted
        t_tot = t[-1] - t[0]
        t_res = t[1] - t[0]
        # average time between peaks
        delta_t = t_tot/n_peaks
        # average fit time window
        fit_window = delta_t*num_peaks
        # group peak positions along fit windows
        fg = []
        fg_start = []
        fg_stop = []
        # offset in indices
        i_offset = int(t_off/t_res)
        # window size in indices
        i_window = int(fit_window/t_res)
        # start of fit groups
        i_group = ( (imax - i_offset).clip(min = 0)/i_window ).astype(int)
        group, fg_start = np.unique(i_group, return_index = True)
        # end of fit groups
        fg_stop = np.roll(fg_start, -1)
        fg = np.transpose(np.array([fg_start[:-1], fg_stop[:-1]]))
        return fg, fit_window
