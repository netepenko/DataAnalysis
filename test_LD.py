#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:45:48 2023

Try to determine the frequency of the noise by histogramming overlapping data segments

@author: boeglinw
"""

import numpy as np
import LT.box as B

from analysis_modules import common_peak_fitting_class as CFC
from analysis_modules import load_digitizer_data as LD
from analysis_modules import correlation_lags as CL
import time

from scipy import signal

def get_modulo(t, V, T, npt = 10000, plot = False, verbose = False):
    dt = t[1] - t[0]  # get time step
    T_indx = int(T/dt + 0.5)  # get the period size in time steps
    if verbose:
        print(f'Number of periods = {int((t[-1]-t[0])/T + 0.5)}')
    td_indx = np.arange(t.shape[0]) # make an array the size of the data
    indx_rel = td_indx%T_indx  # cal. the relative indices into the local array using the rminder of td_indx/T_indx
    V_s = np.zeros(T_indx)  # setup the array for the results
    tt = np.arange(T_indx)*dt # setup the time axis
    h = B.histo(indx_rel[:npt], range = (-.5, V_s.shape[0] - 1 + 0.5), bins = V_s.shape[0], weights = V[:npt]) # histogram the data
    V_s = h.bin_content
    if plot:
        B.pl.plot(indx_rel[:npt], V[:npt], '.')  # plot the various period data if selected
    return np.arange(T_indx)*dt, V_s


def shift_slice(sl, shift):
    return slice(sl.start+shift, sl.stop+shift) 


def make_2d_histo(t, V, tmin = 0., tmax = 1., slice_width = 1e-3, h_min = 0., h_max = 1., hy_bins = 100, shot = 0, channel = 0):

    # 2d histogram setup
    
    # convert times to us
    tmin *= us
    tmax *= us
    slice_width *= us
    
    # y - axis    # x - axis
    hx_bins = int((tmax - tmin)/slice_width) + 1
    
    h_title = f'shot {shot}, channel: {channel}' 
    
    
    h2p = B.histo2d(t, V, range = [[tmin,tmax],[h_min,h_max]], bins = [hx_bins, hy_bins],
                         title = h_title, xlabel = r't[$\mu$ s]', ylabel = 'raw PH [V]')

    return h2p


us = 1e6
#%% load data
data_dir = '/common/uda-scratch/Proton_Detector/'
shot = 48849
file_name = 'DAQ_48849_231117_143332.hws'
#diamond
dd = LD.digi_data(data_dir + file_name)

data_tmin = 1e5
data_tmax = 10e5

t_offset = .1*us

dd.channel=6
dd.tmin = data_tmin
dd.tmax = data_tmax

dd.load_raw_data()
# calculat moving average 3 times 
print('Calculate moving averages for signals')
for i in range(5):
    dd.moving_average(21)



# noise
dn = LD.digi_data(data_dir + file_name)
dn.channel=4
dn.tmin = data_tmin
dn.tmax = data_tmax
dn.load_raw_data()


# calculat moving average 3 times 
print('Calculate moving averages for Noise')
for i in range(5):
    dn.moving_average(21)

#%%
"""
ns = 00; ne = ns + 800000


# find 6 kHz frquency

T0 = 167.224

f = np.linspace(0.9, 1.1, 1000)
t0 = time.time()
Sa = []
for ff in f:
    tt, Va = get_modulo(dn.td[ns:ne], dn.Vps[ns:ne], T0*ff, npt = ne, plot = False)    
    S = np.sum(Va**2)
    Sa.append(S)
Sa = np.array(Sa) 
t1 = time.time()   
print(f'Sum = {S}, time : {t1 - t0}')

B.pl.figure()
B.pl.plot(f, Sa)


i_max = np.argmax(Sa)

T_max = f[i_max]*T0
f_max = 1./T_max*1e6

print(f'Best period : {T_max:.3e} (us) frequency = {f_max:.4e} Hz')

#%% final version
ns = 00; ne = ns + 800000

tt, Va = get_modulo(dn.td[ns:], dn.Vps[ns:], T_max, npt = ne, plot = True)

"""
#%% try background subtraction


# time window in us
delta_t = 200

"""
# t0 = 0.33948662e6

t0 = 0.27*us

sel = LD.get_window_slice(t0, dd.td, t0 + delta_t)

# data
B.pl.plot(dd.td[sel], dd.Vps[sel])


# noise

B.pl.plot(dd.td[sel], dn.Vps[sel])

# determine optimal phase for this slice

corr = signal.correlate(dd.Vps[sel], dn.Vps[sel], mode = 'full')
lags = signal.correlation_lags(dd.Vps[sel].size, dn.Vps[sel].size, mode="full")

lag = lags[np.argmax(corr)]

# shift noise to align with data
Vnr = np.roll(dn.Vps, lag)


# data
B.pl.plot(dd.td[sel], dd.Vps[sel])
B.pl.plot(dd.td[sel], Vnr[sel])


# calculate optimal scaling factor

a = np.sum(dd.Vps[sel]*Vnr[sel])/np.sum(Vnr[sel]**2)

Vc = dd.Vps[sel] - a*Vnr[sel]

B.pl.plot(dd.td[sel], Vc)


"""

#%% loop over all data and subtract noise

# create an arrau of slices
n_slice = int(delta_t//dd.dt)

i_start = np.arange(0, dd.td.shape[0], n_slice)  # starting index
i_end = np.roll(i_start, -1)                        # stopping index

# create the slice array
slices = [slice(ss,ee) for ss, ee in zip(i_start, i_end)][:-1]  # skip the last slice

V_sig = np.zeros_like(dd.Vps)

n_slice = len(slices)
print(f'Analyzing {n_slice} slices ')

for i,sel in enumerate(slices):
 
    if not i%100:
        print(f'working on slice {i} {(i/n_slice*100):.1f} %')
    Vps_loc = dd.Vps[sel]
    Vn_loc = dn.Vps[sel]
    corr = signal.correlate(Vps_loc, Vn_loc, mode = 'full')
    lags = CL.correlation_lags(Vps_loc.size, Vn_loc.size, mode="full")
    
    lag = lags[np.argmax(corr)]    
    sel_r = slice(sel.start - lag, sel.stop - lag)
    # shift noise to align with data
    #Vnr = np.roll(dn.Vps, lag)
    Vnr = dn.Vps[sel_r]
    # calculate optimal scaling factor
    #a = np.sum(dd.Vps[sel]*Vnr[sel])/np.sum(Vnr[sel]**2)
    a = np.sum(dd.Vps[sel]*Vnr)/np.sum(Vnr**2)
    
    #V_sig[sel] = dd.Vps[sel] - a*Vnr[sel]
    V_sig[sel] = dd.Vps[sel] - a*Vnr
    
#%% find peaks
pks = signal.find_peaks(V_sig, height = [0.02, 0.5])

tp = dd.td[pks[0]]  #  peak times  
Vp = pks[1]['peak_heights'] # peak heights

# create 2d histo
h2 = make_2d_histo(tp, Vp, h_min = 0., h_max = 0.3, tmin = data_tmin/us, tmax = data_tmax/us, shot = shot, channel = 6)
B.pl.figure(figsize = (10,4))
h2.plot()

#%% calculate rate
p_min = 0.077
p_max = 0.2

hy = []
S = [];dS = []
for i in range(h2.nbins_x):
    hy_loc = h2.project_y(bins = [i])
    C, dC = hy_loc.sum(p_min, p_max)
    S.append(C)
    dS.append(dC)
    hy.append(hy_loc)
R = np.array(S)/(h2.x_bin_width/us)
dR = np.array(dS)/(h2.x_bin_width/us)

#%%
B.pl.figure(figsize = (14,4))
#B.plot_exp(h2.x_bin_center/us, R, color = 'r')
B.plot_line(h2.x_bin_center/us - t_offset/us, R, color = 'r')

#B.pl.fill_between(h2.x_bin_center/us, R - dR, R + dR, color = 'r', alpha = 0.5)
B.pl.title(f'shot {shot}, channel: {6}')
B.pl.ylabel('Rate (Hz)')
B.pl.xlabel('time (s)')
    