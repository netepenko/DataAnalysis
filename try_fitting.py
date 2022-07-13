#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May  6 15:33:27 2022


@author: boeglinw
"""

import numpy as np
import matplotlib.pyplot as pl
from analysis_modules import channel_data_class as cdc
from analysis_modules import peak_sampling_class as PS
from analysis_modules import raw_fitting_class as RFC


import LT.box as B

us = 1e6


#%%
"""
def print_array(a, name):
    a_str = ''.join([f'{xx},' for xx in a])[:-1]
    print(f'{name} = [{a_str}]')
    
#%% debug fortran data
class debug_FP:

    def __init__(self, file, i = 0):
        d = np.loadtxt(file)
        self.d = d
        self.set_values(i)
        
    def set_values(self, i):
        d = self.d[i]
        self.a = d[:3]
        self.ip = int(d[3])
        self.x0 = d[4]
        self.y0 = d[5]
        self.i_start = int(d[6])
        self.i_end = int(d[7])
        self.xf = d[8]
        self.yf = d[9]
        n_i = self.i_end - self.i_start + 1
        here = 10
        self.x = d[here : here + n_i]
        here = here + n_i
        self.y = d[here : here + n_i]
        
        
    def plot_fit(self):
        a = self.a
        pl.plot(np.array([self.x0]), np.array([self.y0]), 'ro')
        pl.plot(self.x + self.x0, self.y, '.')
        xx = np.linspace(self.x.min(), self.x.max(), 100)
        y = a[0] + (a[1] + a[2]*xx)*xx
        pl.plot(xx + self.x0, y)
    
"""

#%%
cc = cdc.channel_data(29880, 2, 'New_MainDB1.db')
cc.read_database_par()
cc.load_data()
"""
cc.load_npz_data(file_name='DAQ_190813-112521_filtered.npz')
cc.td *= cdc.us
cc.dt *= cdc.us
"""
#%% data range

tmin = cc.par['dtmin']
tmax = cc.par['dtmax']
cc.plot_raw(tmin, tmax)

#%%

tp_min = 0.1
d_tp = 0.02
tp_max = tp_min + d_tp

chi2 = .1
Vstep = .08
Vthresh = .3
#%% get sample peaks

ps = PS.peak_sampling(cc, chi2, plot_single_peaks = False, plot_common_peak = True)
ps.chi2 = chi2
ps.find_good_peaks(tp_min*cdc.us, tp_max*cdc.us, Vstep, Vthresh)

ps.fit_peaks(ps.good_peak_times, save_fit = True)
ps.save_parameters(cc.db_file)

#%% ready for raw fitting

cc.par['Vth'] = .1
cc.par['Vstep'] = .1

rf = RFC.raw_fitting(cc, plot = 10, plot_s = 0, start_plot_at=200000, refine_positions=True)
rf.fit_progress = 100
rf.find_peaks(N_close = 2)
rf.check_cov = False

#%%
rf.use_refined = True

rf.setup_fit_groups()
rf.init_fitting()
print(f'Created {rf.fg.shape[0]} fit groups')
#%%

ng = 1000
rf.plot_fit_group(ng, shifted = False, warnings = True)
rf.plot_fit_group(ng, shifted = True, warnings = True)

#%%
rf.fit_data()

#%%
rf.save_fit()
