#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:45:39 2022

Deconvolve histogram with fitted spred function (SF)

@author: boeglinw
"""

import numpy as np
import LT.box as B

from scipy import signal
from astropy.modeling.models import Voigt1D

import copy as C

def get_window(limits,arr):
    return (limits[0] <= arr) & (arr <= limits[1])



class voigt_fit:
    # initial values can be given when a fit object is created
    def __init__(self, A = 1., fwhm_L=1., fwhm_G = 0.5, x0 = 0.):
        # initialize the class and store initial values
        self.A = B.Parameter(A, 'A')
        self.x0 = B.Parameter(x0, 'x0')
        self.fwhm_L = B.Parameter(fwhm_L, 'fwhm_L')
        self.fwhm_G = B.Parameter(fwhm_G, 'fwhm_G')
        # background
        self.b0 = B.Parameter(0., 'b0')
        self.b1 = B.Parameter(0., 'b1')
        self.b2 = B.Parameter(0., 'b2')
        
        self.fit_list = [self.A, self.x0,self.fwhm_L,self.fwhm_G]

    def voigt(self, x):
        v1 = Voigt1D(x_0=self.x0(), amplitude_L=self.A(), fwhm_L=self.fwhm_L(), fwhm_G=self.fwhm_G())        
        return v1(x)
    
    def bkg(self, x):
        b = self.b0() + self.b1()*x + self.b2()*x**2
        return b
    
    def signal(self, x):
        return self.voigt(x)
    
    def total(self,x):
        return self.voigt(x) + self.bkg(x)
    
    def __call__(self, x):
        return self.signal(x)

    def fit(self, x, y, dy):
        self.x = x
        self.y = y
        if dy is None:
            self.dy = np.ones_like(y)
        else:
            self.dy = dy
        self.fit_res = B.genfit(self.total, self.fit_list, self.x, self.y, y_err = self.dy, plot_fit = False)
        self.fwhm_tot = np.sqrt(self.fwhm_G()**2 + self.fwhm_L()**2)
        self.plot_fit()
    def plot_fit(self):
        B.plot_line(self.fit_res.xpl, self.fit_res.ypl)
        B.plot_line(self.fit_res.xpl, self.bkg(self.fit_res.xpl))
# end of class definition
# the peak position in this case is the parameter x0()

#%% load histogram

h = B.histo(file = 'ch3_projection.data')
h.clear_window()  # needed for histograms loaded from file, this is a bug that needs to be fixed

hc = h.bin_content
he = h.bin_error


# fit pulser peak
# initialize voigt
vf = voigt_fit(A = 7000, fwhm_L=0.03, fwhm_G=0.01, x0 = 1.0)
vf.fit_list = [vf.A, vf.x0, vf.fwhm_L, vf.fwhm_G]
#vf.fit_list = [vf.A, vf.fwhm_L, vf.fwhm_G, vf.x0, vf.b0, vf.b1]


# select fit range (e.g from xlim())
lims = (0.81, 1.2)

sel = get_window(lims, h.bin_center)

# set the fit variables
xb = h.bin_center[sel]; yb = h.bin_content[sel]; dyb = h.bin_error[sel]

# estimate fwhm

top_part = xb[vf(xb) >=vf(xb).max()/2]
fwhm = top_part[-1] - top_part[0]

# do the fit:
h.plot_exp()
vf.fit(xb, yb, dyb)
x0_vf_fit = vf.x0()
"""
gf.fit(xb, yb, dyb)
x0_gf_fit = gf.x0()
"""
#%% make data array for deconvolution based on histogram h and its fit
def get_psf(h, fit, fwhm):
    sig = fwhm
    #
    nb = 4*round(sig/h.bin_width)
    #
    xmin = -nb*h.bin_width 
    #
    xsf = xmin + np.arange(2*nb + 2)*h.bin_width
    # make normalized point spread function
    fit.x0.set(0.)
    psf  = fit(xsf)
    psf /= psf.sum()

    return psf

#%% find indices for and asymmetric psf



def get_indices_asy(i_v, psf, i_s):
    # i_v idex of current value
    # psf point spread function (response function)
    # get data window
    n_psf = psf.shape[0]    
    i_max = np.argmax(psf)
    # check upper half
    n_up = n_psf-i_max
    n_low = i_max
    # relative position of i_v in i_s array
    i_diff = i_s - i_v
    # select data for upper half
    i_w_up = np.where(get_window((0, n_up - 1),i_diff))[0]
    i_w_low = np.where(get_window((-n_low, -1),i_diff))[0] 
    i_window = np.append(i_w_low, i_w_up)
    i_psf_range = i_window - i_v + i_max
    return i_window, i_psf_range


def get_next_iteration(u, psf):
    # loop over all data
    un = np.zeros_like(u)
    i_s = np.arange(u.shape[0])
    for j in i_s:
        i_w, i_pr = get_indices_asy(j, psf, i_s)
        ci = []
        for k in i_w:
            i_w_c, i_pr_c = get_indices_asy(k, psf, i_s)
            c_i = np.sum(u[i_w_c]*psf[i_pr_c])
            ci.append(c_i)
        ci = np.array(ci)
        u_n = u[j]*np.sum(hc[i_w]*psf[i_pr]/ci)
        un[j] = u_n
    return un
        
            
#%% do the iterations
# set the psf
psf_loc = get_psf(h, vf, fwhm)

hs = hc
n_iter = 100
h_iter = []
h_iter.append(hs)
for i in np.arange(n_iter):
    hs = get_next_iteration(hs, psf_loc)
    h_iter.append(hs)

h_corr = C.copy(h)
h_corr.bin_content = hs
h_corr.title = f'Deconvoluted : {n_iter} iterations'

B.pl.figure(figsize = (9,4.5))
h_corr.plot(filled = False, color = 'r')


