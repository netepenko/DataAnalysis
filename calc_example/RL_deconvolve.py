#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:45:39 2022

Deconvolve histogram with fitted spread function (SF)

The deconvolution is a 1-dim version of Lucy-Richardson 

the PSF can be fitted by an asymmetric Voigt * gaussian or a sum of up to 4 gaussians


@author: boeglinw
"""

import numpy as np
import LT.box as B

from scipy import signal
from astropy.modeling.models import Voigt1D

import copy as C



def get_window(limits,arr):
    return (limits[0] <= arr) & (arr <= limits[1])


#%% make a class for the the psf

class PSF:
    
    def __init__(self, psf):
        self.psf = psf
        self.n_psf = psf.shape[0]      # length of psf
        self.loc_max = np.argmax(psf)  # location of maximum os psf

    def __call__(self,i,j):
        # check for scalar values
        i_is_scal = np.isscalar(i)
        j_is_scal = np.isscalar(j)
        #
        all_scal = i_is_scal and j_is_scal
        i_only = i_is_scal and (not j_is_scal)
        j_only = j_is_scal and (not i_is_scal)
        no_scal = (not i_is_scal) and (not j_is_scal)
        if (all_scal):
            # return psf value at i if max. is located at j
            k = (i - j) + self.loc_max
            self.k = k
            if k < 0 :
                return 0.
            elif k >= self.n_psf:
                return 0.
            else:
                return self.psf[k]
            
        elif (i_only or j_only):
            # return psf value at i if max. is located at j
            k = (i - j) + self.loc_max 
            self.k = k
            sel = (0<= k) & (k < self.n_psf)
            self.sel = sel
            psf_v = np.zeros_like(k).astype(float)
            psf_v[sel] = self.psf[k[sel]]
            return psf_v
        
        elif (no_scal):
            # calculate all possible combinations
            jj,ii = np.meshgrid(j,i)
            k = (ii - jj) + self.loc_max
            self.k = k
            sel = (0<= k) & (k < self.n_psf)
            self.sel = sel
            psf_v = np.zeros_like(k).astype(float)
            psf_v[sel] = self.psf[k[sel]]
            return psf_v
                   
#%% RL deconvoluton see https://en.wikipedia.org/wiki/Richardson–Lucy_deconvolution

# using psf class

def get_next_iteration(u, pij, d):
    # d : array of data
    # pij: psf matrix corresponding to d
    # loop over all data
    ci = pij@u  # estimated contribution to bin i from all the pthers using the current iteration
    ri = d/ci   # ratio between estimate and current exp. value in bin
    fj = pij.T@ri  # correction factor for current value
    un = u*fj      # scaled new itertion value for bin content
    return un
                


#%% load histogram to determine PSF from pulser peak


# create pseudo data

n_p = 150  # number of data points

x = np.linspace(0., 1.5, n_p)

x_g = 0.6; sig_g = 0.02
y = 500*np.exp(-(x-x_g)**2/(2*sig_g**2))

# for cnvolution
# asymmetric psf
psf = lambda x: 8*np.exp(-8*x)*(1.+np.tanh(15*(x-0.72)))

psfg = lambda x: np.exp(-(x-x_g)**2/(2*(4*sig_g)**2))

#%% convolute y with psf
#psf_n = psfg(x)/np.sum(psfg(x))
# assymetric
psf_n = psf(x)/np.sum(psf(x))

i_psf_max = np.argmax(psf_n)
psf_a = psf_n[:2*i_psf_max + 1]

# create PSF object
P = PSF(psf_a)
 
# make psf symmetric around mac


y_c = np.convolve(y, psf_a, mode = 'same')

#%% setup the old deconvolution
psf_loc = psf_a

hs = y_c  # set initial guess to the data
#hs = np.ones_like(hc)

n_iter = 200  # number of iterations

h_iter = []
h_iter.append(hs)

#%% perform iterative calculation and save results
for i in np.arange(n_iter):
    if i%10 == 0:
        print(f'Completing {i} iterations {i/n_iter*100.}%')
    hs = get_next_iteration(hs, psf_loc, y_c)
    h_iter.append(hs)

h_iter = np.array(h_iter)

#%% setup the new PSF deconvolution

ia = np.arange(y_c.shape[0])
ja = np.arange(y_c.shape[0])

pij = P(ia,ja)

hs = y_c  # set initial guess to the data
#hs = np.ones_like(hc)

n_iter = 2000  # number of iterations

h_iter = []
h_iter.append(hs)

#%% perform iterative calculation and save results
for i in np.arange(n_iter):
    if i%10 == 0:
        print(f'Completing {i} iterations {i/n_iter*100.}%')
    hs = get_next_iteration(hs, pij, y_c)
    h_iter.append(hs)

h_iter = np.array(h_iter)


#%%  evaluate iteration results
i_iter = 1800
title = f'Deconvoluted : {i_iter} iterations'

B.pl.figure(figsize = (9,4.5))
B.pl.plot(x, h_iter[i_iter], color = 'r')
B.pl.plot(x, h_iter[i_iter], 'bo')
B.pl.title(title)



    
    
    