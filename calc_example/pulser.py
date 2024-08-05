#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:12:27 2023

pulser peak shape

@author: boeglinw
"""

import numpy as np
import LT.box as B




def peak(x, xp, a = 1., b = 1.):
    
    # calculate peak values
    x_0 = np.log((2.*b)/a-1.)/(2.*b)
    y_0 = np.exp(-a*(x_0))*(1.+np.tanh(b*(x_0)))
    
    x_pos = x_0 - xp
    y = 1./y_0*np.exp(-a*(x+x_pos))*(1.+np.tanh(b*(x+x_pos)))
    return y


#%%
x = np.linspace(0., 20, 200)

x0 = 5.

beta = 3.

alpha = 1.
d_alpha = 0.8

frac = 0.25

y1 = peak(x, x0,  a = alpha, b = beta)
y2 = peak(x, x0 + 0.5, a = alpha - d_alpha, b = beta)

y = y1 - frac*y2

B.pl.plot(x, y1)
B.pl.plot(x, frac*y2)

B.pl.plot(x, y)

