#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 07:54:12 2022

@author: boeglinw
"""
import numpy as np
import matplotlib.pyplot as pl
import ffind_peaks2 as FP

import LT.box as B

#%%
def get_peak(x,y,dy = None, plot = False):
   """
   Parameters
   ----------
   x : independent data
   y : exp.  data
   dy : error in exp. data

   Returns
   -------
   peak position, uncertainty in peak position

   """
   # fit parabola to data
   if dy is None:
       p = B.polyfit(x, y, order =2)
   else:
       p = B.polyfit(x, y, dy, order =2)
   #
   # y = a0 + a1*x + a2*x**2
   #
   a0 = p.par[0]
   a1 = p.par[1]
   a2 = p.par[2]
   # covariance matrix
   C = p.cov
   # parabola max/min
   xm = -a1/(2.*a2)
   ym = p(xm)
   # for errors calculate partial derivatives wrsp a1, a2
   dxm_da1 = -1./(2.*a2)
   dxm_da2 = a1/(2*a2**2)
   # calculate total error using the covariance matrix
   dxm2 = (C[1,1]*dxm_da1**2 + 2.*C[1,2]*dxm_da1*dxm_da2 + C[2,2]*dxm_da2**2)
   dxm = np.sqrt(dxm2)
   # if selected plot the fitted curve
   if (plot):
       B.plot_line(p.xpl, p.ypl)
   return (xm, ym)
#%%
    

x  = np.linspace(0., 20, 1000)
y = np.sin(5*x) 
nmin, minpo, nmax, maxpo = FP.find_peaks(y.size, y.size, .1, y)

x_max = x[maxpo[:nmax]]
y_max = y[maxpo[:nmax]]

pl.plot(x,y)
pl.plot(x_max, y_max, 'ro')



ip_max = maxpo[:nmax]

nn = 2

slices = []
xn = []; yn = []

for i in ip_max:
    i_start = max(0,i - nn)
    i_end = min(x.size, i + nn + 1)
    slices.append((i_start, i_end))
    print(i_start, i_end)
    sl = slice(i_start, i_end)
    x0, y0 = x[i], y[i]
    xp,yp = get_peak(x[sl]-x0, y[sl]-y0, plot = True)
    xn.append(xp+x0)
    yn.append(yp+y0)
xn = np.array(xn)
yn = np.array(yn)
pl.plot(xn, yn, 'co')

#%%

xmf, ymf = FP.refine_positions(2, maxpo[:nmax], x, y, x.size, nmax)

pl.plot(xmf, ymf, 'mx')
