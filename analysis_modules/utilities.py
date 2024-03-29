#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:47:10 2022


 Utility and helper functions
 
@author: boeglinw
"""
import numpy as np
import matplotlib.pyplot as pl

bool_res = bool_res = {"false":False, "true":True, '0': False, '1':True}
def Bool(x):
    """
    convert ascii True and False to bool values. Any other value will result in False
    example::
    
       >>> Bool('True')

    returns::

       >>> True
    
    """

    if x.lower() not in bool_res:
        # it's not a logical value, complain
        return None
    else:
        return bool_res[x.lower()]
    

def get_window_slice(xmin, x, xmax):
        # fast function to 
        # get the slice corresponding to xmin and xmax in x
        # the x-values need to be equally spaced        
        dx = x[1] - x[0]
        nmin = max( 0, int(round( (xmin - x[0])/dx )))
        nmax = min( (int(round ((xmax - x[0])/dx ))), len(x) -1  )
        return slice(nmin, nmax + 1)
    
    
def peak(x, a, b):
    # calculate the location of the maximum of the model peak shape
    if 2.*b/a-1.>0:
        x_0 = np.log(2.*b/a-1.)/(2.*b)
    else:
        x_0 = 0
    # value at the maximum location
    y_0 = np.exp(-a*(x_0))*(1.+np.tanh(b*(x_0)))
    # normalized shape centered at 0
    y = 1./y_0*np.exp(-a*(x+x_0))*(1.+np.tanh(b*(x+x_0)))
    # estimate full width
    large = y>=0.5
    try:
        il = np.where(large)[0]
        i1 = il.min()
        i2 = il.max()
        sig = x[i2] - x[i1]
    except:
        sig = 0.
    return sig, y 

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
        