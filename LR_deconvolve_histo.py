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

def gauss(x, sig, x0):
    return np.exp(-0.5*((x- x0)/sig)**2)


class BParameter(B.Parameter):
    
    def __init__(self, val, name, bound = [-np.inf, np.inf]):
        self.bound = bound
        super().__init__(val, name)

        
        
positive = [0., np.inf]

#%% fitting the point spread function with an asymmetric Voigt function
class voigt_fit:
    # initial values can be given when a fit object is created
    def __init__(self, A = 1.,
                 Ag = 500.,
                 sig_g = 0.01,
                 x0 = 0., 
                 x0_g = 0.,
                 fwhm_L_l =1., 
                 fwhm_G_l = 0.5, 
                 fwhm_L_r=1., 
                 fwhm_G_r = 0.5):
        # initialize the class and store initial values
        self.A = BParameter(A, 'A', positive)
        self.Ag = BParameter(Ag, 'Ag', positive)
        self.sig_g = BParameter(sig_g, 'sig_g', positive)
        
        self.x0 = BParameter(x0, 'x0', positive)
        self.x0_g = BParameter(x0, 'x0_g', positive)
        
        self.fwhm_L_l = BParameter(fwhm_L_l, 'fwhm_L_l', positive)
        self.fwhm_G_l = BParameter(fwhm_G_l, 'fwhm_G_l', positive)
        
        self.fwhm_L_r = BParameter(fwhm_L_r, 'fwhm_L_r', positive)
        self.fwhm_G_r = BParameter(fwhm_G_r, 'fwhm_G_r', positive)        
        
        # background
        self.b0 = BParameter(0., 'b0')
        self.b1 = BParameter(0., 'b1')
        self.b2 = BParameter(0., 'b2')
        
        
        
        self.fit_list = [self.A, self.x0, self.fwhm_L_l,self.fwhm_G_l, self.fwhm_L_r,self.fwhm_G_r, self.Ag, self.sig_g, self.x0_g]

    def voigt(self, x):   
        v1_l = Voigt1D(x_0=self.x0(), amplitude_L=1., fwhm_L=self.fwhm_L_l(), fwhm_G=self.fwhm_G_l())
        v1_r = Voigt1D(x_0=self.x0(), amplitude_L=1., fwhm_L=self.fwhm_L_r(), fwhm_G=self.fwhm_G_r())  
        sel_l = x < self.x0()
        sel_r = ~sel_l
        vv = np.zeros_like(x)
        vl = v1_l(x[sel_l])
        vr = v1_r(x[sel_r])
        vv[sel_l] = vl/vl.max()  # normalize heights
        vv[sel_r] = vr/vr.max()          
        return self.A()*vv + self.Ag()*gauss(x, self.sig_g(), self.x0_g())
        
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
        #setup parameter bounds
        p_bounds_l = [pp.bound[0] for pp in self.fit_list]
        p_bounds_u = [pp.bound[1] for pp in self.fit_list]
        self.fit_res = B.genfit(self.total, self.fit_list, self.x, self.y, y_err = self.dy, plot_fit = False, bounds = [p_bounds_l, p_bounds_u])
        self.fwhm_tot = np.sqrt(self.fwhm_G_l()**2 + self.fwhm_L_l()**2)/2. + np.sqrt(self.fwhm_G_r()**2 + self.fwhm_L_r()**2)/2.
        self.plot_fit()
    def plot_fit(self):
        B.plot_line(self.fit_res.xpl, self.fit_res.ypl)
        B.plot_line(self.fit_res.xpl, self.bkg(self.fit_res.xpl))
        
    def set_peakpos(self, x):
        self.x0.set(x)
# end of class definition
# the peak position in this case is the parameter x0()


#%% fitting the point spread function with an asymmetric  sum of gaussian





class sog_fit:
    # initial values can be given when a fit object is created
    def __init__(self, A1 = 1., 
                 A2 = 0.1,  
                 A3 = 0.1,
                 A4 = 0.1,
                 x01 = 0., 
                 x02 = 1., 
                 x03= 1., 
                 x04 = 1.,
                 sig1 = 0.03, 
                 sig2 = 0.1, 
                 sig3 = 0.2,
                 sig4 = 0.15,
                 alpha = 0., beta = 20.):
        # initialize the class and store initial values
        self.A1 = BParameter(A1, 'A1', positive)
        self.A2 = BParameter(A2, 'A2', positive)
        self.A3 = BParameter(A3, 'A3', positive)
        self.A4 = BParameter(A4, 'A4', positive)
        self.x01 = BParameter(x01, 'x01', positive)
        self.x02 = BParameter(x02, 'x02', positive)
        self.x03 = BParameter(x03, 'x03', positive)
        self.x04 = BParameter(x04, 'x04', positive)
        self.sig1 = BParameter(sig1, 'sigma_1', positive)
        self.sig2 = BParameter(sig2, 'sigma_2', positive)
        self.sig3 = BParameter(sig3, 'sigma_3', positive)
        self.sig4 = BParameter(sig4, 'sigma_4', positive)
        
        self.alpha = BParameter(alpha, 'alpha', positive)
        self.beta = BParameter(beta, 'beta', positive)
        # background
        self.b0 = BParameter(0., 'b0')
        self.b1 = BParameter(0., 'b1')
        self.b2 = BParameter(0., 'b2')
        
        self.fit_list = [self.A1, self.A2, self.A3,self.A4,  self.x01, self.x02, self.x03, self.x04, self.sig1 ,self.sig2, self.sig3, self.sig4]
        
        self.peak_pos = None

    def sog(self, x): 
        self.asy(x)
        sig1 = self.sig1()*self.a1
        sig2 = self.sig2()*self.a2
        sig3 = self.sig3()*self.a3
        sig4 = self.sig4()*self.a4
        
        return self.A1()*gauss(x, sig1, self.x01())  + \
                self.A2()*gauss(x, sig2, self.x02())  + \
                    self.A3()*gauss(x, sig3, self.x03()) +  \
                        self.A4()*gauss(x, sig4, self.x04())
    
    def asy(self, x):
        self.a1 = 1 + self.alpha()*(1+np.tanh(self.beta()*(x-self.x01())) )
        self.a2 = 1 + self.alpha()*(1+np.tanh(self.beta()*(x-self.x02())) )
        self.a3 = 1 + self.alpha()*(1+np.tanh(self.beta()*(x-self.x03())) )
        self.a4 = 1 + self.alpha()*(1+np.tanh(self.beta()*(x-self.x04())) )
        
    def bkg(self, x):
        b = self.b0() + self.b1()*x + self.b2()*x**2
        return b
    
    def signal(self, x):
        return self.sog(x)
    
    def total(self,x):
        return self.sog(x) + self.bkg(x)
    
    def __call__(self, x):
        return self.signal(x)

    def fit(self, x, y, dy):
        self.x = x
        self.y = y
        if dy is None:
            self.dy = np.ones_like(y)
        else:
            self.dy = dy
            
        #setup bounds
        bl = [pp.bound[0] for pp in self.fit_list]
        bu = [pp.bound[1] for pp in self.fit_list]
        self.fit_res = B.genfit(self.total, self.fit_list, self.x, self.y, y_err = self.dy, plot_fit = False, bounds = [bl, bu])
        self.fwhm_tot = np.sqrt(self.sig1()**2 + self.sig2()**2 + self.sig3()**2 + self.sig4()**2) * 2.35
        self.plot_fit()
    def plot_fit(self):
        B.plot_line(self.fit_res.xpl, self.fit_res.ypl)
        B.plot_line(self.fit_res.xpl, self.bkg(self.fit_res.xpl))

    def get_peak_pos(self, xmin, xmax, N = 1000):
        xx = np.linspace(xmin, xmax, 1000)
        self.peak_pos = xx[np.argmax(self.total(xx))]

    def set_peakpos(self, x):
        if self.peak_pos is None:
            print('Peak position not determined run get_peak_pos')
            return
        dx = x - self.peak_pos
        self.x01.set(self.x01() + dx)
        self.x02.set(self.x02() + dx)
        self.x03.set(self.x03() + dx)
        self.x04.set(self.x04() + dx)
# end of class definition
# the peak position in this case is the parameter x0()


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
    fit.set_peakpos(0.)
    psf  = fit(xsf)
    psf /= psf.sum()

    return psf



#%% devoncolution procedure

#find indices for an asymmetric

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
        

#%% load histogram to determine PSF from pulser peak

h = B.histo(file = 'ch1_projection.data')
#h = B.histo(file = 'ch2_projection.data')
#h = B.histo(file = 'ch3_projection.data')

h.clear_window()  # needed for histograms loaded from file, this is a bug that needs to be fixed

hc = h.bin_content
he = h.bin_error


#%% fit pulser peak to get PSF
# initialize voigt
vf = voigt_fit(A = 5000, Ag = 1000., sig_g = 0.02, x0 = 1.0, x0_g = 1., fwhm_L_l=0.03, fwhm_G_l=0.01, fwhm_L_r=0.03, fwhm_G_r=0.01 )
#vf.fit_list = [vf.A, vf.x0, vf.fwhm_L, vf.fwhm_G]
#vf.fit_list = [vf.A, vf.fwhm_L_l, vf.fwhm_G_l, vf.fwhm_L_r, vf.fwhm_G_r, vf.x0, vf.b0, vf.b1]


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

xx = np.linspace(lims[0], lims[1], 1000)
B.pl.plot(xx, vf.total(xx))

# fitted peak position
x0_gf_fit = xx[np.argmax(vf.total(xx))]



#%% fit pulser peak to get PSF sig SOG
# initialize voigt
gf = sog_fit(A1 = 4000, A2 = 2000, A3 = 1000., A4 = 500, x01 = 1.0, x02 = 1., x03 = 1., x04 = 1.0, sig1 = 0.014, sig2 = 0.05, sig3 = 0.07, sig4 = 0.1 )



# select fit range (e.g from xlim())
lims = (0.81, 1.2)

sel = get_window(lims, h.bin_center)

# set the fit variables
xb = h.bin_center[sel]; yb = h.bin_content[sel]; dyb = h.bin_error[sel]

# estimate fwhm

top_part = xb[gf(xb) >=gf(xb).max()/2]
fwhm = top_part[-1] - top_part[0]

# do the fit:
B.pl.figure()
h.plot_exp()
gf.fit(xb, yb, dyb)

xx = np.linspace(lims[0], lims[1], 1000)
B.pl.plot(xx, gf.total(xx))

# fitted peak position
gf.get_peak_pos(lims[0], lims[1])

#%% setup the deconvolution
psf_fit = gf  # select the fit function

# set the psf from the fit
psf_loc = get_psf(h, psf_fit, fwhm)

hs = hc
n_iter = 200  # number of iterations

h_iter = []
h_iter.append(hs)

# perform iterative calculation and save results
for i in np.arange(n_iter):
    hs = get_next_iteration(hs, psf_loc)
    h_iter.append(hs)

h_iter = np.array(h_iter)
# make a copy of the initial histogram
h_corr = C.copy(h)

dh_iter = np.diff(h_iter, axis = 0)

#%%  evaluate iteration results
i_iter = 40
h_corr.bin_content = h_iter[i_iter]  # set the corrected bin_content
h_corr.title = f'Deconvoluted : {i_iter} iterations'

B.pl.figure(figsize = (9,4.5))
h_corr.plot(filled = False, color = 'r')
h_corr.plot_exp()

#%%  differenc ebetween iterations
i_iter_l = 0
i_iter_u = 10

h_corr.bin_content = h_iter[i_iter_u] - h_iter[i_iter_l]  # set the corrected bin_content
h_corr.bin_error = np.sqrt(h_iter[i_iter_u] + h_iter[i_iter_l])
h_corr.title = f'difference between : {i_iter_l} and {i_iter_u} iterations'

B.pl.figure(figsize = (9,4.5))
h_corr.plot(filled = False, color = 'r')
h_corr.plot_exp()