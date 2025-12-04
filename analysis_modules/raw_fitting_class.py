# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:05:53 2016


@author: Alex
"""
import numpy as np
import LT.box as B

import h5py

import time
import os

from . import ffind_peaks2 as FP
from . import lfitm1 as LF 
from . import utilities as UT
from . import database_operations as db
import matplotlib.pyplot as plt

#conversion to microseconds constant
us=1.e6

def lish(x):        # Vectorize the line shape
    return LF.lfitm1.line_shape(x)
line_shape = np.vectorize(lish)
        


def get_fit_groups(num_peaks, imax, t_off, t):
    #
    # latest version WB 2022
    #
    # setup peak fitting groups
    # t     :   time of each digitizer point
    # t_off :   time offset, this is used to create shifted fitting groups
    # num_peaks    :  number of peaks to be fitted (on average) in one group
    # imax         : indices of peak positions into the t array
    #
    # calcualte: fg: fit group array containing start and end index into imax array for each fit group
    #            fit_window: average width of fit range in t
    #            fg_limits: array of limits into the data array for each conscutie group
    #            fg_number: array of fit group numbers, these are the indices into the fg_limits array

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
    # offset in indices
    i_offset = int(t_off/t_res)
    # window size in indices
    i_window = int(fit_window/t_res)
    # calculate limits into the data array for each fit group
    i_start = np.arange(i_offset, t.size+i_window, i_window).clip(max = (t.size - 1))
    i_stop = np.roll(i_start,  -1)
    fg_limits =  np.array([i_start[:-1], i_stop[:-1]]).T
    # groups peak positions into fit groups
    # shift position indices by i_offset and keep only positive values
    imax_shift = imax - i_offset
    n_shift = np.sum(imax_shift < 0)   # select only positive peak indices
    imax_sel = imax_shift[n_shift:]   # select the corresponding positions
    i_group = (imax_sel/i_window ).astype(int) # group the positions
    # the group number is the index into the limits array as the group numbers are not necessarily consecutive
    fg_number, fg_start, fg_counts = np.unique(i_group, return_index = True, return_counts = True)
    fg_start += n_shift # increase the index by the number of dropped values to point back into the original array
    # end of fit groups
    fg_stop = fg_start + fg_counts
    fg = np.zeros_like(fg_limits)
    fg[:,0][fg_number] = fg_start
    fg[:,1][fg_number] = fg_stop
    return fg, fit_window, fg_limits
    

class raw_fitting:
    """
    
    Call to to the fitting of raw data
    """
    def __init__(self, channel_data, 
                 tmin = None, tmax = None, 
                 plot=0, plot_s = 0, start_plot_at = None, 
                 refine_positions = False, 
                 use_refined = False, 
                 scan_only = False, 
                 correct_data = False,
                 check_cov = False,
                 fit_progress = 1000,
                 n_check_peaks = 10000):
        """
        create an instance for fitting the peaks

        Parameters
        ----------
        channel_data : channed_data_class instance
            class containing the data for a digitizer channel
        tmin : float, optional
            lower time limit for fitting peaks. If None the value is taken from the data base
        tmax : float, optional
            upper tme limit for fitting peaks.  If None the value is taken from the data base
        plot : int, optional
            number of fit groups to plot with fits. The default is 0.
        plot_s : int, optional
            number of shifted fit groups to plot with fits. The default is 0.            
        start_plot_at : float, optional
            time when plotting fits starts. The default is tmin   
        refine_positions : bool, optional
            calculate refined peak positions by fitting a parabola to top of peaks (default = False)
        use_refined: bool, optional
            use refined peak positions in fitting. (default = False)
        scan_only: bool, optional
            quick scan to check if there are valid data. Returns true is >= n_check_peaks are found (default = False)
        correct_data: bool, optional
            subtract fitted background from data and save to new file for further analysis (default = False)
        check_cov: bool, optional
            check that the covariance matrix  diagonal is everywhere positive (default = False)
        fit_progress: int, optional
            number of fits after which a progess message is printed during fitting (default = 1000)
        n_check_peaks: int, optional
            minimum number of peaks needed to declare that there are valid data ins the file (default = 10000)
            

        Returns
        -------
        None.

        """

        self.plot = plot  # number of fit groups to plot
        self.plot_s = plot_s # number of shifted fit groups to plot
        self.n_check_peaks = n_check_peaks  # min. nummber of peaks found to declare that there are data
        if start_plot_at is None:
            self.start_plot_at = tmin
        else:
            self.start_plot_at = start_plot_at
        self.channel_data = channel_data

        self.fit_progress = fit_progress
        self.refine_positions = refine_positions
        self.use_refined = use_refined
        self.check_cov =  check_cov

        self.correct_data = correct_data
        self.corrected_data_scale = 10000  # resolution 0.1 mV for saved data as integers

        if scan_only:
            tmin = channel_data.td.min()/us
            tmax = channel_data.td.max()/us

        if tmin is None:
            self.tmin = channel_data.par['dtmin']/us
        else:
            self.tmin = tmin
        if tmax is None:
            self.tmax = channel_data.par['dtmax']/us
        else:
            self.tmax = tmax
        
        if scan_only:
            self.find_peaks()
            self.has_data = self.imax_fit.shape[0] > self.n_check_peaks

        
    def do_all(self):
        """
        Do the complete fitting and saving procedure

        Returns
        -------
        None.

        """
        self.find_peaks()
        self.setup_fit_groups()
        self.fit_data()
        self.save_fit()
        
    def find_peaks(self,  N_close = 2):
        """
        Find peak position in time windows.
        - only selects peak larger thatn Vth
        - peak is detected when V falls more the Vstep
        
        - sets the values for:
            self.imax_fit : array if indices indicating the good peaks
            self.tp       : times for the peaks
            self.Vp       : Voltage values for the peaks

        Parameters
        ----------
        N_close : int 
            number of closest neigjbors used to fit parabola for peak finding (default = 2)
        
        Returns
        -------
        None.

        """
        sl = UT.get_window_slice(self.tmin*us, self.channel_data.td, self.tmax*us)
        self.sl = sl
        
        self.V=self.channel_data.Vps[sl]
        self.td=self.channel_data.td[sl]
        self.dt=self.channel_data.dt
        
        if self.correct_data:
            self.V_corr = np.zeros_like(self.V)

        #-------------------------------------
        # setup peak finding
        #-------------------------------------
        Vstep=self.channel_data.par['Vstep']
        Vth=self.channel_data.par['Vth']

        max_neg_V = self.channel_data.par['max_neg_V']
        min_delta_t = self.channel_data.par['min_delta_t']*us

        # create arrays for peak locations
        psize = self.channel_data.psize
        
        N = len(self.V)
        Np = int(len(self.V)/psize)  # rough estimate of the possible max. number of peaks
        # n_min, pmin, n_max, pmax = FP.find_peaks(N, Np, Vstep, self.V) old version
        n_min, pmin, n_max, pmax = FP.find_peaks(Np, Vstep, self.V, n = N)  # new version
    
        # number of maxima
        print(" found : ", n_max, " maxima")
        #number of minima
        print(" found : ", n_min, " minima")
        n_both = min(n_max, n_min)
        
        imax = pmax[:n_both]
        self.imax = imax
        imin = pmin[:n_both]
        self.imin = imin        
        # make sure the number of minima and maxima are the same
        ## get the indices of peaks higher than threshold
        ipeak_tr = np.where(self.V[imax]>Vth)[0]
        self.ipeak_tr = ipeak_tr
        #choose minimums close to previous maximum (to filter out electrical switching noise peaks)
        close_time = np.where((self.td[imin][ipeak_tr]-self.td[imax][ipeak_tr])<min_delta_t)[0] 
        self.close_time = close_time
        # check if next minimum is not too negative (removing switching noise)        
        neg_V_min = np.where(self.V[imin][ipeak_tr][close_time]<max_neg_V)[0] 
        
        self.neg_V_min = neg_V_min
        ipeak_ok = np.delete(ipeak_tr, close_time[neg_V_min])
        #ipeak_ok = np.delete(ipeak_tr, close_time)
        
        imax_fit = imax[ipeak_ok] #indices of peaks to fit in raw data array
        self.imax_fit = imax_fit
        self.ipeak_ok = ipeak_ok
        
        # calculate corrected peak positions
        if self.refine_positions:
            self.t_rf_raw, self.V_rf_raw, self.p_a = FP.refine_positions(N_close, imax_fit, self.td, self.V, self.V.size, imax_fit.size)
            # find good peaks
            sel = ~np.isnan(self.V_rf_raw)
            # make sure the new position is not more than 3 points away
            ok_fit = np.abs(self.t_rf_raw[sel] - self.td[imax_fit][sel]) < 3*self.dt
            self.imax_fit_rf = imax_fit[sel][ok_fit]
            self.tp_rf = self.t_rf_raw[sel][ok_fit]
            self.Vp_rf = self.V_rf_raw[sel][ok_fit]
            print(" Refined : ", self.tp_rf.size, " positions out of ", self.imax_fit.size)
        # set the good peaks
        self.tp = self.td[imax_fit]
        self.Vp = self.V[imax_fit]

    def plot_peak(self,i, n_close, use_raw = False):
        ip = self.imax_fit[i]
        a = self.p_a[i]
        i_start = max(ip - n_close, 0)
        i_stop = min(ip + n_close, self.V.shape[0])
        sl = slice(i_start, i_stop + 1)
        tt = self.td[sl] - self.td[ip]
        xx = np.linspace(tt.min(), tt.max(), 100)
        yy = a[0] + (a[1] + a[2]*xx)*xx
        B.pl.plot(self.td[sl], self.V[sl], '.')
        B.pl.plot(xx+self.td[ip], yy)
        B.pl.plot([self.td[ip]], [self.V[ip]], 'ro')
        


    def plot_fit_group(self, ig, shifted = False, warnings = False):
        """
        Plot data and fited values for fitgroup number i

        Parameters
        ----------
        i : int
            fitgroup number.
        shifted: bool, optional
            use shifted fit group

        Returns
        -------
        None.

        """
        # check that fit groups exist
        if (not hasattr(self, 'fg')) or (not hasattr(self, 'fg_shift')):
            print('fit groups have not been defined yet, nothing to show')
            return
        
        if shifted:
            fg = self.fg_shift
            lims = self.fg_limits_shift
        else:
            fg = self.fg
            lims = self.fg_limits
        if ig >= fg.shape[0]:
            print(f'fit group {ig} does not exist')
            return
        
        sl, fitted_A, sig_fitted_A, bkg_val, bkg, chisq, in_boundary = self.fit_fit_group(fg[ig], lims[ig],  plot_fit = True, warnings = warnings, shifted = shifted)
        print('fit returns: ', sl, fitted_A, sig_fitted_A, bkg_val, bkg, chisq, in_boundary)
        
        return 
        
    def setup_fit_groups(self):
        """
        Forming fit groups using peak width to determine boundaries
        good peak locations after electrical switching noise filtering

         - sets the values for:
             self.fg             : array of pair of indices indicating peaks for each group
             self.fg_shift       : array of pair of indices indicating peaks for each shifted group
             self.fg_n_peaks     : number of peaks in each group
             self.fg_shift_n_peaks : number of peaks in each shifted group
           

        Returns
        -------
        None.

        """

        n_peaks_to_fit = self.channel_data.par['n_peaks_to_fit']

        #--------------------------------
        # define window edges for fitting
        #--------------------------------                        
             
        # determine the fit groups
        if self.use_refined:
            imax_fit = self.imax_fit_rf
        else:
            imax_fit = self.imax_fit
        self.fg, self.fw, self.fg_limits = get_fit_groups(n_peaks_to_fit, imax_fit, 0., self.td)    
        self.fg_shift, self.fw_shift, self.fg_limits_shift = get_fit_groups(n_peaks_to_fit,imax_fit, self.fw/2., self.td) 
        
        # the number of peaks in each fit group
        self.fg_n_peaks = self.fg[:,1] - self.fg[:,0]
        self.fg_shift_n_peaks = self.fg_shift[:,1] - self.fg_shift[:,0]
        
    def find_fitgroup(self,t):
        
        # find the intervalue in the fg_limits
        return np.where((self.td[self.fg_limits[:,0]] < t) & (t <= self.td[self.fg_limits[:,1]]))[0][0]
        

    def init_fitting(self):
        """
        Setup peak fitting parameters and arrays.

        Returns
        -------
        None.

        """
        sig = self.channel_data.par['sig']
        # set values for the peak shape (should have been determined in peak_sampling)
        self.alpha = B.Parameter(1./self.channel_data.par['decay_time'], 'alpha')
        self.beta = B.Parameter(1./self.channel_data.par['rise_time'], 'beta')

        # set boundary values
        self.n_sig_low = self.channel_data.par['n_sig_low']
        self.n_sig_boundary = self.channel_data.par['n_sig_boundary']            
        # determine lower (and upper) edges
        self.dt_l = self.n_sig_low*sig  # in us
        self.dl = int(self.dt_l/self.channel_data.dt) # in channels 
                
        self.boundary = self.n_sig_boundary*sig # boundary reagion in terms of sigma (peak width)
        self.in_boundary = np.zeros_like(self.tp, dtype = 'bool')  # array of logicals indicating if a peak in a boundary region
        
        # number of bkg parameters
        self.bkg_len=self.channel_data.var['bkg_len']
        self.vary_codes_bkg=self.channel_data.var['vary_codes_bkg']
        self.poly_order=self.channel_data.par['poly_order']
        
        return



    def fit_fit_group(self, ff, lims, plot_fit = False, plot_init_only = False, warnings = True, shifted = False):
        """
        Fit a peaks withing figroup ff. init_fitting() must be called before this call. This function is normally not
        called directly by the user but by fit_data

        Parameters
        ----------
        ff : array (2)
            fit group limits (slice into the peak position array).
        lims: array(2)
            corresponding raw data limie (slice into the data array) 
        plot_fit : Bool, optional
            If True plot the data and the fit for this fit froup. The default is False.
        plot_init_only: Bool, optional
            Plot initial values only, do no fit. The default is False.

        Returns
        -------
        sl : slice
            slice into the data array used.
        fitted_A
            fitted peak height.
        sig_fitted_A
            uncertainty in peak height.
        bkg
            fitted background parameters.
        chisq
            chisquare for this fit.

        """
        # check if there are any peaks to fit to handle
        # empty fit groups
        if ff[0] == ff[1]:
            # there are no peak in this group
            no_peaks = True
            sl = slice(-1,0)
        else:
            no_peaks = False
        # time range of fit groups data
        lims_t_start = self.td[lims[0]]
        lims_t_end = self.td[lims[1]]
        # get full range data for fitting
        # slice for data range for fitting into the full array, extended by the number of sigmas selected
        i_start_data = max(0, lims[0] - self.dl)
        i_stop_data = min(lims[1] + self.dl, self.td.shape[0])
        it_fit = slice(i_start_data,i_stop_data)
        # find which peaks are in a boundary area
        if (it_fit.start == it_fit.stop):
            # nothing to fit continue
            return sl, -1, -1, [], -1.
        # boundary start and stop times
        bdry_start = lims_t_start + self.boundary
        bdry_end = lims_t_end - self.boundary
        # prepare good peak data for fitting
        if no_peaks:
            # set 0 for fitting at start of time range
            tpk = self.td[it_fit][0]
            tt = self.td[it_fit] - tpk
            Vt = self.V[it_fit] 
            Vp_fit = np.array([]).astype('float')
            tp_fit = np.array([]).astype('float')
            t_peaks = np.array([]).astype('float')
            in_boundary = np.array([]).astype('bool')
            n_peaks = 0
        else:
            if (self.use_refined):
                # only used good refined positions
                """
                Vp = self.V[self.imax_fit_rf]
                tp = self.td[self.imax_fit_rf]
                """
                Vp = self.Vp_rf
                tp = self.tp_rf
            else:
                Vp = self.Vp
                tp = self.tp
            # form a slice for the current peaks
            sl = slice(ff[0], ff[1])
            # times for the peaks
            if self.use_refined:
                tp_fit = self.tp_rf[sl]
            else:
                tp_fit = tp[sl]
            # amplitudes for the peaks
            Vp_fit = Vp[sl]
            # first peak to be fitted is at 0.
            tpk = tp_fit[0]
            # determine if the peaks are in a boundar region
            in_boundary = ((tp_fit - lims_t_start) < self.boundary) | ((lims_t_end - tp_fit) <self. boundary)
            # place first peak at 0
            tt = self.td[it_fit] - tpk
            Vt = self.V[it_fit] 
            # initialize fortran fit
            t_peaks = tp_fit - tp_fit[0]
            n_peaks = Vp_fit.shape[0]
        # initialize vary codes array
        vc = np.array(self.vary_codes_bkg + [1 for v in Vp_fit])
        # initalize fit
        LF.lfitm1.init_all(self.alpha(),self.beta(), t_peaks, n_peaks, self.poly_order, vc)
        # do the fit
        chisq = LF.lfitm1.peak_fit( tt, Vt, np.ones_like(Vt), Vt.shape[0])
        # get the amplitudes the first bkg_len parameters are the bkg.
        fitted_A = np.copy(LF.lfitm1.a[self.bkg_len:])
        # get background parameters
        bkg = np.copy(LF.lfitm1.a[:self.bkg_len])
        p = np.polynomial.Polynomial(bkg) # background polynomial
        # background value at peak location
        bkg_val = p(tp_fit - tpk)
        # get covariance matrix
        cov = np.copy(LF.lfitm1.covar)
        if self.check_cov:
            # check for negative diagonal elements
            neg_cov = cov.diagonal()<0.
            if neg_cov.max():   
                neg_el = np.where(neg_cov)[0]
                print(f'Cov. diagonal is negative at {neg_el}, limits = {lims}')
                print(f'Cov. negative values {cov.diagonal()[neg_cov]}')
        # calculate the error of the amplitudes
        sig_fitted_A = np.sqrt(np.abs(cov.diagonal()[self.bkg_len:]))* np.sqrt(abs(chisq ))
        #
        if (chisq < 0.):
            # failed fit
            if warnings:
                print(f' fit_fit_group: fit for fit limits {it_fit} failed, chisq = {chisq}')
            LF.lfitm1.free_all()
            if (plot_fit and (len(tt)!= 0) ):
                B.pl.figure()
                #B.pl.plot(tt+tpk, Vt, '.', color = 'b' )
                B.pl.plot(tt+tpk, Vt, '.', color = 'y' )
                B.pl.plot(tp_fit, Vp_fit, 'o', color = 'r')
              
                B.pl.xlabel('t(us)')
                B.pl.ylabel('V')
                if shifted:
                    B.pl.title('Shifted fit groups, bad fit')
                else:                  
                    B.pl.title('Not shifted fit groups, bad fit')
            return sl, -1, -1, -1, np.array([]), chisq, np.array([])
        if (plot_fit and (len(tt)!= 0) ):
            B.pl.figure()
            p = np.polynomial.Polynomial(bkg) # background
            #B.pl.plot(tt+tpk, Vt, '.', color = 'b' ) 
            B.pl.plot(tt+tpk, Vt, '.', color = 'y' )
            B.pl.plot(tp_fit, Vp_fit, 'o', color = 'r')
            B.pl.plot(tt+tpk, line_shape(tt), color = 'blueviolet' )
            B.pl.plot(tt+tpk, p(tt), color = 'g' )
            vy0,vy1 = B.pl.ylim()
            vx0 = self.td[lims[0]]
            vx1 = self.td[lims[1]]
            #B.pl.vlines([vx0, vx1], vy0, vy1, color = 'r')
            #B.pl.vlines([bdry_start, bdry_end], vy0, vy1, color = 'm')
            B.pl.xlabel('Time(us)',  size=16, weight='bold' )
            B.pl.ylabel('V',  size=16, weight='bold' )
            plt.yticks(fontsize=14,weight = 'bold' )
            plt.xticks(fontsize=14,weight ='bold')
            if shifted:
                B.pl.title('Shifted fit groups')
            else:
                B.pl.title('Not shifted fit groups')
            self.plot -=1
        if self.correct_data:
            # subract fitted background from data
            p = np.polynomial.Polynomial(bkg) # background polynomial
            self.V_corr[it_fit] = Vt - p(tt)

        LF.lfitm1.free_all()
        # return fit results
        return sl, fitted_A, sig_fitted_A, bkg_val, bkg, chisq, in_boundary
            
    def fit_data(self, f_start = 0, f_end = None):
        """
        Peform the fitting of both fit groups
        
         - sets the values for:
             self.A_fit      :  fitted peak height
             self.sig_A_fit  :  error in fitted peak height
             self.bkg_par    : fitted background parameters
        Returns
        -------
        None

        """                

        # initialize fitting 
        self.init_fitting()
        if f_end is None:
            f_end = self.fg.shape[0]
            
        # total number of peaks
        Np = self.Vp.shape[0]
        
        # loop over fit groups and define peaks in the boundary area
        # arrays for storing fit results
        self.A_fit = np.zeros_like(self.tp)
        self.sig_A_fit = np.zeros_like(self.tp)
        
        
        # bkg fit parameters
        self.bkg_par = np.zeros( shape = (len(self.tp), self.bkg_len)) 
        # bkg values at peak location
        self.bkg_val = np.zeros_like(self.tp)
        
        N_fitted = 0
        
        ifailed = 0
        # setup plotting if desired
        # do the fitting
        fg = self.fg[f_start:f_end]
        lims = self.fg_limits[f_start:f_end]
        fg_n_peaks = self.fg_n_peaks[f_start:f_end]
        
        n_plot = self.plot

        # start time for timer
        t_start = time.perf_counter()
        #------------------------------------------
        # loop over fit groups for fitting 1st pass
        #------------------------------------------  
        print(20*'=' + ' first pass ' + 20*'=')
        for i,ff in enumerate(fg):
            ll = lims[i]
            tt_0 = self.td[ll[0]]
            # print information on the current status of fitting
            N_fitted += fg_n_peaks[i]
            if (i%self.fit_progress == 0) & (i!=0):
                t_current = time.perf_counter()
                a_rate = 0.
                t_diff = (t_current - t_start)
                if (t_diff != 0.):
                    a_rate = float(N_fitted)/t_diff
                    frac = N_fitted/Np*100.
                print(f'Fit {i} {frac:.1f}, % completed, elapsed time {t_diff:.3f}, rate = {a_rate:.1f} Hz')
            if (n_plot > 0) and (tt_0 >= self.start_plot_at):
                n_plot -= 1
                plot_fit = True
            else:
                plot_fit = False
            sl, fitted_A, sig_fitted_A, bkg_val, bkg, chisq, in_boundary = self.fit_fit_group(ff, ll, plot_fit = plot_fit)
            # get the relevant fit parameters
            if (chisq > 0.):
                # its_shift.append((tp_fit, sl, Vp_fit, chisq, np.copy(LF.a), np.copy(LF.covar)))
                # same the values
                self.A_fit[sl] = fitted_A
                self.sig_A_fit[sl] = sig_fitted_A
                self.bkg_val[sl] = bkg_val
                self.bkg_par[sl] = bkg
                self.in_boundary[sl] = in_boundary
            else:
                ifailed += 1
                print(f'----> fit for fitgroup {i+f_start} failed chis1 = {chisq}, limits = {ll}')

        print(ifailed, ' fits failed out of', i)
        
        #------------------------------------------
        # loop over fit groups for fitting 2nd pass
        #------------------------------------------       
        
        # loop over the peaks

        t_start = time.perf_counter()
        # reset counter
        N_fitted = 0
        
        # arrays for storing fit results
        self.A_fit_s = np.zeros_like(self.tp)
        self.sig_A_fit_s = np.zeros_like(self.tp)
        # fit parameters
        self.bkg_par_s = np.zeros( shape = (len(self.tp), self.bkg_len)) 
        # bkg values at peak location
        self.bkg_val_s = np.zeros_like(self.tp)
        
        print('new start time: ', t_start)
        # fit shifted set

        n_plot_s = self.plot_s

        fg_shift = self.fg_shift[f_start:f_end]
        lims_shift = self.fg_limits_shift[f_start:f_end]
        fg_shift_n_peaks = self.fg_shift_n_peaks[f_start:f_end]
        print(20*'=' + ' second pass ' + 20*'=')
        ifailed1 = 0
        for i,ff in enumerate(fg_shift):
            ll = lims_shift[i]
            N_fitted += fg_shift_n_peaks[i]
            if (i%self.fit_progress == 0)&(i!=0):
                t_current = time.perf_counter()
                a_rate = 0.
                t_diff = (t_current - t_start)
                if (t_diff != 0.):
                    a_rate = float(N_fitted)/t_diff
                    frac = N_fitted/Np*100.                    
                    print(f'Fit {i} {frac:.1f}, % completed, elapsed time {t_diff:.3f}, rate = {a_rate:.1f} Hz')
            if (n_plot_s > 0) and (tt_0 >= self.start_plot_at):
                n_plot_s -= 1
                plot_fit = True
            else:
                plot_fit = False
            sl, fitted_A, sig_fitted_A, bkg_val, bkg, chisq, in_boundary = self.fit_fit_group(ff, ll, plot_fit = plot_fit)
            # get the relevant fit parameters
            if (chisq > 0.):
                # its_shift.append((tp_fit, sl, Vp_fit, chisq, np.copy(LF.a), np.copy(LF.covar)))
                # same the values
                self.A_fit_s[sl] = fitted_A
                self.sig_A_fit_s[sl] = sig_fitted_A
                self.bkg_val_s[sl] = bkg_val
                self.bkg_par_s[sl] = bkg
            else:
                ifailed1 += 1
                print(f'----> fit for fitgroup {i + f_start} failed chis1 = {chisq}, limits = {ll}')        
        print(ifailed1, ' fits failed out of', i)
        
        
        """                     
        copy results of those peaks that are in the boundaryregion from the 2nd fit
        in_boundary is true for all peaks that lie in a boundary region
        need to get the fit results from the shifted fit for those peaks e.g. for the indices
        """
        
        self.A_fit[self.in_boundary] = self.A_fit_s[self.in_boundary]
        self.sig_A_fit[self.in_boundary] = self.sig_A_fit_s[self.in_boundary]
        self.bkg_par[self.in_boundary] = self.bkg_par_s[self.in_boundary]
        self.bkg_val[self.in_boundary] = self.bkg_val_s[self.in_boundary]
        
        
    def save_fit(self, new_row = False, overwrite = False):
        """
        save the data as numpy compressed data files at the locations indicated by
        in the data base

        Parameters
        ----------
        new_row : Bool, optional
            Create a new row with new version number (default = False)
        overwrite: Bool, optional
            Overwrite existing file (default = False)
        Returns
        -------
        None.

        """    
        o_dir = self.channel_data.var['res_dir']
        shot = self.channel_data.par['shot']
        channel = self.channel_data.par['channel']
        version = self.channel_data.par['version']
        dbfile = self.channel_data.db_file
        
        if new_row:
            q_table  = 'Raw_Fitting'
            q_where = f'Shot = {shot} AND Channel = {channel} AND Version = {version}'
            success, version = db.duplicate_row(dbfile, q_table, q_where)
            if success:
                print(f'Created new row in {q_table} new version is {version}')
        o_file = f'{o_dir}/fit_results_{shot}_{channel:d}_{version:d}_{self.tmin:5.3f}_{self.tmax:5.3f}.npz'
        if  not os.path.exists(os.path.dirname(o_file)):
            os.makedirs(os.path.dirname(o_file))
        if (not overwrite) and os.path.isfile(o_file):
            fn = os.path.splitext(o_file)
            o_file= fn[0] + '_' + time.strftime('%d_%m_%Y_%H_%M_%S') + fn[1] 
        n_lines = self.tp.shape[0]
        np.savez_compressed(o_file, t=self.tp, V=self.Vp, A=self.A_fit, sig_A=self.sig_A_fit, bkg=self.bkg_par, bkg_val=self.bkg_val)
        print("Wrote : ", n_lines, " lines to the output file: ", o_file)
        self.last_saved_in = o_file
        # store the file name in the database
        q_table  = 'Raw_Fitting'
        q_where = f'Shot = {self.channel_data.shot} AND Channel = {self.channel_data.channel} AND Version = {version}'
        q_what = f'Result_File_Name = "{o_file}"'
        db.writetodb(self.channel_data.db_file, q_what, q_table, q_where)

        # store the inpu file name in the database
        q_table  = 'Raw_Fitting'
        q_where = f'Shot = {self.channel_data.shot} AND Channel = {self.channel_data.channel} AND Version = {version}'
        q_what = f'Input_File_Name = "{self.channel_data.data_filename}"'
        db.writetodb(self.channel_data.db_file, q_what, q_table, q_where)

        
    def save_corr(self, keep_iteration = False):
        """
        save a background subtracted file, that can be used later for a refined analysis. The file is stored
        in the same directory as the raw data

        Parameters
        ----------
        keep_iteration:  Bool, optional
            Do nto change the iteration number. Default = False
        Returns
        -------
        None.

        """    
        shot = self.channel_data.par['shot']
        channel = self.channel_data.par['channel']
        version = self.channel_data.par['version']
        dbfile = self.channel_data.db_file
        # get raw file name
        which_shot = f'Shot = {shot}'
        raw_exp_dir, raw_exp_file = db.retrieve(dbfile,  'Folder, File_Name', 'Shot_List', which_shot)[0]
        raw_name, raw_ext =  os.path.splitext(raw_exp_file)
        
        # get current exp. file name
        exp_file = self.channel_data.par['exp_file']
        
        # directory to store the corrected files
        corr_folder = raw_exp_dir + 'corrected/'
        corr_dir = self.channel_data.par['root_dir'] + corr_folder
        if  not os.path.exists(corr_dir): # create the directory if it does not exist
            os.makedirs(corr_dir)        
        
        # analyze file name to determine iteration number
        if exp_file == raw_exp_file:
            f_dir = raw_exp_dir
            f_name = raw_name
            f_ext = raw_ext
        else:
            f_dir, f_full_name = os.path.split(exp_file)
            f_name, f_ext = os.path.splitext(f_full_name)

        if f_ext in [raw_ext,'.npz']:  # a raw or filtered file file
            iteration = 0
            f_name = raw_name + f'_{shot}_{channel}_{version}_{iteration}'
        elif f_ext == '.hdf':
            ff = f_name.split('_')
            iteration = int(ff[-1])
            if not keep_iteration:
                iteration += 1  # get old iteration number and increase it
            ff[-1] = f'{iteration}'
            f_name = '_'.join(ff) # assemble new file name
        file_name = f_name + '.hdf' 
        
        o_file = corr_dir + file_name
        n_lines = self.td.shape[0]
        # create hdf data sets
        hdf_file = h5py.File(o_file, "w")
        sV = (self.V_corr*self.corrected_data_scale).astype('int16')
        # compressed data file of integers
        V_corr_dataset = hdf_file.create_dataset('V_corr', sV.shape, data = sV, compression = 'gzip')
        V_corr_dataset.attrs['t0'] = self.td[0]
        V_corr_dataset.attrs['dt'] = self.dt
        V_corr_dataset.attrs['V_corr_scale'] = self.corrected_data_scale
        hdf_file.close()
        print("Wrote : ", n_lines, " lines to the output file: ", o_file)        
        
        # store the file name in the database in the Raw_Fitting table
        q_table  = 'Raw_Fitting'
        q_where = f'Shot = {shot} AND Channel = {channel} AND Version = {version}'
        q_what = f'Corrected_Data_File_Name = "{o_file}"'
        db.writetodb(self.channel_data.db_file, q_what, q_table, q_where)      
        
        # keep track of corrected file in the Shot_list_Correted table
        q_table  = 'Shot_List_Corrected'
        q_where = f'Shot = {shot} AND Channel = {channel} AND Version = {version}'
        q_where_iter = f'Shot = {shot} AND Channel = {channel} AND Version = {version} AND Iteration = {iteration}'

        q_names =  ['Shot',    'Channel',   'Version',    'Iteration',   'File_Name',      'Folder',           'Comment']        
        q_values = [shot,      channel,     version, iteration,     f'"{file_name}"', f'"{corr_folder}"','"No Comment"' ] 
        # create of update the corresponding entry
        if not db.check_condition(self.channel_data.db_file, q_table, q_where_iter):
            db.insert_row_into(self.channel_data.db_file, q_table, q_names, q_values)
        else:
            db.update_row(self.channel_data.db_file, q_table, q_names, q_values, q_where_iter)
        # adll done

