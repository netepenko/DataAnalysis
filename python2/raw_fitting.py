# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:05:53 2016

@author: Alex
"""
import numpy as np
import LT.box as B
import matplotlib.pyplot as pl
import ffind_peaks as FP
from lfitm1 import lfitm1 as LF
import time
import static_functions as fu
import os
#conversion to microseconds constant
us=1.e6


def fit_interval(self, tmin=None, tmax=None, plot=None, save=None):
        
        sl = fu.get_window_slice(tmin*us, self.td, tmax*us)
        V=self.Vps[sl]
        td=self.td[sl]
        dt=self.dt
        
#        Vline=np.array([]) #to save fitted line
#        tline=np.array([]) #to save time point for fitted line (even though they should be same as td)
        
        if len(V) > 1000000: 
            print 'Too much to plot, will skip plotting.'
            plot=False
        plot_s=False #do not plot if interval is not specified
        
        
        
        Vstep=self.par['Vstep']
        Vth=self.par['Vth']
        
        sig=self.par['sig']
        
        if sig == None:
            dtmin=self.par['dtmin']
            n_samp = self.par['n_samp']
            ts = td[0:n_samp]-dtmin
            sig=fu.peak(ts, 1/self.par['decay_time'], 1/self.par['rise_time'])[0]
        
        bkg_len=self.var['bkg_len']
        vary_codes_bkg=self.var['vary_codes_bkg']
        poly_order=self.par['poly_order']

        def lish(x):        #''' Vectorize the line shape'''
            return LF.line_shape(x)
        line_shape = np.vectorize(lish)
        
        alpha = B.Parameter(1./self.par['decay_time'], 'alpha')
        beta = B.Parameter(1./self.par['rise_time'], 'beta')
        
        results = np.zeros((2,), dtype = 'int32')
        pmin = np.zeros((len(V)/5, ), dtype='int32')
        pmax = np.zeros((len(V)/5, ), dtype='int32')
        FP.find_peaks(len(V), Vstep, V, results, pmin, pmax)
    
        ## number of maxima
        n_max = results[1]
        print " found : ", n_max, " maxima"
        imax = pmax[:n_max]
        
        ## number of minima
        nmin = results[0]
        print " found : ", nmin, " minima"
        imin = pmin[:nmin]
        
        ## get the indices of peaks higher then threshold
        ipeak_tr = np.delete(np.where(V[imax]>Vth)[0],-1)
        
        #choose minimums close to previous maximum (to filter out electrical switching noise peaks)
        closete = np.where((td[imin][ipeak_tr]-td[imax][ipeak_tr])<0.5)[0] 
        
        # check if next minimum is not too negative (removing switching noize)        
        clles = np.where(V[imin][ipeak_tr][closete]<-0.3)[0] 
        ipeak = np.delete(ipeak_tr, closete[clles])
        
        
        # peak locations after electrical switching noise filtering 
        Vp = V[imax][ipeak]
        tp = td[imax][ipeak]
        
        
        imax_fit = imax[ipeak] #indeces of peaks to fit in raw data array 
        # loop over the peaks
        Np = Vp.shape[0]
        t_start = time.clock()
        
        # forming fit groups
        n_sig_low = self.par['n_sig_low']
        n_sig_boundary = self.par['n_sig_boundary']
        n_peaks_to_fit = self.par['n_peaks_to_fit']

        #--------------------------------
        # define window edges for fitting
        #--------------------------------                        
       
        # lower edge
        dt_l = n_sig_low*sig
        dl = int(dt_l/dt)
        
        
        boundary = n_sig_boundary*sig
        in_boundary = (np.zeros_like(tp) == 1)
        
        # determine the fit groups
        fg, fw = fu.get_fit_groups_new(n_peaks_to_fit, imax_fit, 0., td)    
        fg_shift, fw = fu.get_fit_groups_new(n_peaks_to_fit, imax_fit, fw/2., td) 
        #if interval is specified for fitting
        if tmin and tmax:
            
            if (tmin*us>=self.par['dtmin']) and (tmax*us<=self.par['dtmax']) and tmin<tmax:
                #find fit groups covering the interval
#                print td[imax_fit][0], td[imax_fit][-1], tmin*us, tmax*us
                inmin=np.where(td[imax_fit]>=tmin*us)[0][0] #index of first peak in fitting interval in time data
                in_max=np.where(td[imax_fit]<=tmax*us)[0][-1] #index of tmax in time data
                gtf=np.empty((0,2), int)    #list of groups in interval to fit
                gtfs=np.empty((0,2), int)
                for f in fg:
                    if f[0]>in_max and f[1]>in_max:
                        break
                    if f[0]>inmin or f[1]>inmin:
                        gtf=np.vstack([gtf,f])
                for f in fg_shift:
                    if f[0]>in_max and f[1]>in_max:
                        break
                    if f[0]>inmin or f[1]>inmin:
                        gtfs=np.vstack([gtfs,f])
                            
                #plot=True
                #pl.figure()
                fg = gtf
                fg_shift=gtfs #to fit only specified groups
            else:
                print "Interval out of range or incorrect.", self.par['dtmin']/us, self.par['dtmax']/us
                return
        
        # the number of peaks in each fit group
        fg_n_peaks = fg[:,1] - fg[:,0] + 1
        fg_shift_n_peaks = fg_shift[:,1] - fg_shift[:,0] + 1

        # loop over fit groups and define peaks in the boundary area
        # arrays for storing fit results
        A_fit = np.zeros_like(tp)
        sig_A_fit = np.zeros_like(tp)
        
        # bkg fit parameters
        bkg_par = np.zeros( shape = (len(tp), bkg_len)) 
        
        
        #--------------------------------                        
        #--------------------------------
        # loop over fit groups for fitting
        N_fitted = 0
        lims = []
        
        ifailed = 0
        if plot:
            pl.xlabel('t(us)')
            pl.ylabel('V')
            pl.title('Not shifted fitting groups')
        for i,ff in enumerate(fg[:]):
            N_fitted += fg_n_peaks[i]
            if (i%10 == 0) & (i!=0):
                t_current = time.clock()
                a_rate = 0.
                t_diff = (t_current - t_start)
                if (t_diff != 0.):
                    a_rate = float(N_fitted)/t_diff
                print "Fit ", i, float(N_fitted)/Np*100., "% completed, time ", t_current, ' rate =', a_rate
            # form a slice for the current peaks
            sl = slice(ff[0], ff[1]+1)
            # times for the peaks
            tp_fit = tp[sl]
            # amplitudes for the peaks
            Vp_fit = Vp[sl]
            # array indices into full data arrays for the peaks
            ipos_fit = imax_fit[sl]
            # first peak to be fitted is at 0.
            tpk = tp_fit[0]
            # get full range data for fitting
            # index of peak into the data array
            first_peak_ind = ipos_fit[0]
            last_peak_ind = ipos_fit[-1] 
            # slice for data range for fitting into the full array 
            i_start_data = max(0, first_peak_ind - dl)
            i_stop_data = min(last_peak_ind + dl, td.shape[0])
            it_fit = slice(i_start_data,i_stop_data)
            # find which peaks are in a boundary area
            if (it_fit.start == it_fit.stop):
                # nothing to fit continue
                continue
            start_fit_time  = td[it_fit][0]
            end_fit_time = td[it_fit][-1]
            lims.append([start_fit_time, end_fit_time, tpk, it_fit])
            # determine of the peaks are in a boundar region
            in_boundary[sl] = ((tp_fit - start_fit_time) < boundary) | ((end_fit_time - tp_fit) < boundary)
            # place first peak at 0
            tt = td[it_fit] - tpk
            Vt = V[it_fit] 
            # initialize fortran fit
            t_peaks = tp_fit - tp_fit[0]
            n_peaks = Vp_fit.shape[0]
            # initialize vary codes array
            vc = np.array(vary_codes_bkg + [1 for v in Vp_fit])
            # initalize fit
            LF.init_all(alpha(),beta(), t_peaks, n_peaks, poly_order, vc)
            # do the fit
            chisq = LF.peak_fit( tt, Vt, np.ones_like(Vt), Vt.shape[0])
            if (chisq < 0.):
                # failed fit
                print ' fit ', i, 'failed, chisq = ', chisq
                ifailed += 1
                LF.free_all()
                continue
           
#            if len(tt)!=0:
#                Vline=np.concatenate((Vline, line_shape(tt)))
#                tline=np.concatenate((tline, tt+tpk))
#            # plot fitting results for check if interval specified
            
            if plot and len(tt)!=0:
                pl.plot(tt+tpk, Vt, '.', color = 'b' ) 
                pl.plot(tt+tpk, line_shape(tt), color = 'm' )
                
            # get the amplitudes the first 3 parameters are the bkg.
            fitted_A = np.copy(LF.a[bkg_len:])
            # get background parameters
            bkg = np.copy(LF.a[:bkg_len])
            # get covariance matrix
            cov = np.copy(LF.covar)
            # calculate the error of the amplitudes
            sig_fitted_A = np.sqrt( cov.diagonal()[bkg_len:]*chisq )
            # get the relevant fit parameters
            if (chisq > 0.):
                # its_shift.append((tp_fit, sl, Vp_fit, chisq, np.copy(LF.a), np.copy(LF.covar)))
                # same the values
                A_fit[sl] = fitted_A
                sig_A_fit[sl] = sig_fitted_A
                bkg_par[sl] = bkg         
            # free the arrays
            LF.free_all()

        
        print ifailed, ' fits failed out of', i
#        if tmin and tmax:
#            return #don't do the second pass if checking the fitting in specified interval
#        
        #--------------------------------                        
        # get second pass for boundary fit
        #--------------------------------
        
        # loop over the peaks
        lims_s = []
        t_start = time.clock()
        
        N_fitted = 0
        
        # arrays for storing fit results
        A_fit_s = np.zeros_like(tp)
        sig_A_fit_s = np.zeros_like(tp)
        
        # fit parameters
        bkg_par_s = np.zeros( shape = (len(tp), bkg_len)) 
        
        print 'new start time: ', t_start
        # fit shifted set
        
        ifailed1 = 0
        for i,ff in enumerate(fg_shift[:]):
            N_fitted += fg_shift_n_peaks[i]
            if (i%10 == 0)&(i!=0):
                t_current = time.clock()
                a_rate = 0.
                t_diff = (t_current - t_start)
                if (t_diff != 0.):
                    a_rate = float(N_fitted)/t_diff
                print "Fit ", i, float(N_fitted)/Np*100., "% completed, time ", t_current, ' rate =', a_rate
            # form a slice for the current peaks
            sl = slice(ff[0], ff[1]+1)
            # times for the peaks
            tp_fit = tp[sl]
            # amplitudes for the peaks
            Vp_fit = Vp[sl]
            # array indices into full data arrays for the peaks
            ipos_fit = imax_fit[sl]
            # first peak to be fitted is at 0.
            tpk = tp_fit[0]
            # get full range data for fitting
            # index of peak into the data array
            first_peak_ind = ipos_fit[0]
            last_peak_ind = ipos_fit[-1] 
            # slice for data range for fitting into the full array 
            i_start_data = max(0, first_peak_ind - dl)
            i_stop_data = min(last_peak_ind + dl, td.shape[0])
            it_fit = slice(i_start_data,i_stop_data)
            # find which peaks are in a boundary area
            if (it_fit.start == it_fit.stop):
                # nothing to fit continue
                continue
            start_fit_time  = td[it_fit][0]
            end_fit_time = td[it_fit][-1]
            lims_s.append([start_fit_time, end_fit_time, tpk, it_fit])
            # place first peak at 0
            tt = td[it_fit] - tpk
            Vt = V[it_fit] 
            # initialize fortran fit
            t_peaks = tp_fit - tp_fit[0]
            n_peaks = Vp_fit.shape[0]
            # initialize vary codes array
            vc = np.array(vary_codes_bkg + [1 for v in Vp_fit])
            # initalize fit
            LF.init_all(alpha(),beta(), t_peaks, n_peaks, poly_order, vc)
            # do the fit
            chisq = LF.peak_fit( tt, Vt, np.ones_like(Vt), Vt.shape[0])
            if (chisq < 0.):
                # failed fit
                print ' fit ', i, 'failed, chisq = ', chisq
                ifailed1 += 1
                LF.free_all()
                continue
            
                
            # plot result if interval specified
            if plot_s:
                pl.figure()
                if len(tt)!=0:                    
                    pl.plot(tt+tpk, Vt, '.', color = 'b')
                    pl.plot(tt+tpk, line_shape(tt), color = 'm')
                pl.title('Shifted fitting groups')
            # save the parameters
            # get the amplitudes the first 3 parameters are the bkg.
            fitted_A = np.copy(LF.a[bkg_len:])
            # get background parameters
            bkg = np.copy(LF.a[:bkg_len])
            # get covariance matrix
            cov = np.copy(LF.covar)
            # calculate the error of the amplitudes
            sig_fitted_A = np.sqrt( cov.diagonal()[bkg_len:]*chisq )
            # get the relevant fit parameters
            if (chisq > 0.):
                # its_shift.append((tp_fit, sl, Vp_fit, chisq, np.copy(LF.a), np.copy(LF.covar)))
                # same the values
                A_fit_s[sl] = fitted_A
                sig_A_fit_s[sl] = sig_fitted_A
                bkg_par_s[sl] = bkg 
            # free the arrays
            LF.free_all()
        
        print ifailed1, ' fits failed out of', i
        
        
        #--------------------------------                        
        # copy results of those peaks that are in the boundaryregion from the 2nd fit
        # in_boundary is try for all peaks that lie in a boundary region
        # need to get the fit results from the shifted fit for those peaks e.g. for the indices
        #--------------------------------
        
        A_fit[in_boundary] = A_fit_s[in_boundary]
        sig_A_fit[in_boundary] = sig_A_fit_s[in_boundary]
        bkg_par[in_boundary] = bkg_par_s[in_boundary]
        
        
        #---------------------------------------------           
        # save the data as numpy compressed data files
        #---------------------------------------------
        if not save:
            print 'no results saved!'
        else:
            # save the fitted data
            o_file = self.var['res_dir'] + "fit_results_"+ str(self.par['shot']) + "_{0:5.3f}_{1:5.3f}_{2:d}.npz".format(tmin, tmax, self.par['channel'])
            if  not os.path.exists(os.path.dirname(o_file)):
                os.makedirs(os.path.dirname(o_file))
            if os.path.isfile(o_file):
                o_file= self.var['res_dir'] + "fit_results_"+ str(self.par['shot']) + "_{0:5.3f}_{1:5.3f}_{2:d}".format(tmin, tmax, self.par['channel'])+time.strftime('%d_%m_%Y_%H_%M_%S')+".npz" 
            n_lines = tp.shape[0]
            np.savez_compressed(o_file, t=tp, V=Vp, A=A_fit, sig_A=sig_A_fit, bkg=bkg_par)
            print "Wrote : ", n_lines, " lines to the output file"

            #save fitted lines to the file
#            np.savez_compressed(self.var['res_dir'] + "fit_line_"+ str(self.par['shot']) + ".npz", Vline=Vline, tline=tline)
#            print "saved line"