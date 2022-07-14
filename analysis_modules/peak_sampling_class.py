# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:49:02 2016

@author: Alex

         WB May 6 2022
         

- Previoualy is part of the channel_data_class
- New: this is its own class

- all times in us

- The normal usage would be something like:
    import peak_sampling as PS
    
    sel = (0.1 <t )&(t<.15)
    
    ps = PS.peak_sampling(t[sel], V[sel], 29879, 0)
    
    t_min = .1; t_max = .15; dV = .1; V_min = .2
    
    ps.find_good_peaks( t_min, t_max, dV, V_min)
    
    ps.save_parameters('MyDbfile')
    

"""
import numpy as np
import LT.box as B

from . import database_operations as db
from . import utilities as UT
from . import ffind_peaks2 as FP
#conversion to microseconds constant
us=1.e6

Np = 12  # number of sampled peaks

class peak_sampling:
    
    def __init__(self, channel_data, chi2 = None, psize = None, plot_single_peaks = True, plot_common_peak = True, rise_time = None, decay_time = None):
        """
        peak sampling

        Parameters
        ----------
        td : float array
            time.
        Vps : float array
            digitizer signal (V)
        shot : int
            shot number
        psize : int 
            parameter to size peak shape array, optional. The default is 5.
        plot_peaks (optional): Bool
            plot found peaks
        rise_time: float
            initial value for signal rise time
        decay_time: float
            initial value for signal decay time

        Returns
        -------
        
        peak_sampling instance

        """
        self.channel_data = channel_data
        
        if chi2 is None:
            self.chi2 = self.channel_data.par['Chi2']
        else:
            self.chi2 = chi2    
        
        if psize is None:
            self.psize = self.channel_data.psize
        else:
            self.psize = psize
            self.channel_data.psize = psize
            
        if rise_time is None:
            self.rise_time = self.channel_data.par['rise_time']
        else:
            self.rise_time = rise_time
            self.channel_data.par['rise_time'] = rise_time
        if decay_time is None:
            self.decay_time = self.channel_data.par['decay_time']
        else:
            self.decay_time = decay_time
            self.channel_data.par['decay_time'] = decay_time

        self.plot_single_peaks = plot_single_peaks
        self.plot_common_peak = plot_common_peak
        self.good_peak_times = []
        
    def load_times(self, dbfile):
        """
        load time slices for model peaks from the data base

        Parameters
        ----------
        dbfile : str
            databas file name.

        Returns
        -------
        None.

        """
        
        q_from  = 'Peak_Sampling'
        q_where = f'Shot = {self.shot} AND Channel = {self.channel} AND Version = {self.channel_data.Version}'
        
        q_what = 'b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12'  
        tsmin = np.asarray(db.retrieve(dbfile, q_what, q_from, q_where) ) *us
        
        q_what = 'e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12'
        tsmax = np.asarray(db.retrieve(dbfile, q_what, q_from, q_where) ) *us
        
        self.good_peak_times = np.array([tsmin, tsmax]).T

            
    def fit_shape(self, ts, Vt):
        """
        Fit standard peak shape to data ts, Vt

        Parameters
        ----------
        ts : numpy array (float)
            times
        Vt : numpy array (float)
            Voltages

        Returns
        -------
        F : genfit opject
            fit results.
        alpha: float
            fit parameter (1/rise_time)
        beta: float
            fit parameter. (1/decay_time)
        H: float
            fit parameter. (signal height)
        offset: float
            fit parameter (constant background)
        x0: float
            fit parameter (peak position)

        """
        alpha = B.Parameter(1./self.decay_time, 'alpha')
        beta = B.Parameter(1./self.rise_time, 'beta')
        x0 = B.Parameter(ts[np.argmax(Vt)], 'x0') 
        H = B.Parameter(1.,'H')
        offset = B.Parameter(0., 'offset')         
        
        # shift peak shape 
        def signal(x):
            sig, y = UT.peak(x-x0(), alpha(), beta() )
            return y*H() + offset()
        
        F=B.genfit(signal, [alpha, beta, x0, H, offset], x = ts, y = Vt, plot_fit = False)
               
        return F, alpha(), beta(), H(), offset(), x0()
        
    def fit_peaks(self, t_slices = None, save_fit = True, save_slices = True):
        """
        normalize and fit peaks in within the time slices t_slice
        
        - make sure that the time ranges selected in ts are larger than 
         psize*(rise_time + decay_time) otherwise the
         signal will be cut        

        Parameters
        ----------
        t_slices : numpy array (float, shape = (n,2)), optional
            array of time slices for good peaks (normally calculated in find_good_peaks)
        save_fit : Bool, optional
            save the fit results in class. The default is True.
        save_slices : Bool, optional
            save the time slices in class. The default is True.

        Returns (if save_fit == False)
        -------
        F : genfit opject
            fit results.
        alpha: float
            fit parameter (1/rise_time)
        beta: float
            fit parameter. (1/decay_time)
        H: float
            fit parameter. (signal height)
        offset: float
            fit parameter (constant background)
        x0: float
            fit parameter (peak position)

        

        """
        # the the stored values by default
        if t_slices is None:
            t_slices = self.good_peak_times
        
        # create common array for peak data
        Vtotal=np.zeros(int(self.psize*(self.rise_time + self.decay_time)/self.channel_data.dt) )
        counters = np.zeros(int(self.psize*(self.rise_time + self.decay_time)/self.channel_data.dt) )
        # local names
        Vps=self.channel_data.Vps  # digitizer data
        td=self.channel_data.td
        # save time in class if selected
        if save_slices:
            self.good_peaks = t_slices
        # psize * rise time im indices
        i_shift = int(self.psize*(self.rise_time/self.channel_data.dt) )
        # loop over time slices for model peaks
        for i, ts in enumerate(t_slices):
            tmin = ts[0]
            tmax =ts[1]
            # calculate a slice
            sl=UT.get_window_slice(tmin, td, tmax)
            Vmax = Vps[sl].max()
            imax = Vps[sl].argmax()
            Vt = Vps[sl]/Vmax
            ts = td[0:Vt.shape[0]]-td[0]
            F, alpha, beta, H, offset, x_0 = self.fit_shape(ts,Vt)
            if self.plot_single_peaks: #plot_peak_samples:
                    B.pl.figure()
                    B.pl.plot(ts, Vps[sl], '*', label=f'original peak {i}')
                    B.pl.plot(ts, Vt, '.', label=f'normalized peak {i}')
                    B.pl.plot(ts,F.func(ts), color='b')
                    B.pl.legend()
                    B.pl.xlabel('t(us)')
                    B.pl.ylabel('V')
                    B.pl.axis('tight')
            # normalize peak height to 1
            Vt = (Vt-offset)/H
            # shift the peaks to make sure that the maximum is at the same position
            for i, V in enumerate(Vtotal):
                try:
                    Vtotal[i] = Vtotal[i] + Vt[i + imax -  i_shift]  # start all 
                    counters[i] += 1
                except:
                    pass
        ttotal = td[0:Vtotal.shape[0]]-td[0]
        # calculate overage values 
        sel = counters > 0.
        Vtotal[sel] =  Vtotal[sel] / counters[sel]
        F, alpha, beta, H, offset, x0 = self.fit_shape(ttotal[sel],Vtotal[sel])   
        if self.plot_common_peak: #plot average peak shape and fit
                B.pl.figure()
                tts = ttotal[sel]
                Vts = Vtotal[sel]
                B.pl.plot(tts, Vts, '.', label = 'averaged signal')
                B.pl.plot(tts,F.func(tts), color='b', label = 'Common peak shape')
                B.pl.legend()
                B.pl.xlabel('t(us)')
                B.pl.ylabel('V')
                B.pl.axis('tight')

        if (save_fit) :
            # get width of peak:
            sig = UT.peak(F.xpl-x0, alpha, beta)[0]
            self.decay_time = 1/alpha
            self.rise_time = 1/beta
            self.sig = sig
        else:  
            return F, alpha, beta, H, offset, x0
        # all done               
    
    
    def find_good_peaks(self, tmin = None, tmax = None, Vstep = None, Vthres = None):
        """
         search for good model peaks between tmin and tmax.
         
         - the data are searched to find Np good peaks
         - once found the time slices are stored
         - the peaks are fitted and a common peak shape is determined and saved
         - default values are taken from database

        Parameters
        ----------
        tmin : float, optional
            minimum time.
        tmax : float, optional
            maximum time.
        Vstep : float, optional
            Voltage step for peak finding algorithm.
        Vthres : float, optional
            Threshold value for peak height.

        Returns
        -------
        None.

        """
        # get default values from database
                    
        if tmin is None:
            tmin = self.channel_data.par['ps_tmin']
        if tmax is None:
            tmax = self.channel_data.par['ps_tmax']
        if Vstep is None:
            Vstep = self.channel_data.par['ps_Vstep']
        if Vthres is None:
            Vthres = self.channel_data.par['ps_Vth']
        
       
        print('Looking for good peaks in interval (%f, %f)' %(tmin, tmax))
        
        sl = UT.get_window_slice(tmin, self.channel_data.td, tmax)
        Vps=self.channel_data.Vps[sl]
        td=self.channel_data.td[sl]
        
        #time step of data        
        psize = self.psize
        # find peaks first and then select those above threshold for good peak
        """
        results = np.zeros((2,), dtype = 'int32')
        pmin = np.zeros(int(len(Vps)/psize, ), dtype='int32')
        pmax = np.zeros(int(len(Vps)/psize, ), dtype='int32')
        FP.find_peaks(len(Vps), Vstep, Vps, results, pmin, pmax) 
        """
        
        N_dat = len(Vps)
        N_p = int(N_dat/psize)
        nmin, pmin, nmax, pmax = FP.find_peaks(N_dat, N_p, Vstep, Vps)
        
        # number of maxima
        print(" found : ", nmax, " maxima")
        # store locations of peaks
        imax = pmax[:nmax]
        # get the indices of peaks higher then threshold, last peak removed
        ipeak_tr = imax[Vps[imax] > Vthres]
        ngood_peaks = ipeak_tr.shape[0]
        # times and posistions for potentially good peaks
        tp = td[ipeak_tr]
        Vp = Vps[ipeak_tr]
        #go through found peaks and determine the shape of peak
        #by requiring chi square to be less than some value set in GUI
        # initialize counters
        i=1
        j=0            
        while j < Np and i < ngood_peaks:
            # calculate common window size for peak fitting
            
            t_start = tp[i]-self.psize*self.rise_time
            t_stop = tp[i]+self.psize*self.decay_time
            sl=UT.get_window_slice(t_start, td, t_stop )
            # normalize peak shape, Vp[i] is maximum value
            Vt = Vps[sl]/(Vp[i])
            # create common tinae array starting at 0
            ts = td[0:Vt.shape[0]]-td[0]
            # fit this peak
            F, alpha, beta, H, offset, x0 = self.fit_shape(ts,Vt)
            # increment current peak index
            if F.chi2 > self.chi2: 
                print("--------------->bad peak, ignore <-----------------")
                if (i >= ngood_peaks):
                    print("Finding good peaks failed after %d attempts." %i)
                    return
            else:
                # add good peak to list
                self.good_peak_times.append([t_start, t_stop])
                print(f'plot peak {j}')                    
                if self.plot_single_peaks: #plot_peak_samples:
                        B.pl.figure()
                        B.pl.plot(ts, Vps[sl], '*', label=f'original peak {j}')
                        B.pl.plot(ts, Vt, '.', label=f'normalized peak {j}')
                        B.pl.plot(ts,F.func(ts), color='b', label='Fit_line')
                        B.pl.legend()
                        B.pl.xlabel('t(us)')
                        B.pl.ylabel('V')
                        B.pl.axis('tight')
                        # B.pl.savefig("../Analysis_Results/%d/Good_peaks/Peak_%d.png" %(self.par['shot'], j))
                j+=1
            i+=1 # increment current peak counters
            # print(i, j)
        # calculate the average of all renormalized peaks
        # now calculate average signal
        #self.fit_peaks(self.good_peak_times, save_fit = True)          

            
    def save_parameters(self, dbfile):
        """
        save fit results in data base

        Parameters
        ----------
        dbfile : str
            database file name.

        Returns
        -------
        None.

        """
        # 
        q_table  = 'Peak_Sampling'
        q_where = f'Shot = {self.channel_data.shot} AND Channel = {self.channel_data.channel} AND Version = {self.channel_data.Version}'
        
        # store the good peak time slices
        q_what = (''.join([f'b{i+1} = {tt[0]/us}, e{i+1} = {tt[1]/us}, ' for i,tt in enumerate(self.good_peak_times)] )).strip()[:-1] # skip last comma
                
        db.writetodb(dbfile, q_what, q_table, q_where)

        # store the shape parameters
        q_what = f'decay_time = {self.decay_time/us} , rise_time = {self.rise_time/us}'
        db.writetodb(dbfile, q_what, q_table, q_where)
        
        q_table = 'Raw_Fitting'
        q_what = f'sig = {self.sig/us}' 
        db.writetodb(dbfile, q_what, q_table, q_where)    
        
       
