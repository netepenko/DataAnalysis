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
from . import static_functions as fu
from . import ffind_peaks as FP
#conversion to microseconds constant
us=1.e6

Np = 12  # number of sampled peaks

class peak_sampling:
    
    def __init__(self, channel_data, psize = None, plot_peaks = True, rise_time = None, decay_time = None):
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

        self.plot_peaks = plot_peaks
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
        q_where = f'Shot = {self.shot} AND Channel = {self.channel}'
        
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
            sig, y = fu.peak(x-x0(), alpha(), beta() )
            return y*H() + offset()
        
        F=B.genfit(signal, [alpha, beta, x0, H, offset], x = ts, y = Vt)
               
        return F, alpha(), beta(), H(), offset(), x0()
        
    def fit_peaks(self, t_slices, save_fit = True, save_slices = True):
        """
        normalize and fit peaks in within the time slices t_slice
        
        - make sure that the time ranges selected in ts are larger than 
         psize*(rise_time + decay_time) otherwise the
         signal will be cut        

        Parameters
        ----------
        t_slices : numpy array (float, shape = (n,2))
            DESCRIPTION.
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
        # create common array for peak data
        Vtotal=np.zeros(int(self.psize*(self.rise_time + self.decay_time)/self.dt) )
        counters = np.zeros(int(self.psize*(self.rise_time + self.decay_time)/self.dt) )
        # local names
        Vps=self.channel_data.Vps  # digitizer data
        td=self.channel_data.td
        # save time in class if selected
        if save_slices:
            self.good_peaks = t_slices
        # psize * rise time im indices
        i_shift = int(self.psize*(self.rise_time/self.dt) )
        # loop over time slices for model peaks
        for i, ts in enumerate(t_slices):
            tmin = ts[0]
            tmax =ts[1]
            # calculate a slice
            sl=fu.get_window_slice(tmin, td, tmax)
            Vmax = Vps[sl].max()
            imax = Vps[sl].argmax()
            Vt = Vps[sl]/Vmax
            ts = td[0:Vt.shape[0]]-td[0]
            F, alpha, beta, H, offset, x_0 = self.fit_shape(ts,Vt)
            if self.plot_peaks: #plot_peak_samples:
                    B.pl.figure()
                    B.pl.plot(ts, Vps[sl], '*', label='Peak %d original')
                    B.pl.plot(ts, Vt, '.', label='Peak %d Normalized' %i)
                    B.pl.plot(ts,F.func(ts), color='b', label='Fit_line')
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
        if (save_fit) :
            # get width of peak:
            sig = fu.peak(F.xpl-x0, alpha, beta)[0]
            self.decay_time = 1/alpha
            self.rise_time = 1/beta
            self.sig = sig
        else:  
            return F, alpha, beta, H, offset, x0
        # all done               
    
    
    def find_good_peaks(self, tmin, tmax, Vstep, Vthres):
        """
         search for good model peaks between tmin and tmax.
         
         - the data are searched to find Np good peaks
         - once found the time slices are stored
         - the peaks are fitted and a common peak shape is determined and saved

        Parameters
        ----------
        tmin : float
            minimum time.
        tmax : float
            maximum time.
        Vstep : float
            Voltage step for peak finding algorithm.
        Vthres : float
            Threshold value for peak height.

        Returns
        -------
        None.

        """
        #dtmin, dtmax time interval for finding good nodel peaks
                    
        print('Looking for good peaks in interval (%f, %f)' %(tmin, tmax))
        
        sl = fu.get_window_slice(tmin, self.td, tmax)
        Vps=self.channel_data.Vps[sl]
        td=self.channel_data.td[sl]
        
        #time step of data        
        psize = self.psize
        # find peaks first and then select those above threshold for good peak
        results = np.zeros((2,), dtype = 'int32')
        pmin = np.zeros((len(Vps)/psize, ), dtype='int32')
        pmax = np.zeros((len(Vps)/psize, ), dtype='int32')
        FP.find_peaks(len(Vps), Vstep, Vps, results, pmin, pmax) 
        
        # number of maxima
        nmax = results[1]
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
            sl=fu.get_window_slice(t_start, td, t_stop )
            # normalize peak shape, Vp[i] is maximum value
            Vt = Vps[sl]/(Vp[i])
            # create common tinae array starting at 0
            ts = td[0:Vt.shape[0]]-td[0]
            # fit this peak
            F, alpha, beta, H, offset, x0 = self.fit_shape(ts,Vt)
            # increment current peak index
            if F.chi2 > self.chi2: 
                print("bad peak")
                if (i >= ngood_peaks):
                    print("Finding good peaks failed after %d attempts." %i)
                    return
            else:
                # add good peak to list
                self.good_peak_times.append([t_start, t_stop])                    
                if self.plot_peaks: #plot_peak_samples:
                        B.pl.figure()
                        B.pl.plot(ts, Vps[sl], '*', label='Peak %d original')
                        B.pl.plot(ts, Vt, '.', label='Peak %d Normalized' %j)
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
        self.fit_peaks(self.good_peak_times,Vps, save_fit = True)          

            
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
        q_where = f'Shot = {self.shot} AND Channel = {self.channel}'
        
        # store the good peak time slices
        q_what = (''.join([f'b{i} = {tt[0]/us}, e{i} = {tt[1]/us}, ' for i,tt in enumerate(self.good_peak_times)] )).strip()[-1] # skip last comma
                
        db.writetodb(dbfile, q_what, q_table, q_where)

        # store the shape parameters
        q_what = f'decay_time = {self.decay_time/us} , rise_time = {self.rise_time/us}'
        db.writetodb(dbfile, q_what, q_table, q_where)
        
        q_table = 'Raw_Fitting'
        q_what = f'sig = {self.sig/us}' 
        db.writetodb(dbfile, q_what, q_table, q_where)    
        
       
