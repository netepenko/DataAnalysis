# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:49:02 2016

@author: Alex
"""
import os
import numpy as np
import LT.box as B
import matplotlib.pyplot as pl
import database_operations as db
import static_functions as fu
import ffind_peaks as FP
#conversion to microseconds constant
us=1.e6

#method for selecting peaks, plots data at start of selection, or if plot parameter set to True
def select_peak(self, n=0, plot=False):
        if n>0 and n<13 and isinstance(n, int):
            self.var['peak_num'] = n
        if self.var['peak_num'] == 13:
            return
        pn = self.var['peak_num']
        if not self.var['data_plot']:
            print "-----------------------------------------------------------"
            print "Find 12 peaks representative of the average-shaped         "
            print "peak in the raw data. The peak height does not matter.     "
            print "-----------------------------------------------------------"
            print "To record the peaks you find in DB file                    "
            print "zoom into an appropriate peak and call select_peak()       "
            print "Enter peak number as parameter if you want to start not    "
            print "from the first peak and plot=True to plot data             "
            print "-----------------------------------------------------------"
            
            # ----------------------------------------------------
            # Plot Data-------------------------------------------
            # ----------------------------------------------------
            self.var['data_plot'] = pl.figure()
            pl.plot(self.td,self.Vps,'.')
            pl.show()
            return
        
        try:
            x,y = pl.xlim()
        except:
            print "No figure exists with raw data plotted. Call plot_raw() to plot."
        
        db.writetodb('b'+str(pn)+'='+str(x)+', e'+str(pn)+'='+str(y), 'Peak_Sampling',
                     'Shot = '+str(self.par['shot'])+' AND Channel = '+str(self.par['channel']))
        print 'Peak %d added. ' % self.var['peak_num']
        self.var['peak_num'] +=1 #increment added peak number
        

  
   
                         
   

#fit sample peaks and write shape parameters in db
def peak_shape(self):
        
        Vps=self.Vps
        td=self.td
        
        dtmin=self.par['dtmin']
        n_samp = self.par['n_samp'] #number of ponints of peak shape
        n_max = self.par['n_max'] #position of maximum (index)
        n_below = self.par['n_below'] # number of channels below n_below
        n_above = self.par['n_above'] # number of channels above n_above

#        print "-----------Loading selected peaks data-------------------"
#        tsmin = np.asarray(db.retrieve('b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12', \
#        'Peak_Sampling', 'Shot = ' + str(self.par['shot']) +' AND ' + 'Channel = ' +  \
#        str(self.par['channel'])))*us
#        tsmax = np.asarray(db.retrieve('e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12', \
#        'Peak_Sampling', 'Shot = ' + str(self.par['shot']) +' AND ' + 'Channel = ' +  \
#        str(self.par['channel'])))*us

        #find peaks first and then select those above threshold for good peak
        Vgp=self.Vgp #put in DB later
        Vstep=self.par['Vstep']
        results = np.zeros((2,), dtype = 'int32')
        pmin = np.zeros((len(Vps)/5, ), dtype='int32')
        pmax = np.zeros((len(Vps)/5, ), dtype='int32')
        FP.find_peaks(len(Vps), Vstep, Vps, results, pmin, pmax) 
        
        ## number of maxima
        nmax = results[1]
        print " found : ", nmax, " maxima"
        imax = pmax[:nmax]
        ## number of minima
        nmin = results[0]
        print " found : ", nmin, " minima"
        imin = pmin[:nmin]
        
        ## get the indices of peaks higher then threshold
        ipeak_tr = np.delete(np.where(Vps[imax]>Vgp)[0],-1)
        
        # peak locations after electrical switching noise filtering 
        Vp = Vps[imax][ipeak_tr]
        tp = td[imax][ipeak_tr]
        dt=self.dt
        imax_fit = imax[ipeak_tr] #indeces of peaks to fit in raw data array
        tssa = [] #list of time slices of selected peaks
        
        #go though found peaks and determine the shape of peak
        #by setting fit error to be less then some value
        
        # fit function
        alpha = B.Parameter(1./self.par['decay_time'], 'alpha')
        beta = B.Parameter(1./self.par['rise_time'], 'beta')
        x0 = B.Parameter(self.par['position'], 'x0') 
        H = B.Parameter(1.,'H')
        offset = B.Parameter(0., 'offset') #self.offset        
        
        # shift peak shape 
        def signal(x):
            sig, y = fu.peak(x-x0(), alpha(), beta() )
            return y*H() + offset()
        
        # fit all peaks with a constant back ground
        H_a = []
        alpha_a = []
        beta_a = []
        x0_a = []
        offset_a = []
        
        
        Vtotal=[]
        i=0
        j=0
        attempts=0
        pl.figure()
        
        if  not os.path.exists("../Analysis_Results/%d/Good_peaks/" %self.par['shot']):
                os.makedirs("../Analysis_Results/%d/Good_peaks/" %self.par['shot'])
        while j < 12 and attempts < 100 and i < len(tp):
            #tssa.append(fu.get_window_slice(tp[i]-n_samp/2*dt, td, tp[i]+n_samp/2*dt))
            sl=fu.get_window_slice(tp[i]-n_samp/2*dt, td, tp[i]+n_samp/2*dt)
        
        # retreive sample signal examples
#        for i, t1 in enumerate(tsmin):
#            t2 = tsmax[i]
#            tssa.append(fu.get_window_slice(t1,td,t2))
        #if len(tssa) == 1:
        #    sys.exit(0)
        #print 'here'
        #return
        
        
        
        
        
        #Vss_a = []
        #for i,sl in enumerate(tssa):
            tss_size = sl.stop - sl.start
            # normalize to max
            Vt = Vps[sl]/(Vps[sl].max())
            # find max position
            im = np.where(Vt == Vt.max())[0][0]
            # start in slice
            istart_s = max(0,im - n_below)
            # end in slice
            iend_s = min(tss_size, im + n_above)
            # length of slice used
            ilen_s = iend_s - istart_s
            # start position in sampling array
            istart_p = min(n_below, im)
            Vsamp = np.zeros(shape = (n_samp))
            Vsamp[n_max - istart_p:n_max - istart_p + ilen_s] = Vt[istart_s:iend_s]
        #    Vss_a.append(Vsamp)
            Vss = np.array(Vsamp)# Vss_na = np  ..Vss_a)    
        # corrected for the start time of the data
            ts = td[0:n_samp]-dtmin
        
        #--------------------------------
        # fit peak shape
        #--------------------------------                        
        
#        # fit function
#        alpha = B.Parameter(1./self.par['decay_time'], 'alpha')
#        beta = B.Parameter(1./self.par['rise_time'], 'beta')
#        x0 = B.Parameter(self.par['position'], 'x0') 
#        H = B.Parameter(1.,'H')
#        offset = B.Parameter(0., 'offset') #self.offset        
#        
#        # shift peak shape 
#        def signal(x):
#            sig, y = fu.peak(x-x0(), alpha(), beta() )
#            return y*H() + offset()
#        
#        # fit all peaks with a constant back ground
#        H_a = []
#        alpha_a = []
#        beta_a = []
#        x0_a = []
#        offset_a = []

        #for Vss in Vss_na:
            nz = Vss != 0
            alpha.set(1./self.par['decay_time'])
            beta.set(1./self.par['rise_time'])
            x0.set(self.par['position'])
            H.set(1.)
            offset.set(0.)
            F=B.genfit(signal, [H, alpha, beta, x0, offset], x = ts[nz], y = Vss[nz])
            
            if F.chi2 > self.chi2: 
                   
                i+=1
                print "bad peak"
                attempts += 1
                continue 
            # save all the fit parameters
            H_a.append(H())
            alpha_a.append(alpha())
            beta_a.append(beta())
            x0_a.append(x0())
            offset_a.append(offset())
        # all samples have been fitted
        
        #renormalize with fitted parameters
        #for i,Vss in enumerate(Vss_na):
            nz = Vss != 0.
            #Vss_na[i][nz] = (Vss[nz]-offset_a[i])/H_a[i]
            Vss[nz] = (Vss[nz]-offset_a[j])/H_a[j]
            if j==0: Vtotal=np.zeros_like(Vss)
            Vtotal = Vtotal + Vss
        # plot the peaks for checking
            pl.figure(1)
            pl.plot(td[sl],Vps[sl], '.', label='Peak %d' %j)
            pl.legend()
            pl.savefig("../Analysis_Results/%d/Good_peaks/peak_%d.png" %(self.par['shot'],j))
            pl.close()
            if True: #plot_peak_samples:
                    pl.figure(2)
                    pl.plot(ts, Vss, '.')
                    pl.axis('tight')
            j+=1
            i+=1
        # calculate the average of all renormalized peaks
        Vtotal=Vtotal/float(j)
        #Vtotal = np.sum(Vss_na, axis = 0)/Vss_na.shape[0]
        self.Vtotal = Vtotal
        # now calculate average signal
        # fit peak shape
        # reset parameters
        # fit function
        alpha.set(1./self.par['decay_time'])
        beta.set(1./self.par['rise_time'])
        x0.set(self.par['position'])
        H.set(1.)
        offset.set(0.)
        F=B.genfit(signal, [alpha, beta, x0], x=ts, y = Vtotal)
        
        


        # check the fit
        if True: 
            pl.plot(ts, Vtotal, color='g', label='average')
            pl.plot(ts,F.func(ts), color='b', label='fit_line')
            
        pl.legend()
        
        alpha_f = alpha()
        beta_f = beta()
        x0_f = x0()
        # get width of peak:
        sig = fu.peak(F.xpl-x0(), alpha(), beta() )[0]
        self.par['decay_time'] = 1/alpha_f
        self.par['rise_time'] = 1/beta_f
        self.par['position'] = x0_f
        self.par['sig'] = sig
        db.writetodb('decay_time = '+ str(self.par['decay_time']/us) + ', rise_time = ' + str(self.par['rise_time']/us) + ', position = ' + str(self.par['position']/us) , 'Peak_Sampling', 'Shot ='+str(self.par['shot'])+' AND Channel='+str(self.par['channel']))
        db.writetodb('sig = ' + str(self.par['sig']/us), 'Raw_Fitting', 'Shot ='+str(self.par['shot']) + ' AND Channel='+str(self.par['channel'])) 