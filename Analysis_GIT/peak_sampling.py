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

        

def load_peaks(self):
    sz = 5
    #        print "-----------Loading selected peaks data-------------------"
    tsmin = np.asarray(db.retrieve('b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12', \
    'Peak_Sampling', 'Shot = ' + str(self.par['shot']) +' AND ' + 'Channel = ' +  \
    str(self.par['channel'])))*us
    tsmax = np.asarray(db.retrieve('e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12', \
    'Peak_Sampling', 'Shot = ' + str(self.par['shot']) +' AND ' + 'Channel = ' +  \
    str(self.par['channel'])))*us
    
    Vtotal=np.zeros(int(sz*(self.par['rise_time'] + self.par['decay_time'])/self.dt))
    
    for i in range(len(tsmin)):
        tmin = tsmin[i]
        tmax =tsmax[i]
        Vps=self.Vps
        td=self.td
        sl=fu.get_window_slice(tmin, td, tmax)
        Vt = Vps[sl]/(Vps.max())
        ts = td[0:Vt.shape[0]]-td[0]
        F, alpha, beta, H, offset, x_0 = self.fit_shape(ts,Vt)
        
        pl.figure()
        pl.plot(ts, Vt, '.', label='Peak %d Normalized' %i)
        pl.plot(ts,F.func(ts), color='b', label='fit_line')
        pl.legend()
        pl.axis('tight')
        
        Vt = (Vt-offset)/H
         
        for i, V in enumerate(Vtotal):
            try:
                Vtotal[i] = Vtotal[i] + Vt[i+ Vt.argmax()- int(sz*(self.par['rise_time']/self.dt))]
            except:
                pass
       
    ttotal = td[0:Vtotal.shape[0]]-td[0]
    Vtotal =  Vtotal / len(tsmin)
    
    F, alpha, beta, H, offset, x0 = self.fit_shape(ttotal,Vtotal)
    
    pl.figure()
    pl.plot(ttotal, Vtotal, '.', label='Average of peaks')
    pl.plot(ttotal,F.func(ttotal), color='b', label='fit_line')
    pl.legend()
    pl.axis('tight')      
    sig = fu.peak(F.xpl-x0, alpha, beta)[0]
    self.par['decay_time'] = 1/alpha
    self.par['rise_time'] = 1/beta
    self.par['sig'] = sig
    db.writetodb('decay_time = '+ str(self.par['decay_time']/us) + ', rise_time = ' + str(self.par['rise_time']/us), 'Peak_Sampling', 'Shot ='+str(self.par['shot'])+' AND Channel='+str(self.par['channel']))
    db.writetodb('sig = ' + str(self.par['sig']/us), 'Raw_Fitting', 'Shot ='+str(self.par['shot']) + ' AND Channel='+str(self.par['channel']))     
        

        
def fit_shape(self, ts, Vt):
     # fit function
    # initialize fit function parameters
    alpha = B.Parameter(1./self.par['decay_time'], 'alpha')
    beta = B.Parameter(1./self.par['rise_time'], 'beta')
    x0 = B.Parameter(ts[np.argmax(Vt)], 'x0') 
    #x0 = B.Parameter(self.par['position'], 'x0') 
    H = B.Parameter(1.,'H')
    offset = B.Parameter(0., 'offset') #self.offset        
    
    # shift peak shape 
    def signal(x):
        sig, y = fu.peak(x-x0(), alpha(), beta() )
        return y*H() + offset()
    
    F=B.genfit(signal, [alpha, beta, x0, H, offset], x = ts, y = Vt)
           
    return F, alpha(), beta(), H(), offset(), x0()
                         

#fit sample peaks and write shape parameters in db
def find_good_peaks(self, save):
        tmin = self.par['dtmin']
        tmax = self.par['dtmax']
        sl = fu.get_window_slice(tmin, self.td, tmax)
        Vps=self.Vps[sl]
        td=self.td[sl]
        
        #time step of data        
        

        # find peaks first and then select those above threshold for good peak
        Vgp=self.Vgp #put in DB later
        Vstep=self.par['Vstep']
        results = np.zeros((2,), dtype = 'int32')
        pmin = np.zeros((len(Vps)/5, ), dtype='int32')
        pmax = np.zeros((len(Vps)/5, ), dtype='int32')
        FP.find_peaks(len(Vps), Vstep, Vps, results, pmin, pmax) 
        
        # number of maxima
        nmax = results[1]
        print " found : ", nmax, " maxima"
        imax = pmax[:nmax]
        # number of minima
        nmin = results[0]
        print " found : ", nmin, " minima"
        #imin = pmin[:nmin]
        
        # get the indices of peaks higher then threshold, last peak removed(idk why...)
        ipeak_tr = np.delete(np.where(Vps[imax]>Vgp)[0],-1)
        
        tp = td[imax][ipeak_tr]
        Vp = Vps[imax][ipeak_tr]


        #go through found peaks and determine the shape of peak
        #by requiring  fit error to be less then some value
        
        
        
        i=1
        j=0
        attempts=0
        #pl.figure()
        
        if  not os.path.exists("../Analysis_Results/%d/Good_peaks/" %self.par['shot']):
                os.makedirs("../Analysis_Results/%d/Good_peaks/" %self.par['shot'])
        sz=5 #mutliplier at slice size
        while j < 12 and attempts < 1000 and i < len(tp):
            #tssa.append(fu.get_window_slice(tp[i]-n_samp/2*dt, td, tp[i]+n_samp/2*dt))
            sl=fu.get_window_slice(tp[i]-sz*self.par['rise_time'], td, tp[i]+sz*self.par['decay_time'])
        
            Vt = Vps[sl]/(Vp[i])
            
            ts = td[0:Vt.shape[0]]-td[0]
            
            F, alpha, beta, H, offset, x0 = self.fit_shape(ts,Vt)
            
            if F.chi2 > self.chi2: 
                   
                i+=1
                print "bad peak"
                attempts += 1
                continue 
        
            if save:
                x = (tp[i]-sz*self.par['rise_time'])/us
                y = (tp[i]+sz*self.par['decay_time'])/us      
                pn = j + 1
                db.writetodb('b'+str(pn)+'='+str(x)+', e'+str(pn)+'='+str(y), 'Peak_Sampling',
                     'Shot = '+str(self.par['shot'])+' AND Channel = '+str(self.par['channel']))        
            if True: #plot_peak_samples:
                    pl.figure()
                    pl.plot(ts, Vt, '.', label='Peak %d Normalized' %j)
                    pl.plot(ts,F.func(ts), color='b', label='fit_line')
                    pl.legend()
                    pl.axis('tight')
                    
            #renormalize with fitted parameters
            Vt = (Vt-offset)/H
            
            if j==0: Vtotal=np.zeros_like(Vt)
            Vtotal = Vtotal + Vt
       
       
            
            j+=1
            i+=1
        
        # calculate the average of all renormalized peaks
        Vtotal=Vtotal/float(j)
        self.Vtotal = Vtotal
        # now calculate average signal
        F, alpha, beta, H, offset, x0 = self.fit_shape(ts,Vtotal)
        
        # check the fit
        if True: 
            pl.figure()
            pl.plot(ts, Vtotal,'.', color='g', label='Peaks average')
            pl.plot(ts,F.func(ts), color='b', label='fit_line')
            pl.legend()
        
        # get width of peak:
        sig = fu.peak(F.xpl-x0, alpha, beta)[0]
        self.par['decay_time'] = 1/alpha
        self.par['rise_time'] = 1/beta
        self.par['sig'] = sig
        
        if save:
            db.writetodb('decay_time = '+ str(self.par['decay_time']/us) + ', rise_time = ' + str(self.par['rise_time']/us) + ', Vgp =' + str(self.Vgp),
                                              'Peak_Sampling', 'Shot ='+str(self.par['shot'])+' AND Channel='+str(self.par['channel']))
            db.writetodb('sig = ' + str(self.par['sig']/us), 'Raw_Fitting', 'Shot ='+str(self.par['shot']) + ' AND Channel='+str(self.par['channel'])) 