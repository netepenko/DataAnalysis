# -*- coding: utf-8 -*-
# analyze fitted pulse heights

"""
Created on Thu Oct 13 15:00:11 2016

@author: Alex
"""
import database_operations as db
import LT.box as B
import numpy as np
import os
us=1.e6 #conversion to microseconds constant

class rates_plotting():
    
    def __init__(self, shot, channel, ifile):
        #frequently used variable to retrieve data from database, indicates shot and chennel         
        wheredb =  'Shot = '+str(shot)+' AND Channel = '+str(channel)
        
        self.par={}
        self.var={}

        self.par['shot']=shot
        self.par['channel']=channel        
        

        (time_slice_width,) = db.retrieve('time_slice_width','Rates_Plotting', wheredb)        
        self.par['time_slice_width']=time_slice_width


        (h_min, h_max, h_bins) = db.retrieve('h_min, h_max, h_bins', 'Rates_Plotting', wheredb)
        self.par['h_min']= h_min
        self.par['h_max']=h_max
        h_bins=int(h_bins)
        self.par['h_bins']=h_bins


        (draw_p, draw_t, draw_sum) = db.retrieve('draw_p, draw_t, draw_sum', 'Rates_Plotting', wheredb)
        self.par['draw_p']=draw_p
        self.par['draw_t']=draw_t        
        self.par['draw_sum']=draw_sum
        

        (p_min, p_max, t_min, t_max, pul_min, pul_max) = db.retrieve('p_min, p_max, t_min, t_max, \
        pul_min, pul_max', 'Rates_Plotting', wheredb)
        self.par['p_min']=p_min
        self.par['p_max']=p_max
        self.par['t_min']=t_min
        self.par['t_max']=t_max
        self.par['pulser_min'] = pul_min
        self.par['pulser_max'] = pul_max
        

        (A_init, sig_init, sig_ratio)=db.retrieve('A_init, sig_init, sig_ratio', 'Rates_Plotting', wheredb)
        self.par['A_init']=A_init
        self.par['sig_init']=sig_init
        self.par['sig_ratio']=sig_ratio        
                

        (t_offset,)=db.retrieve('t_offset', 'Shot_List', 'Shot = '+str(shot))        
        self.par['t_offset']=t_offset
                

        (dtmin,dtmax) = np.asarray(db.retrieve('dtmin, dtmax', 'Raw_Fitting', wheredb))*us
        self.par['dtmin'] = dtmin
        self.par['dtmax'] = dtmax
        

        (add_pulser,)=db.retrieve('add_pulser', 'Raw_Fitting', wheredb)        
        self.par['add_pulser'] = add_pulser
        
        #------------assign class variables -------------
        self.var['f_name'] = ifile
        #'../Analysis_Results/'  + str(shot) + '/Raw_Fitting/' + "fit_results_4_"+ str(shot) + "_{0:5.3f}_{1:5.3f}_{2:d}.npz".format(dtmin/us, dtmax/us, channel)
        self.var['of_name'] = '../Analysis_Results/'  + str(shot) + '/Rate_Plotting/' + "rate_results_4_"+ str(shot) + "_{0:5.3f}_{1:5.3f}_{2:d}.npz".format(dtmin/us, dtmax/us, channel)  
        
        
    #plot histograms for fitted pulse height and simple pulse height    
    def draw_histos(self, i=0):
        B.pl.clf()
        
        if False:# draw_raw:
            pass
            #hp[i].plot()
            #h[i].plot(color = 'r')
        else:
            self.h[i].plot()
        B.pl.show()

    def cumulative_hist(self):
        ht=self.h[0]
        for h in self.h[1:]:
            ht = ht + h
        B.pl.figure()
        ht.plot()
        return

    def plot_results(self):
        
        draw_p = self.par['draw_p']
        draw_t = self.par['draw_t']   
        draw_sum = self.par['draw_sum']
    
    
        # histogram setup
        h_min = self.par['h_min'] 
        h_max = self.par['h_max']
        h_bins = self.par['h_bins']
        h_range = (h_min, h_max)
    
        #if pulser were added
        add_pulser=self.par['add_pulser']    
        # no histo fitting at this time
        fit_histos = False 
        t_offset = self.par['t_offset'] #cdata.par.get_value('t_offset')*us
        # get directories
        f_name = self.var['f_name'] 
    
        # load data (faster loading then from txt)
        d = np.load(f_name)
        tr = d['t'] + t_offset
        Vpr = d['V']      # raw PH
        Ar = d['A']       # fitted PH
        dAr = d['sig_A']
#        chir = np.zeros_like(tr)
    
        # positive signals
        pa = Ar>0.
        r = np.abs(dAr[pa]/Ar[pa])
        # cut on error ratio
        r_cut_off = self.par['sig_ratio']
        # this is for good events
        gr = r < r_cut_off
        
        # this is to study background
        #gr = r > r_cut_off
    
        tg = tr[pa][gr]
        A = Ar[pa][gr]
#        dA = dAr[pa][gr]
#        chi = chir[pa][gr]
        
        # simple pulse heights for testing
        ta = tr[pa]
        Ap = Vpr[pa]   # raw data
    
        # time slice the data
        dt = self.par['time_slice_width']*us # step width
        
        # bin numbers
        i_t = (tg/dt).astype('int')
        i_t_a = (ta/dt).astype('int')
        # time slicing for fitted peaks
        slice_t = []
        slices = []
        i_start = 0
        i_bin = i_t[0]
        for i, ip in enumerate(i_t):
            if ip != i_bin or i==len(i_t)-1 :
                i_end = i
                # print "found slice from : ", i_start, i_end
                slice_t.append(i_bin*dt + 0.5*dt)
                slices.append( slice(i_start, i_end) )
                i_bin = ip
                i_start = i
    
        # time slicing for all peaks
        slice_t_a = []
        slices_a = []
        i_start_a = 0
        i_bin_a = 0
        for i, ip in enumerate(i_t_a):
            if ip == i_bin_a:
                continue
            else:
                i_end = i
                # print "found slice from : ", i_start, i_end
                slice_t_a.append(i_bin_a*dt + 0.5*dt)
                slices_a.append( slice(i_start_a, i_end) )
                i_bin_a = ip
                i_start_a = i
    
    
        # histogram
        h = []
        hp = []
        
        for i,s in enumerate(slices):
            hi = B.histo( A[s], range = h_range, bins = h_bins)
            h_time = "{0:6.4f} s".format( slice_t[i]/us )
            hi.title = h_time
            h.append( hi  )
        
        for i,s in enumerate(slices_a):
            hip = B.histo( Ap[s], range = h_range, bins = h_bins)
            h_time = "{0:6.4f} s".format( slice_t_a[i]/us )
            hip.title = h_time
            hp.append( hip  )
        
        self.h=h
        self.hp=hp
            
        print "created ", len(h), " histograms h"
        
        A_sp = []
        dA_sp= []
        
        A_st = []
        dA_st= []
        
        A_pul = []
        dA_pul = []
    
        # for protons
        p_min =  self.par['p_min']#cdata.par.get_value('p_min')
        p_max =  self.par['p_max']#cdata.par.get_value('p_max')
        # for tritons
        t_min = self.par['t_min']
        t_max = self.par['t_max']
        # fit histograms
        
        if add_pulser:
            pul_min = self.par['pulser_min']
            pul_max = self.par['pulser_max']
    
        # inital parameters
        A_init = self.par['A_init']
        sigma_init = self.par['sig_init']
        
        for i, hi in enumerate(h):
            # proton fit
            # hi.mean.set(p_mean) #check if necessary
            hi.A.set(A_init)
            hi.sigma.set(sigma_init)
            # fitting with limits
            if fit_histos:
                hi.fit(xmin = p_min, xmax = p_max)
            # sum histograms
            sp, dsp = hi.sum(xmin = p_min , xmax = p_max) 
            st, dst = hi.sum(xmin = t_min , xmax = t_max) 
            spul, dspul = hi.sum(xmin = pul_min , xmax = pul_max) 
            A_sp.append(sp)
            dA_sp.append(dsp)
            A_st.append(st)
            dA_st.append(dst)
            A_pul.append(spul)
            dA_pul.append(dspul)
        # proton amplitudes
        A_sp = np.array(A_sp)
        dA_sp = np.array(dA_sp)
        # tritomn amplitudes
        A_st = np.array(A_st)
        dA_st = np.array(dA_st)
        # pulser amplitudes
        A_pul = np.array(A_pul)
        dA_pul = np.array(dA_pul)
        
        # total 
        A_t = A_sp + A_st
        dA_t = np.sqrt(dA_sp**2 + dA_st**2)
        
        # fitting results
        if fit_histos:
            A_f = []
            dA_f = []
            chi_red = []
            m_f = []
            dm_f = []
            fact = np.sqrt(2.*np.pi)
            for i, hi in enumerate(h):
                bw = hi.bin_width
                sig = np.abs(hi.sigma())
                name, Am, dAm  = hi.fit_par['A'].get()
                name, p, dp = hi.fit_par['mean'].get()
                A_f.append(Am*sig*fact/bw)
                dA_f.append(dAm*sig*fact/bw)
                m_f.append(p)
                dm_f.append(dp)
        	chi_red.append(hi.F.chi2_red)    
            A_f = np.array(A_f)
            dA_f = np.array(dA_f)
            m_f = np.array(m_f)
            dm_f = np.array(dm_f)
            chi_red = np.array(chi_red)
        
        # proton sum signal 
        if draw_p == 'True':  
            B.plot_exp(np.array(slice_t)/us, A_sp/dt*us, dA_sp/dt*us, linestyle = '-', marker = 'o', color = 'b', ecolor='grey', capsize = 0.) #,markeredgecolor='g',
        
        # triton sum signal
        if draw_t == 'True':
            B.plot_exp(np.array(slice_t)/us, A_st/dt*us, dA_st/dt*us, color = 'g', ecolor='grey', capsize = 0.)
        
        # pulser sum signal
        if add_pulser == 'True':
            B.plot_exp(np.array(slice_t)/us, A_pul/dt*us, dA_pul/dt*us, color = 'm', ecolor='grey', capsize = 0.)
        
        # Total signal
        if draw_sum == 'True':
            B.plot_exp(np.array(slice_t)/us, A_t/dt*us, dA_t/dt*us, linestyle = '-',ecolor='grey', marker = '.',   capsize = 0., label='Ch %d'%self.par['channel'])#color = 'r',ecolor='grey', #
        
        o_file = self.var['of_name']
        
        if  not os.path.exists(os.path.dirname(o_file)):
                os.makedirs(os.path.dirname(o_file))
        
        B.pl.xlabel('t [s]')
        B.pl.ylabel('Rate [Hz]')
        B.pl.title('Shot : '+ str(self.par['shot']) +'/ channel: '+str(self.par['channel']))
        B.pl.xlim((self.par['dtmin']/us, self.par['dtmax']/us))
        B.pl.show()
        B.pl.savefig('../Analysis_Results/%d/Rate_Plotting/Rate_%s.png' %(self.par['shot'],self.var['f_name'][-16:-6]))
        # write results
        
        
#        if os.path.isfile(o_file):
#            inp = raw_input("Do you want to overwrite the results file? (y)es or (n)o: ") 
#            if inp == "yes" or inp == "y": 
#               os.remove(o_file)
#               print 'Old file removed.'
#            elif inp == "no" or inp == "n": 
#               return
        
        np.savez_compressed(o_file, t=np.asarray(slice_t)/us, Ap=np.asarray(A_sp)/dt*us, dAp=np.asarray(dA_sp)/dt*us, At=np.asarray(A_st)/dt*us,
                                dAt=np.asarray(dA_st)/dt*us, A=np.asarray(A_t)/dt*us, dA=np.asarray(dA_t)/dt*us)

