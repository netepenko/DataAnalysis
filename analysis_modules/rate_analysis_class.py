# -*- coding: utf-8 -*-
# analyze fitted pulse heights

"""
Created on Thu Oct 13 15:00:11 2016

Modified Jul 7 2022


@author: Alex, WB
"""

import LT.box as B
import numpy as np
import os

from . import database_operations as db
from . import utilities as UT

us=1.e6 #conversion to microseconds constant


# helper functions for time slicing
def cut(x):
    return slice(*x)

def slice_array(x, dx):
    i_t = (x/dx).astype('int')
    i_slice, slice_start = np.unique(i_t, return_index = True)
    slice_stop = np.roll(slice_start, -1)
    slice_stop[-1] = slice_stop[-2]
    sl = np.array([slice_start, slice_stop]).T
    x_slice = i_slice*dx + 0.5*dx
    return x_slice, np.apply_along_axis(cut, 1, sl)


def get_dbdata(dbfile, q_what, q_table, q_where ):
    result = []
    try:
        result = db.retrieve(dbfile, q_what, q_table, q_where)
    except Exception as err:
        print(f'Problem getting {q_what} from table {q_table} where {q_where}: {err}')
    if result == []:
        print('No results getting  {q_what} from table {q_table} where {q_where}')

class analysis_data:

    def __init__(self, rp):
        self.rp = rp        
        t_offset = rp.par['t_offset'] #cdata.par.get_value('t_offset')*us
        # get directories
        Afit_name = rp.var['Afit_name']

        # load data (faster loading then from txt)
        d = np.load(Afit_name)
        tr = d['t'] + t_offset
        # all data
        Vpr = d['V']      # raw PH
        Ar = d['A']       # fitted PH
        dAr = d['sig_A']  # uncertainty in fit
        
        self.Vpr = Vpr
        self.Ar = Ar
        self.dAr = dAr

        # positive signals
        pa = Ar>0.
        self.pa = pa
        r = np.abs(dAr[pa]/Ar[pa])
        # cut on error ratio
        self.r_cut_off = rp.par['sig_ratio']
        # this is for good events
        gr = r < self.r_cut_off
        self.gr = gr
        # this is to study background
        #gr = r > r_cut_off

        # selected good data
        self.tg = tr[pa][gr]
        self.Ag = Ar[pa][gr]


        # simple pulse heights for testing
        self.tp = tr[pa]
        self.Ap = Vpr[pa]   # raw data

        
class rate_analysis:

    def __init__(self, dbfile, shot, channel, Afit_file = None, Version = None):
        #frequently used variable to retrieve data from database, indicates shot and chennel
        if Version is None:
            wheredb =  f'Shot = {shot} AND Channel = {channel}'
        else:
            wheredb =  f'Shot = {shot} AND Channel = {channel} AND Version = {Version}'

        self.dbfile = dbfile
        self.par={}
        self.var={}

        self.par['shot']=shot
        self.par['channel']=channel

        (time_slice_width,) = db.retrieve(dbfile, 'time_slice_width','Rate_Plotting', wheredb)[0]                
        self.par['time_slice_width']=time_slice_width


        (h_min, h_max, h_bins) = db.retrieve(dbfile,'h_min, h_max, h_bins', 'Rate_Plotting', wheredb)[0]
        self.par['h_min']= h_min
        self.par['h_max']=h_max
        h_bins=int(h_bins)
        self.par['h_bins']=h_bins


        (draw_p, draw_t, draw_sum) = db.retrieve(dbfile,'draw_p, draw_t, draw_sum', 'Rate_Plotting', wheredb)[0]
        self.par['draw_p']=UT.Bool(draw_p)
        self.par['draw_t']=UT.Bool(draw_t)
        self.par['draw_sum']=UT.Bool(draw_sum)


        (p_min, p_max, t_min, t_max, pul_min, pul_max) = db.retrieve(dbfile,'p_min, p_max, t_min, t_max, \
        pul_min, pul_max', 'Rate_Plotting', wheredb)[0]
        self.par['p_min']=p_min
        self.par['p_max']=p_max
        self.par['t_min']=t_min
        self.par['t_max']=t_max
        self.par['pulser_min'] = pul_min
        self.par['pulser_max'] = pul_max


        (A_init, sig_init, sig_ratio)=db.retrieve(dbfile,'A_init, sig_init, sig_ratio', 'Rate_Plotting', wheredb)[0]
        self.par['A_init']=A_init
        self.par['sig_init']=sig_init
        self.par['sig_ratio']=sig_ratio


        (t_offset,)=db.retrieve(dbfile,'t_offset', 'Shot_List', 'Shot = '+str(shot))[0]
        self.par['t_offset']=t_offset


        (dtmin,dtmax) = np.asarray(db.retrieve(dbfile,'dtmin, dtmax', 'Raw_Fitting', wheredb)[0])*us
        self.par['dtmin'] = dtmin
        self.par['dtmax'] = dtmax


        (add_pulser,)=db.retrieve(dbfile,'add_pulser', 'Raw_Fitting', wheredb)[0]
        self.par['add_pulser'] = UT.Bool(add_pulser)

        if Afit_file is None:
            (Afit_file,) = db.retrieve(dbfile,'file_name', 'Raw_Fitting', wheredb)[0]
        self.var['Afit_name'] = Afit_file
        dir_name, file_name = os.path.split(Afit_file)
        f_name, f_ext = os.path.splitext(file_name)
        new_dir = '/'.join(dir_name.split(os.path.sep)[:-1])
        # setup output directory name
        # set output file name
        self.var['of_name'] = f'{new_dir}/Rate_Plotting/rate_results_{shot}_{dtmin/us:5.3f}_{dtmax/us:5.3f}_{channel:d}.npz'
        self.h2 = None
        self.h2p = None


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

    def plot_results(self):

        draw_p = self.par['draw_p']
        draw_t = self.par['draw_t']
        draw_sum = self.par['draw_sum']

        #if pulser were added
        add_pulser=self.par['add_pulser']
        
        dt = self.par['time_slice_width']*us

        # proton sum signal
        if draw_p :
            B.plot_exp(self.slice_t/us, self.A_sp/dt*us, self.dA_sp/dt*us, linestyle = '-', marker = 'o', color = 'b', ecolor='grey', capsize = 0.) #,markeredgecolor='g',

        # triton sum signal
        if draw_t:
            B.plot_exp(self.slice_t/us, self.A_st/dt*us, self.dA_st/dt*us, color = 'g', ecolor='grey', capsize = 0.)

        # pulser sum signal
        if add_pulser:
            B.plot_exp(self.slice_t/us, self.A_pul/dt*us, self.dA_pul/dt*us, color = 'm', ecolor='grey', capsize = 0.)

        # Total signal
        if draw_sum :
            B.plot_exp(self.slice_t/us, self.A_t/dt*us, self.dA_t/dt*us, linestyle = '-',ecolor='grey', marker = '.',   capsize = 0., label='Ch %d'%self.par['channel'])
        o_file = self.var['of_name']

        if  not os.path.exists(os.path.dirname(o_file)):
                os.makedirs(os.path.dirname(o_file))

        B.pl.xlabel('t [s]')
        B.pl.ylabel('Rate [Hz]')
        B.pl.title('Shot : '+ str(self.par['shot']) +'/ channel: '+str(self.par['channel']))
        B.pl.xlim((self.par['dtmin']/us, self.par['dtmax']/us))
        B.pl.show()
        """
        B.pl.savefig('../Analysis_Results/%d/Rate_Plotting/Rate_%s.png' %(self.par['shot'],self.var['Afit_name'][-16:-6]))
        """


    def make_2d_histo(self, tmin = None, tmax = None):
        # get data
        a_data = analysis_data(self)
        # time slice the data
        dt = self.par['time_slice_width']*us # step width
        # good data
        tg = a_data.tg
        Ag = a_data.Ag
        
        # all pos. data
        tp = a_data.tp
        Ap = a_data.Ap

        # 2d histogram setup
        # y - aaxis
        hy_min = self.par['h_min']
        hy_max = self.par['h_max']
        hy_bins = self.par['h_bins']        
        # x - axis
        if tmin is None:
            tmin = tp.min()
        if tmax is None:
            tmax = tp.max()
        hx_bins = int((tmax - tmin)/dt) + 1
        
        h_title = f'shot {self.par["shot"]}, channel = {self.par["channel"]}' 
        
        self.h2 = B.histo2d(tg, Ag, range = [[tmin,tmax],[hy_min,hy_max]], bins = [hx_bins, hy_bins],
                            title = h_title, xlabel = r't[$\mu$ s]', ylabel = 'fitted PH [V]')
        self.h2p = B.histo2d(tp, Ap, range = [[tmin,tmax],[hy_min,hy_max]], bins = [hx_bins, hy_bins],
                             title = h_title, xlabel = r't[$\mu$ s]', ylabel = 'raw PH [V]')

    def plot_2d(self, raw = False, **kwargs):
        if  self.h2 is None:
            print('No 2d histogram created !')
            return
        
        if raw:
            self.h2p.plot(**kwargs)
        else:
            self.h2.plot(**kwargs)
            
    
    def delete_2d_histo(self):
        self.h2 = None
        self.h2p = None
        
        
    def time_slice_data(self):
        
        # get data
        a_data = analysis_data(self)

        # good data
        tg = a_data.tg
        Ag = a_data.Ag
        
        # all pos. data
        tp = a_data.tp
        Ap = a_data.Ap
        

        # histogram setup
        h_min = self.par['h_min']
        h_max = self.par['h_max']
        h_bins = self.par['h_bins']
        
        h_range = (h_min, h_max)

        #if pulser were added
        add_pulser=self.par['add_pulser']
        # no histo fitting at this time
        fit_histos = False
        


        # time slice the data
        dt = self.par['time_slice_width']*us # step width

        # good slices
        slice_t, slices = slice_array(tg, dt)
        # all slices
        slice_t_a, slices_a = slice_array(tp, dt)
        
        self.slice_t = slice_t
        self.slices = slices
        self.slice_t_a = slice_t_a
        self.slices_a = slices_a
        
        # histogram
        h = []
        hp = []
        h_sum =  B.histo( np.array([]), range = h_range, bins = h_bins)
        hp_sum =  B.histo( np.array([]), range = h_range, bins = h_bins)
        for i,s in enumerate(slices):
            hi = B.histo( Ag[s], range = h_range, bins = h_bins)
            h_time = "{0:6.4f} s".format( slice_t[i]/us )
            hi.title = h_time
            h.append( hi  )
            h_sum += hi

        for i,s in enumerate(slices_a):
            hip = B.histo( Ap[s], range = h_range, bins = h_bins)
            h_time = "{0:6.4f} s".format( slice_t_a[i]/us )
            hip.title = h_time
            hp.append( hip  )
            hp_sum += hip
            
        # store created histograms
        self.h=h
        self.hp=hp
        self.h_tot = h_sum
        self.hp_tot = hp_sum

        print("created ", len(h), " histograms h")

        # calculate rates for protons and tritons
        # proton counts
        A_sp = []
        dA_sp= []
        # triton counts
        A_st = []
        dA_st= []
        # pulser counts
        A_pul = []
        dA_pul = []

        # for protons
        p_min =  self.par['p_min']#cdata.par.get_value('p_min')
        p_max =  self.par['p_max']#cdata.par.get_value('p_max')
        # for tritons
        t_min = self.par['t_min']
        t_max = self.par['t_max']

        # for pulser if selected
        if add_pulser:
            pul_min = self.par['pulser_min']
            pul_max = self.par['pulser_max']

        for i, hi in enumerate(h):
            # proton fit
            # fitting with limits
            if fit_histos:
                pass
                # hi.fit(xmin = p_min, xmax = p_max)
            # sum histograms
            # intgerate proton peak
            sp, dsp = hi.sum(xmin = p_min , xmax = p_max)  
            # integrate triton peaj
            st, dst = hi.sum(xmin = t_min , xmax = t_max)
            # store results
            A_sp.append(sp)
            dA_sp.append(dsp)
            A_st.append(st)
            dA_st.append(dst)
            # integrate pulser
            if add_pulser:
                spul, dspul = hi.sum(xmin = pul_min , xmax = pul_max)
                A_pul.append(spul)
                dA_pul.append(dspul)            

        # proton amplitudes
        self.A_sp = np.array(A_sp)
        self.dA_sp = np.array(dA_sp)
        # tritomn amplitudes
        self.A_st = np.array(A_st)
        self.dA_st = np.array(dA_st)
        # pulser amplitudes
        if add_pulser:
            self.A_pul = np.array(A_pul)
            self.dA_pul = np.array(dA_pul)

        # total
        self.A_t = self.A_sp + self.A_st
        self.dA_t = np.sqrt(self.dA_sp**2 + self.dA_st**2)

        # fitting results
        if fit_histos:
            pass


        # write results
        """

#        if os.path.isfile(o_file):
#            inp = raw_input("Do you want to overwrite the results file? (y)es or (n)o: ")
#            if inp == "yes" or inp == "y":
#               os.remove(o_file)
#               print 'Old file removed.'
#            elif inp == "no" or inp == "n":
#               return

        np.savez_compressed(o_file, t=np.asarray(slice_t)/us, Ap=np.asarray(A_sp)/dt*us, dAp=np.asarray(dA_sp)/dt*us, At=np.asarray(A_st)/dt*us,
                                dAt=np.asarray(dA_st)/dt*us, A=np.asarray(A_t)/dt*us, dA=np.asarray(dA_t)/dt*us)
        """
