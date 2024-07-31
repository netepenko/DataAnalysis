# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:49:02 2016

@author:  WB November 14 2023
         

- New: this is its own class

- all times in us

- The normal usage would be something like:
    import peak_sampling as PS
    

    

"""
import numpy as np
import LT.box as B

#conversion to microseconds constant
us=1.e6

#%% useful functions

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
 
    
# determine the peak position by fitting a parabola to data and determine its
# extremum location

def get_peak(x,y,dy = None, plot = False):
   """
   Parameters
   ----------
   x : independent data
   y : exp.  data
   dy : error in exp. data

   Returns
   -------
   peak position, uncertainty in peak position

   """
   # fit parabola to data
   if dy is None:
       p = B.polyfit(x, y, order =2)
   else:
       p = B.polyfit(x, y, dy, order =2)
   #
   # y = a0 + a1*x + a2*x**2
   #
   a0 = p.par[0]
   a1 = p.par[1]
   a2 = p.par[2]
   # covariance matrix
   C = p.cov
   # parabola max/min
   xm = -a1/(2.*a2)
   # for errors calculate partial derivatives wrsp a1, a2
   dxm_da1 = -1./(2.*a2)
   dxm_da2 = a1/(2*a2**2)
   # calculate total error using the covariance matrix
   dxm2 = (C[1,1]*dxm_da1**2 + 2.*C[1,2]*dxm_da1*dxm_da2 + C[2,2]*dxm_da2**2)
   dxm = np.sqrt(dxm2)
   # if selected plot the fitted curve
   if (plot):
       B.plot_line(p.xpl, p.ypl)
   return (xm, dxm)



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
    
        

#%% sampling class
class peak_sampling:
    
    def __init__(self, td, Vs, chi2 = None, 
                 p_fact = 5, 
                 plot_single_peaks = True, 
                 plot_common_peak = True, 
                 rise_time = None, 
                 decay_time = None,
                 N_p = 12):
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
        p_fact : int 
            parameter to size peak shape array, optional. The default is 5. Size = p_fact*(size(rise_time + decay_time))
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
        self.td = td
        self.Vs = Vs
        self.p_fact = p_fact
        self.plot_single_peaks = plot_single_peaks
        self.plot_common_peak = plot_common_peak
        self.rise_time = rise_time
        self.decay_time = decay_time
        self.good_peak_times = []
        
        self.sl = None
        
       
    def get_peak(self, refine = False, np = 3, plot = True):
        """
        Find a peak and determine its location within the limits of the current plot

        Parameters
        ----------
        refine : Bool
            if true refine position by fitting a parabola to the maximum area +/- np data points
            default (False)
        
        np : Int
            number of neighoring points to be included (defaul = 3)

        Returns
        -------
        peak position

        """
        x1, x2 = B.pl.xlim()
        sl = get_window_slice(x1, x2, self.td)
        i_max = np.argmax(self.Vs[sl]) + sl.start
        tp = self.td[i_max]
        if refine:
            i1 = i_max - np
            i2 = i_max + np
            tp, dtp = get_peak(self.td[i1:i2], self.Vs[i1:i2], plot = plot)
        return tp
            
       
    def get_time_slices(self, t_p):
        """
        Get time slices cenetered around t_p using given p_fact, rise_time and decay_time
        
        The user only needs to provide a set of peak positions

        Parameters
        ----------
        t_p : numpy array (float)
            peak positions for good peaks

        Returns
        -------
        None.

        """
        
        dt_low=self.p_fact*self.rise_time        
        dt_high=int(self.p_fact*self.decay_time/self.dt)


        self.t_start = t_p - dt_low
        self.t_stop = t_p + dt_low 

        
        self.sl = [get_window_slice(self.t_start[x[0]], self.td, self.t_stop[x[0]] ) for x in enumerate(self.t_start)] 



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
            sig, y = peak(x-x0(), alpha(), beta() )
            return y*H() + offset()
        
        F=B.genfit(signal, [alpha, beta, x0, H, offset], x = ts, y = Vt, plot_fit = False)
               
        return F, alpha(), beta(), H(), offset(), x0()
        
    def fit_peaks(self, save_fit = True, save_slices = True):
        """
        normalize and fit peaks in within the time slices t_slice
        
        - make sure that the time ranges selected in ts are larger than 
         p_fact*(rise_time + decay_time) otherwise the
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
        if self.sl is None:
            print('No peaks to fit')
            return
        # create common array for peak data
        Vtotal=np.zeros(int(self.p_fact*(self.rise_time + self.decay_time)/self.dt) )
        counters = np.zeros_like(Vtotal)
        # local names
        Vps=self.Vps  # digitizer data
        td=self.td
        # p_fact * rise time im indices
        i_shift = int(self.p_fact*(self.rise_time/self.dt) )
        # loop over time slices for model peaks
        for i, sl in enumerate(self.sl):
            Vmax = Vps[sl].max()  # get signal maximum in this slice
            imax = Vps[sl].argmax() 
            Vt = Vps[sl]/Vmax       # normalize data to 1
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
        # calculate averaged values 
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
            sig = peak(F.xpl-x0, alpha, beta)[0]
            self.decay_time = 1/alpha
            self.rise_time = 1/beta
            self.sig = sig
        else:  
            return F, alpha, beta, H, offset, x0
        # all done               
    

            
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
        # do saving here
        pass

        
       
