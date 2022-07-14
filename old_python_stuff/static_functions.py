# general peak shape
import numpy as np
def peak(x, a, b):
    if 2.*b/a-1.>0:
        x_0 = np.log(2.*b/a-1.)/(2.*b)
    else:
        x_0 = 0
    y_0 = np.exp(-a*(x_0))*(1.+np.tanh(b*(x_0)))
#    good = np.where(abs(a*(x+x_0))<10.0)
#    y=np.zeros_like(x)
    y = 1./y_0*np.exp(-a*(x+x_0))*(1.+np.tanh(b*(x+x_0)))
#    # estimate width
    ismall = y<0.5
    try:
        idiff = np.diff(np.where(ismall)[0])
        i1=np.where(idiff>1)[0][0]
        i2 = i1 + idiff.max()
        sig = x[i2] - x[i1]
    except:
        sig = 0.
    return sig, y 
     
    
def get_window_slice(xmin, x, xmax):
        # get the slice corresponding to xmin and xmax in x
        # the x-values need to be equally spaced
        
        dx = x[1] - x[0]
        nmin = max( 0, int(round( (xmin - x[0])/dx )))
        nmax = min( (int(round ((xmax - x[0])/dx ))), len(x) -1  )
        return slice(nmin, nmax + 1)
        
def get_fit_groups_new(num_peaks, imax, t_off, t):
        # t time of each digitizer point
        # t_off time offset
        # np number of peaks to be fitted (on average)
        # imax indices of peak positions into the t array
    
        # total number of peaks   
        n_peaks = imax.shape[0]
        # total time interval to be fitted
        t_tot = t[-1] - t[0]
        t_res = t[1] - t[0]
        # average time between peaks
        delta_t = t_tot/n_peaks
        # fit time window
        fit_window = delta_t*num_peaks
        # group peak positions along fit windows
        fg = []
        fg_start = []
        fg_stop = []
        new_group = True
        same_group = False
        i_offset = int(t_off/t_res)
        i_window = int(fit_window/t_res)
        
        i_group = (imax - i_offset)/i_window
        init_first_group = True 
        for i, ig in enumerate(i_group):
            # skip negative indices
            if ig < 0.:
                continue
            if init_first_group:
                i_current_group = ig
                init_first_group = False
                new_group = False
                fg_start.append(i)            
            same_group =  (ig == i_current_group)
            if same_group:
                continue
            else:
                fg_stop.append(i)
                new_group = True
            pass
            if new_group:
                fg_start.append(i)
                new_group = False
                same_group = True
                i_current_group = ig
            pass
        pass
        if (len(fg_stop) == len(fg_start) -1):
            fg_stop.append(fg_start[-1])
        
        fg = np.transpose(np.array([fg_start, fg_stop]))
        return fg, fit_window