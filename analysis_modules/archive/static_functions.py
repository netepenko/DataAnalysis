"""

Generic peak shape for a pulse located at tp:
    
    y = A/y0 e^(-a (t + t0 - tp) ) (1 + tanh(b(t + t0 - tp)) )
    
    here A, a, and b are fit parameters
    
    x0, y0 are the location and the value ot the maximum
    
    t + t0 makes sure that the peak is located at 0 

The functions below are used to determine the standard pulse shape for a data file


"""

import numpy as np






    


def get_fit_groups_new_ori(num_peaks, imax, t_off, t):
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
    i_group = ((imax - i_offset)/i_window).astype(int)
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
    return fg, fit_window, i_group
        
def get_fit_groups_new(num_peaks, imax, t_off, t):
    # setup peak fitting groups
    # t     :   time of each digitizer point
    # t_off :   time offset, this is used to create shifted fitting groups
    # num_peaks    :  number of peaks to be fitted (on average) in one group
    # imax         : indices of peak positions into the t array
    #
    # returns: fg: fit group array containins start and end index into imax array for each fit group
    #          fit_window: average width of fit range in t

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
    fg = []
    fg_start = []
    fg_stop = []
    # offset in indices
    i_offset = int(t_off/t_res)
    # window size in indices
    i_window = int(fit_window/t_res)
    i_start = np.arange(i_offset, t.size+i_window, i_window).clip(max = (t.size - 1))
    i_stop = np.roll(i_start,  -1)
    # start of fit groups
    # make sure i_group is not negative
    i_group = ( (imax - i_offset)/i_window ).astype(int)
    group_nr, fg_start = np.unique(i_group, return_index = True)
    # end of fit groups
    fg_stop = np.roll(fg_start, -1)
    fg = np.transpose(np.array([fg_start[:-1], fg_stop[:-1]]))
    return fg, fit_window, i_group



if __name__ == "__main__":
    # simuklated pea
    im = np.sort(np.random.uniform(0., 1000., 100)).astype(int)
    t = np.linspace(0., 100, 1000)*1e-5
    dt = t[1]-t[0]
    num_peaks = 20
    t_off = num_peaks/2*dt
    i_off = 0.
    fgn, fwn = get_fit_groups_new(num_peaks, im, t_off, t)
    