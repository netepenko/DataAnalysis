# Combine fitted rates, plot them and produce rate output for orbit_fit
#
# this produces the overview plot and the rates plots as a function of views or radii if available

import LT.box as B
import numpy as np
import database_operations as db
#import sys
# parse arguments
import argparse as AG
import match_arrays as MA

from matplotlib.ticker import MaxNLocator

def window(a,c):
    i_s = (c[0]<=a)&(a<=c[1])
    return i_s

def get_mean(t, A, dA, c):
    i_s = window(t,c)
    if dA[i_s].shape[0] == 0:
        return c.mean(), 0., 0.
    nz = (dA[i_s] > 0.)
    w = dA[i_s]**2
    A_sel = A[i_s]
    t_mean = t[i_s].mean()
    if A_sel.shape[0] == 0:
        return t_mean, 0., 0.
    w_mean = np.sum(A_sel[nz]/w[nz])/np.sum(1./w[nz])
    sig_w_mean = np.sqrt(1./np.sum(1./w[nz]))
    # pdb.set_trace()
    return t_mean, w_mean, sig_w_mean 

def get_rate(s):
    # time of slice
    t0 = s[:,0]
    t1 = s[:,1]
    ch = s[:,2]
    Rb = s[:,3]
    dRb = s[:,4]
    Ra = s[:,5]
    dRa = s[:,6]
    nrb = s[:,7]
    nra = s[:,8]
    r_b = s[:,9]
    r_a = s[:,10]
    return t0, t1,  Rb, dRb, Ra, dRa, nrb, nra, ch, r_b, r_a

def closest(x, a):
    # return the index in a closest to x
    min_diff = (np.abs(a - x)).min()
    return np.where(np.abs(a - x) == min_diff)[0][0]

def get_radius(r_file, tb, ta, ch):
    # extract radus
    if r_file == None:
        return float(ch), float(ch)
    else:
        # find the closest time
        irb = closest(tb, t_rad)
        ira = closest(ta, t_rad)
        # get the channel numbers
        cha = channel_id[:,irb].astype(int)
        # select the channel
        c_sel = (cha == ch)
        # get the before and after radius
        rb = radii[c_sel,irb][0]
        ra = radii[c_sel,ira][0]
    return rb,ra
    
def save_step(i,f_name):
    tb, ta, Rb, dRb, Ra, dRa, nrb, nra, n_ch, r_b, r_a = get_rate(steps[i])
    o = open(f_name, 'w')
    o.write('# rates for a single time step\n')
    o.write('#\ time_b = {}\n'.format(tb[0]))
    o.write('#\ time_a = {}\n'.format(ta[0]))
    o.write('#\ n_r_b = {}\n'.format(nrb[0]))
    o.write('#\ n_r_a = {}\n'.format(nra[0]))
    o.write('#! ch[i,0]/ det[i,1]/ Rb[f, 2]/ dRb[f,3]/ Ra[f,4]/ dRa[f,5]/ n_ch[i,6]/ rad_b[f,7]/ rad_a[f,8]/\n')
    for i, R_b in enumerate(Rb):
        ich = channels[i]
        idet = detectors[i]
        dR_b = dRb[i]
        R_a = Ra[i]
        dR_a = dRa[i]
        nn_ch = int(n_ch[i]) # as check
        rr_b = r_b[i]
        rr_a = r_a[i]
        o.write('{} {} {} {} {} {} {} {} {}\n'.format(ich, idet, R_b, dR_b, R_a, dR_a, nn_ch, rr_b, rr_a))
    o.close()

# control file name
c_file = []

# get the control file
parser = AG.ArgumentParser()
#parser.add_argument("control_file", nargs = '?', help="Control file ", default = 'ctrl_combine.data')
parser.add_argument("Shot", nargs = '?', help="Control file ", default = 29975)
args = parser.parse_args()
shot=args.Shot

#cdata = pfile(args.control_file)

# get the radius file if it exits
r_file = './Analysis_Results/'+str(shot)+'/emissivity_model_results/orbit_mean_rad_mid_plane_'+str(shot)+'.data'
#try:
#    r_file = cdata.get_value('radius_file')
#    print 'using radius file : ', r_file
#except:
#    r_file = None

#shot = cdata.get_value('shot_number', str)

# get directories
res_dir='./Analysis_Results/'+str(shot)+'/emissivity_model_results/'
#try:
#    rdir = cdata.get_value('result_directory')
#    res_dir = './' + shot + '/' + rdir
#except:
#    res_dir = './' + shot + '/results/'
#print 'using results directory : ', res_dir
#try:
#    sdir = cdata.get_value('steps_directory')
#    step_dir = './' + shot + '/' + sdir
#except:
#    step_dir = './' + shot + '/steps/'
#print 'using steps directory : ', step_dir
step_dir='./Analysis_Results/'+str(shot)+'/Step_Dir/'

#try:
#    cdir = cdata.get_value('control_directory')
#    ctrl_dir = './' + shot + '/' + cdir + '/'
#except:
#    ctrl_dir = './' + shot + '/control/'
#print 'using control directory : ', ctrl_dir


# read the data files to be combines

#c_file = cdata.get_value('control_files').split(',')
#print c_file
# plotting limits
(A_min, A_max) = db.retrieve('A_min, A_max','Combined_Rates', 'Shot = '+str(shot))
#A_min = cdata.get_value('A_min')
#A_max = cdata.get_value('A_max')

(t_min_pl, t_max_pl) = db.retrieve('t_min, t_max','Combined_Rates', 'Shot = '+str(shot))
#t_min_pl = cdata.get_value('t_min')
#t_max_pl = cdata.get_value('t_max')

R_exp_min = A_min #cdata.get_value('R_min')
R_exp_max = A_max #cdata.get_value('R_max')

R_norm_min = A_min#cdata.get_value('R_norm_min')
R_norm_max = A_max#cdata.get_value('R_norm_max')
y_scale = 1. #cdata.get_value('norm_scale')

# channels
# channels = map(int, cdata.get_value('channels').split(','))

# colors according to channel 
colors = ['y', 'r', 'g', 'b', 'm', 'c']

# calculate averages


R_b = []
R_a = []

# normed rates by neutron rate
R_b_norm = []
R_a_norm = []



# get ts_file first
ts_file = 'ts_file.data'
#ts_file = B.get_file(ctrl_dir + c_file[0]).par.get_value('time_slice_file')
tsf = B.get_file(step_dir + ts_file)
#
# get detector channel/ detector number arrays
(channels_str,) = db.retrieve('Channels', 'Combined_Rates', 'Shot = '+str(shot))
channels = list(map(int, channels_str.split(',')))

detectors=[1,2,3]#[1,2,3,4,5,6] #channels+1
#detectors = np.array(tsf.par.get_value('detectors').split(',')).astype(int)
#channels = np.array(tsf.par.get_value('channels').split(',')).astype(int)
print(detectors)
print(channels)
# before crash
t_min_b = np.array( tsf.get_data_eval('t_min', " data['comment'] == 'cb' ")) 
t_max_b = np.array( tsf.get_data_eval('t_max', " data['comment'] == 'cb' ")) 
# neutron rate
n_r_b = np.array( tsf.get_data_eval('neutron_rate', " data['comment'] == 'cb' ")) 

tb = np.array([t_min_b, t_max_b] ).transpose()

# after crash
t_min_a = np.array( tsf.get_data_eval('t_min', " data['comment'] == 'ca' ")) 
t_max_a = np.array( tsf.get_data_eval('t_max', " data['comment'] == 'ca' ")) 
# neutron rate
n_r_a = np.array( tsf.get_data_eval('neutron_rate', " data['comment'] == 'ca' ")) 

ta = np.array([t_min_a, t_max_a]).transpose()
    
B.pl.figure(0)

# loop over control files for the individual channels/detectors
# c_file is an array containing the control files for the various channels
for i,cf in enumerate(channels):#c_file):
    #c_data = B.get_file(ctrl_dir + cf)
    
    # get time slices for analysis
            
    #f_name = c_data.par.get_value('input_result_file', str) #fit resulst
    #pf_name = c_data.par.get_value('output_file', str) #rate resulst
    (dtmin,dtmax) = db.retrieve('dtmin, dtmax', 'Raw_Fitting',
    'Shot = '+str(shot)+' AND Channel = '+str(cf))
    pf_name= './Analysis_Results/'  + str(shot) + '/Rate_Plotting/' + "rate_results_4_"+ str(shot) + "_{0:5.3f}_{1:5.3f}_{2:d}.npz".format(dtmin, dtmax, cf)  
        
    #shot = c_data.par.get_value('shot_number',str)
    print(shot)
    #channel = c_data.par.get_value('channel_number',str)
    #ichan = int(channel)
        
    # rate data
    #pd = B.get_file(res_dir + pf_name)
    
    try:
        d = np.load(pf_name)
    except:
        print("cannot open : ", pf_name, " skipping")
        continue
    t = d['t']#B.get_data(pd, 't')
    Ap  = d['Ap']#B.get_data(pd, 'Ap')
    dAp = d['dAp']#B.get_data(pd, 'dAp')
      
    #t = B.get_data(pd, 't')
    #Ap  = B.get_data(pd, 'Ap')
    #dAp = B.get_data(pd, 'dAp')
    At  = d['At']#B.get_data(pd, 'Ap')
    dAt = d['dAt']#B.get_data(pd, 'dAp')
    
    #At  = B.get_data(pd, 'At')
    #dAt = B.get_data(pd, 'dAt')
    
    A  = d['A']#B.get_data(pd, 'Ap')
    dA = d['dA']#B.get_data(pd, 'dAp')
    
    #A  = B.get_data(pd, 'A')
    #dA = B.get_data(pd, 'dA')
    # loop over time windows, they are always in pairs before (b) and after (a)
    for j, t_wb in enumerate(tb):
        # check channel ID for consistency
        if cf != channels[i]:#int(channel) != channels[i]:
            print("Channel numbers are not consistent !", cf, channels[i])#hannel, channels[i]
        # time window
        t_wa = ta[j] 
        # neutron rates
        nr_b = n_r_b[j]
        nr_a = n_r_a[j]   
        # calculate averages
        # use only proton data
        t_mean_b, w_mean_b, sig_w_mean_b =  get_mean(t, Ap, dAp, t_wb)
        t_mean_a, w_mean_a, sig_w_mean_a =  get_mean(t, Ap, dAp, t_wa)
        # append information to an array  
        R_b.append([[t_mean_b, w_mean_b, sig_w_mean_b], cf, j, nr_b] ) #int(channel), j, nr_b] )
        R_a.append([[t_mean_a, w_mean_a, sig_w_mean_a], cf, j, nr_a])
        print('Channel : ', cf)#hannel
        print('Mean before : ', get_mean(t, Ap, dAp, t_wb))
        print('Mean after : ',  get_mean(t, Ap, dAp, t_wa)) 
        
    # Total signal
    B.plot_exp(t, Ap, dAp, color = colors[cf], label = 'Ch {}'.format(channels[i]), capsize = 0.)
    # B.plot_line(t, A, color = colors[i])
    
B.pl.xlabel('t [s]')
B.pl.ylabel('Rate')
B.pl.legend(loc = 'upper right')
B.pl.title('Shot : '+ str(shot) )
B.pl.ylim((A_min, A_max))
B.pl.xlim((t_min_pl, t_max_pl))

ymin,ymax = B.pl.ylim()
# before
B.pl.vlines(t_min_b, ymin, ymax, color = 'r')
B.pl.vlines(t_max_b, ymin, ymax, color = 'r')
# after
B.pl.vlines(t_min_a, ymin, ymax, color = 'm')
B.pl.vlines(t_max_a, ymin, ymax, color = 'm')


# get the various orbit radii for the 4 channels  
if r_file != None:
    try:
        rd = B.get_file(r_file)
    except:
        print(r_file, ' does not exist, will not use radii!')
        r_file = None
if r_file != None:
    # radii
    r1 = B.get_data(rd, 'r0')
    r2 = B.get_data(rd, 'r1')
    r3 = B.get_data(rd, 'r2')
    r4 = B.get_data(rd, 'r3')
    # times
    t_rad = B.get_data(rd, 't')/1000. # in s
    # get channel number for checking
    c1 = B.get_data(rd, 'ch0')
    c2 = B.get_data(rd, 'ch1')
    c3 = B.get_data(rd, 'ch2')
    c4 = B.get_data(rd, 'ch3')
    radii = np.array([r1, r2, r3, r4])
    channel_id = np.array([c1, c2, c3, c4])


# write averaged rate into output file
output_file = res_dir + tsf.par.get_value('pro_file')

o = open(output_file, 'w')

# write header

o.write('# rates averaged according to time slices in :{}\n'.format(tsf.filename))

o.write('# \n')

o.write('#! ch[i,0]/ t_mean_b[f,1]/ t_mean_a[f,2]/ Rb[f,3]/ dRb[f,4]/ Ra[f,5]/ dRa[f,6]/ slice[i,7]/ nRb[f,8]/ nRa[f,9]/  rad_b[f,10]/  rad_a[f,11]/ \n')

for i,l in enumerate(R_b):
    chn = l[1]
    sl = l[2]
    RR_b = l[0]
    RR_a = R_a[i][0]
    tm_b = RR_b[0]
    tm_a = RR_a[0]
    rb = RR_b[1]
    drb = RR_b[2]
    ra = RR_a[1]
    dra = RR_a[2]
    nrb = l[3]
    nra = R_a[i][3]
    rad_b,rad_a = get_radius(r_file, tm_b, tm_a, chn)
    o.write('{} {} {} {} {} {} {} {} {} {} {} {} \n'.format(chn, tm_b, tm_a,  rb, drb, ra, dra, sl, nrb, nra, rad_b, rad_a) )
o.close()

# using the data
# for using the data
# rf.get_data_list_eval('t_mean:ch:Rb:dRb:Ra:dRa',"data['slice'] == 0")

# plot sliced rates

# read back the file
rf = B.get_file(output_file)

# get the times for the crashes
# t_markers = np.array(rf.get_data('t_mean_b'))
# ymin,ymax = B.pl.ylim()

# draw vertical red lines
#B.pl.vlines(t_markers, ymin, ymax, color = 'r')

B.pl.show()

# prepare data files for profile fitting
steps =[]

# get the time slice indices
n_sl = B.get_data(rf, 'slice').max()+1
# get the data sets for the corresponding time slices containing the various channel data
for i_sl in range(n_sl):
    sel_data = "data['slice'] == {}".format(i_sl)
    steps.append(np.array(rf.get_data_list_eval('t_mean_b:t_mean_a:ch:Rb:dRb:Ra:dRa:nRb:nRa:rad_b:rad_a',sel_data)))


n_cols = 3
n_rows = np.ceil(len(steps)/float(n_cols)).astype(int)


# plot all
first = True

print("------------- data rates  -----------")
fig, p_axes = B.pl.subplots(n_rows, n_cols, sharex = True, sharey = True)
i_plot = 0
two_D = False
if len(p_axes.shape) > 1:
    p0 = p_axes[0,0]
    two_D = True
else:
    p0 = p_axes[0] 

# get the various orbit radii for the 4 channels  
if r_file != None:
    try:
        rd = B.get_file(r_file)
    except:
        print(r_file, ' does not exist, will not use radii!')
        r_file = None
if r_file != None:
    p0.set_xlim((0.75,1.35))
    p0.set_xticks([0.8,.9,1.0,1.1,1.2])
else:
    p0.set_xlim((0.,5.))
    p0.set_xticks([0,1,2,3,4])

p0.set_ylim((R_exp_min, R_exp_max))
p0.set_yticks([25e3, 50e3, 75e3, 100e3])

# set tet locations
x_text_0 = 1.0
x_text_1 = 1.0
y_text = 0.8*p0.get_ylim()[1]

for ir in range(n_rows):
    for ic in range(n_cols):
        if i_plot >= len(steps):
            break
        if two_D:
            ax = p_axes[ir, ic]
        else:
            ax = p_axes[ic]
        #ax.set_ylim((0., 250e3)
        t_b, t_a, Rb, dRb, Ra, dRa, nrb, nra, n_ch, rr_b, rr_a  = get_rate(steps[i_plot])
        if r_file != None:
            r_detectors = radii[:,0]#i_plot]
            ch_radii = channel_id[:,0]#i_plot]
            r_err = np.ones_like(r_detectors)*0.03
            # match each channel to each radius
            m_rad, m_rate = MA.match_arrays(ch_radii, n_ch)
            # draw different colors rates
            for i, rr in enumerate(r_detectors[m_rad]):
                if n_ch[m_rate][i] != ch_radii[m_rad][i]:
                    print('inconsistent channels ', i, n_ch[i], ch_radii[i])
                    continue
                print('full rate ----->', n_ch[m_rate][i], rr)
                ichan = n_ch[m_rate[i]].astype(int)
                ax.errorbar([rr], [Rb[m_rate][i]], [dRb[m_rate][i]], xerr = [r_err[i]],\
                    ls = 'None', marker = 'o', capsize = 0, color = colors[ichan])
                ax.errorbar([rr], [Ra[m_rate][i]], [dRa[m_rate][i]], xerr = [r_err[i]],\
                    ls = 'None', marker = 'D', capsize = 0, color = colors[ichan])
            txt = 't = {0:.1f}'.format(t_b[0]*1000.)        
            ax.text(x_text_0, y_text, txt)
            txt = 't = {0:.1f}'.format(t_b[0]*1000.)        
            ax.text(x_text_0, y_text, txt)
            
        else:
            for i,dd in enumerate(detectors):
                ax.errorbar([dd], [Rb[i]], [dRb[i]], ls = 'None', marker = 'o', capsize = 0, color = colors[i])
                ax.errorbar([dd], [Ra[i]], [dRa[i]], ls = 'None', marker = 'D', capsize = 0, color = colors[i])
            txt = 't = {0:.1f}'.format(t_b[0]*1000.)             
            ax.text(x_text_1,y_text, txt)
        i_plot += 1
B.pl.subplots_adjust(hspace=0, wspace = 0)
B.pl.show()


fig, p_axes1 = B.pl.subplots(n_rows, n_cols, sharex = True, sharey = True)
i_plot = 0
two_D = False
if len(p_axes1.shape) > 1:
    p1 = p_axes1[0,0]
    two_D = True
else:
    p1 = p_axes1[0]   

R_min = np.inf
R_max = -np.inf


print("------------- normalized rates -----------")

# find limits first

for ir in range(n_rows):
    for ic in range(n_cols):
        if i_plot >= len(steps):
            break
        if two_D:
            ax = p_axes1[ir, ic]
        else:
            ax = p_axes1[ic]
        # ax.set_ylim((0., 10))
        r_detectors = detectors
        t_b, t_a, Rb, dRb, Ra, dRa, nrb, nra, n_ch, rr_b, rr_a = get_rate(steps[i_plot])        
        Rb = Rb/nrb * y_scale
        dRb = dRb/nrb * y_scale
        Ra = Ra/nra * y_scale
        dRa = dRa/nra * y_scale
        R_min = min((Rb - dRb).min(), R_min)
        R_min = min((Ra - dRa).min(), R_min)
        R_max = max((Rb + dRb).max(), R_max)
        R_max = max((Ra + dRa).max(), R_max)

if r_file != None:
    p1.set_xlim((0.75,1.35))
    p1.set_xticks([0.8,.9,1.0,1.1,1.2])
else:
    p1.set_xlim((0.,5.))
    p1.set_xticks([0,1,2,3,4])

p1.set_ylim((R_norm_min, R_norm_max))

p1.yaxis.set_major_locator(MaxNLocator(5))

# set tet locations
y_text_1 = 0.8*p1.get_ylim()[1]

for ir in range(n_rows):
    for ic in range(n_cols):
        if i_plot >= len(steps):
            break
        if two_D:
            ax = p_axes1[ir, ic]
        else:
            ax = p_axes1[ic]
        # ax.set_ylim((0., 10))
        r_detectors = detectors
        t_b, t_a, Rb, dRb, Ra, dRa, nrb, nra, n_ch, rr_b, rr_a = get_rate(steps[i_plot])        
        Rb = Rb/nrb * y_scale
        dRb = dRb/nrb * y_scale
        Ra = Ra/nra * y_scale
        dRa = dRa/nra * y_scale
        R_min = min((Rb - dRb).min(), R_min)
        R_min = min((Ra - dRa).min(), R_min)
        R_max = max((Rb + dRb).max(), R_max)
        R_max = max((Ra + dRa).max(), R_max)
        if r_file != None:
            # match each channel to each radius
            r_detectors = radii[:,0]#i_plot]
            r_err = np.ones_like(r_detectors)*0.03  
            ch_radii = channel_id[:,0]#i_plot]     
            m_rad, m_rate = MA.match_arrays(ch_radii, n_ch)                   
            for i, rr in enumerate(r_detectors[m_rad]):
                if n_ch[m_rate][i] != ch_radii[m_rad][i]:
                    print('inconsistent channels ', i, n_ch[i], ch_radii[i])
                    continue
                # draw different colors rates
                ichan = n_ch[m_rate][i].astype(int)
                print('normalized ----->', ichan, n_ch[m_rate][i], rr)
                ax.errorbar([rr], [Rb[m_rate][i]], [dRb[m_rate][i]], xerr = [r_err[i]],\
                    ls = 'None', marker = 'o', capsize = 0, color = colors[ichan])
                ax.errorbar([rr], [Ra[m_rate][i]], [dRa[m_rate][i]], xerr = [r_err[i]],\
                    ls = 'None', marker = 'D', capsize = 0, color = colors[ichan])
            txt = 't = {0:.1f}'.format(t_b[0]*1000.)
            ax.text(x_text_0,y_text_1, txt)
        else:
            for i,dd in enumerate(detectors):
                ax.errorbar([dd], [Rb[i]], [dRb[i]], ls = 'None', marker = 'o', capsize = 0, color = colors[i])
                ax.errorbar([dd], [Ra[i]], [dRa[i]], ls = 'None', marker = 'D', capsize = 0, color = colors[i])
            txt = 't = {0:.1f}'.format(t_b[0]*1000.)
            ax.text(x_text_1,y_text_1, txt)
        i_plot += 1
B.pl.subplots_adjust(hspace=0, wspace = 0)
B.pl.show()

print("------------- save step data -----------")
# save all steps
for i,s in enumerate(steps):
    file_name = step_dir + 'step_{}_{}.data'.format(i,shot)
    save_step(i, file_name)
# all saved
