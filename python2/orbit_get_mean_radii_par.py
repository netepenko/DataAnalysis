# Make a file with orbit intersections
#
# calculate the mean mid-plane intersection radiuii for ensemble calculation
#
# uses 5 digit track numbering and assumes the standard format from ensemle calculations
# 
# this version is for orbit3 output

import fileinput as FI
import numpy as np
import LT.box as B
import os.path as P
import os
import database_operations as db
import glob as G
#import sys
#from LT.parameterfile import pfile


# special version for orbit data

import orbit_view_data_fast as vd
# import orbit_view_data as vd
# parse arguments
import argparse as AG


parser = AG.ArgumentParser()
#parser.add_argument("control_file", nargs = '?', help="Control file ", default = 'control_get_radii.data')
parser.add_argument("Shot", nargs = '?', help="Control file ", default = 29975)
args = parser.parse_args()
shot=args.Shot
#----------------------------------------------------------------------
# find the first mid-plane crossing of trajectory v
def get_zero_crossing(v):
    i_n = np.where(v.zt<=0.)[0][0]
    i_p = i_n - 1
    zp = v.zt[i_p]
    rp = v.rt[i_p]
    zn = v.zt[i_n]
    rn = v.rt[i_n]
    m = (zp -zn)/(rp - rn)
    r0 = rn - zn/m
    return r0
 
def get_names_ch(s):
    vf = s.replace(' ','').split(',')
    name = []
    chan = []
    for v in vf:
        fields = v.split('/')
        v_name = fields[0]
        ch = np.array(fields[1:]).astype(int)
        name.append(v_name)
        chan.append(ch)
    return name, chan

def get_magnetic_axis(of):
    z_ok = False
    r_ok = False
    for line in FI.input(of):
        if line.find('rmaxis')>=0:
            rmaxis = float(line.split('=')[1])
            r_ok = True
        if line.find('zmaxis')>=0:
            zmaxis = float(line.split('=')[1])
            z_ok = True
        if (r_ok & z_ok):
            FI.close()
            break
    return rmaxis, zmaxis 


# get parameters from file


#cd = pfile(args.control_file)

def get_times(view_names):
    times = []
    for vn in view_names:
        f = vn.split('_')
        t = []
        for ff in f:
            try:
                x = float(ff)
            except:
                continue
            t.append(x)
        times.append(t[-1])
    return times
 

# define the emissivity model
#----------------------------------------------------------------------
view_files = []
mag_axis = []

det_exp = []
ch_exp = []
R_exp = []
dR_exp = []

R_exp_n = []
dR_exp_n = []
# get data


(view_dir,)= db.retrieve('view_dir', 'Combined_Rates', 'Shot = '+str(shot))
#view_dir = cd.get_value('view_dir')
(view_name,) = db.retrieve('view_names', 'Combined_Rates', 'Shot = '+str(shot))
#view_names, v_chan = get_names_ch( cd.get_value('views') )
(channels_str,) = db.retrieve('Channels', 'Combined_Rates', 'Shot = '+str(shot))
v_chan_f = map(int, channels_str.split(','))
v_chan=[[0,1,2,3,4,5]]
view_names=[]
view_names.append(view_name)
#v_chan.append(v_chan_f)
v_chan_f = np.array(v_chan).flatten()
#print view_names
#print v_chan_f
#print v_chan
#try:
#    res_dir = cd.get_value('results_dir')
#except:
res_dir = './Analysis_Results/'+str(shot)+'/emissivity_model_results/'

(r_min,r_max)= db.retrieve('r_min, r_max', 'Combined_Rates', 'Shot = '+str(shot))
#r_min = cd.get_value('r_min')
#r_max = cd.get_value('r_max')

#shot = cd.get_value('shot', int)

channels = []
# assemble the information
# orbit view data and rate data MUST match

for i, v_d in enumerate(view_names):
    # loop over directories
    v_f = view_dir + '/' + v_d + '/'   #mymod Alex
    # get the magnetic axis data
    rm,zm = get_magnetic_axis( v_f + 'orbit_output')  
    ch = []
    for j, n in enumerate(v_chan[i]):
        n += 1
        # loop over detectors in views
        patt = 'track_{0:1d}????.data'.format(n)
        view_patt = v_f + patt
        print view_patt
        view_f = G.glob(view_patt)
        print view_f
        # contains all tracks for a given detector
        view_files.append(view_f)
        mag_axis.append((rm,zm))
        vdd = B.get_file(view_f[0])
        cc = vdd.par.get_value('channel_number',int)
        ch.append(cc)
    channels.append(ch)


# map views and channels
all_views = []
for i, vf in enumerate(view_files):
    print 'loading detecgtor : ', v_chan_f[i], ' from : '+P.split(vf[0])[0]
    views = []
    for f in vf:
        # print ' getting view : ', f
        views.append(vd.view(f))
    all_views.append(views)


# get stored channel information

xv = np.arange(len(views))


# get zero crossings
R_mid = []
for views in all_views:
    R0 = np.array([ get_zero_crossing(v) for v in views ])
    R_mid.append([R0.mean(), np.sqrt(R0.var())])
R_mid = np.array(R_mid)

# reshape array, assumes that all views have the same number of channels
R0a = R_mid[:,0].reshape(len(view_names), len(v_chan[0]))
sig_R0a = R_mid[:,1].reshape(len(view_names), len(v_chan[0]))

times = get_times(view_names) 

# plot the radii

# midplane intersection for detectors
r1 = R0a[:,0]
r2 = R0a[:,1]
r3 = R0a[:,2]
r4 = R0a[:,3]
r5 = R0a[:,4]
r6 = R0a[:,5]

dr1 = sig_R0a[:,0]
dr2 = sig_R0a[:,1]
dr3 = sig_R0a[:,2]
dr4 = sig_R0a[:,3]
dr5 = sig_R0a[:,4]
dr6 = sig_R0a[:,5]

# range in R covered (2 sigma)

B.plot_exp(times, r1, dr1, color = 'r' , label = 'view 1, ch {}'.format(channels[0][0]))
B.plot_exp(times, r2, dr2, color = 'g', label = 'view 2, ch {}'.format(channels[0][1]))
B.plot_exp(times, r3, dr3, color = 'b', label = 'view 3, ch {}'.format(channels[0][2]))
B.plot_exp(times, r4, dr4, color = 'y', label = 'view 4, ch {}'.format(channels[0][3]))
B.plot_exp(times, r5, dr5, color = 'm', label = 'view 5, ch {}'.format(channels[0][4]))
B.plot_exp(times, r6, dr6, color = 'c', label = 'view 6, ch {}'.format(channels[0][5]))

B.pl.xlabel('time (s)')
B.pl.ylabel('R mid-plane (m)')
B.pl.ylim((r_min, r_max))
B.pl.title('Mid-Plane radii for {}'.format(shot))

B.pl.legend(loc = 'upper right')

B.pl.show()

# save the radii in a file, they can be used for plotting
file_name = res_dir + 'orbit_mean_rad_mid_plane_{}.data'.format(shot)
if  not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))
if os.path.isfile(file_name):
    inp = raw_input("Do you want to overwrite the results file? (y)es or (n)o: ") 
    if inp == "yes" or inp == "y": 
       os.remove(file_name)
       print 'Old file removed.'
o = open(file_name,'w')
o.write('# mean mid-plane intersections for a range of times\n')
o.write('#! t[f,0]/ r0[f,1]/ r1[f,2]/ r2[f,3]/ r3[f,4]/ r4[f,5]/ r5[f,6]/ ch0[f,5]/ ch1[f,6]/ ch2[f,7]/ ch3[f,8]/ ch4[f,9]/ ch5[f,10]/ \n')

for i,tt in enumerate(times):
    print "writing time : ", tt, i
    cc = channels[i]
    o.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(tt, \
            r1[i], \
            r2[i], \
            r3[i], \
            r4[i], \
            r5[i], \
            r6[i], \
            cc[0], cc[1], cc[2], cc[3], cc[4], cc[5] )) #dr1[i]
# done
o.close()
