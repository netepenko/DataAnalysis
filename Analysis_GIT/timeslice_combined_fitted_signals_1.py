# creat data to determine time slices for
# profile determination
#
# ths scripts uses the ctrl_combine_29880_05.data files (and similar)
#
# allow to load an existing time slice file and 
# add new time slices ord change them

import LT.box as B
import numpy as np
#import shutil as SU
import argparse as AG
import database_operations as db
import sys
import os

fig = None

class SawTeeth:
    def __init__(self, fig, file_obj = None):
        self.fig = fig
        self.o = file_obj
        self.axes = fig.gca()
        self.ymin, self.ymax = self.axes.get_ylim()
        self.cid = None
        self.before = None
        self.after = None
        self.button = None
        self.xdata = None
        self.ydata = None
        self.before = True
        self.points_before = []
        self.points_after = []
        self.skip = True
        self.space_bar = False
        self.cidkey_press = self.fig.canvas.mpl_connect(
            'key_press_event', self.on_key_press)
        self.cidkey_release = self.fig.canvas.mpl_connect(
            'key_release_event', self.on_key_release)

    def on_key_press(self,event):
        ok = event.key == ' '
        if ok :
            self.space_bar = (not self.space_bar)
        if self.space_bar:
            print 'Skip mouse'
        else:
            print 'Use mouse'        
        sys.stdout.flush()
        
    def on_key_release(self,event):
        pass
                
    def clear_points_before(self):
        self.points_before = []
    
    def clear_points_after(self):
        self.points_after = []

    def add_crash_high_edges(self):
        #print "Pick high edges with left, low with right, terminate with middle mouse click "
        if self.cid == None:
            self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.before = True
        
    def add_crash_low_edges(self):
        print "Pick low edges, terminate with right mouse click "
        if self.cid == None:
            self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.before = False

    def load_ts_file(self, ts_file_name):
        # read existing time slice file
        dts = B.get_file(ts_file_name)
        # get the data
        tmin = np.array(dts.get_data('t_min'))
        tmax = np.array(dts.get_data('t_max'))
        comment = np.array(dts.get_data('comment'))
        # select the belore and after points
        sel_before = (comment == 'cb')
        sel_after = (comment == 'ca')
        tb = tmax[sel_before]
        ta = tmin[sel_after]
        # now add them to the existing arrays
        for i,tt in enumerate(tb): 
            self.points_before.append([tb[i], self.ymin])
            self.points_after.append([ta[i], self.ymin])
        self.sort_slices()
        
    def sort_slices(self):
        # sort the slices by time
        p_b = self.points_before
        p_a = self.points_after
        tb = np.array(p_b)[:,0]
        isort = np.argsort(tb)
        for i, i_s in enumerate(isort):
            self.points_before[i] = p_b[i_s]
            self.points_after[i] = p_a[i_s]
        # all done
            
    def show_slices(self):
        # plot the slices
        for i, b in enumerate(self.points_before):
            tb1  = b[0] - d_time
            tb2 =  b[0]
            a = self.points_after[i]
            ta1 = a[0]
            ta2 = a[0] + d_time
            B.pl.vlines(tb1, self.ymin, self.ymax, color = 'r')
            B.pl.vlines(tb2, self.ymin, self.ymax, color = 'r')
            B.pl.vlines(ta1,  self.ymin, self.ymax, color = 'm')
            B.pl.vlines(ta2,  self.ymin, self.ymax, color = 'm')
        
        
    def finished(self):
        if self.cid == None:
            return
        else:
            fig.canvas.mpl_disconnect(self.cid)
            self.cid = None   
        
    def __call__(self, event):
        # print 'Mouse Event', event
        # exit if toolbar is active
        if fig.canvas.manager.toolbar._active is not None:
            return
        if self.space_bar: return
        if event.inaxes!=self.axes: 
            print 'outside of plot!'
            return
        if event.button == 2:   #was > 1
            print "finished adding points"
            self.add_slices()
            self.finish_all()
            self.finished()
            return
        if event.button == 1: #self.before:
            print 'add before event'
            self.points_before.append([event.xdata, event.ydata])
            B.pl.vlines(event.xdata, self.ymin, event.ydata)
        if event.button == 3: #else:
            print 'add after event'
            self.points_after.append([event.xdata, event.ydata])
            B.pl.vlines(event.xdata, self.ymin, event.ydata, color='orange')
        
        fig.canvas.draw()
        #fig.update()
        
    def add_slices(self):
        # loop over  edges and add /subtrace windows
        for i, b in enumerate(self.points_before):
            tb1  = b[0] - d_time
            tb2 =  b[0]
            a = self.points_after[i]
            ta1 = a[0]
            ta2 = a[0] + d_time
            print 'adding slice : ', i
            self.o.write('\n# crash : {}\n'.format(i))
            self.o.write('{} {} {} {}\n'.format(tb1, tb2, 'cb', 1.) )
            self.o.write('{} {} {} {}\n'.format(ta1, ta2, 'ca', 1.) )

    def finish_all(self):
        self.o.close()
        print 'Closed File'
    
#def save_all():
#    SU.copy('ts_file.data', ts_file)


def a_fmt(a):
    sf = ''
    for i,aa in enumerate(a):
        sf += '{} '
    return sf


def plot_data(channels):
    
    for i,ch in enumerate(channels):#file_array):
      #cdata = B.get_file(ctrl_dir + cf)
      # get time slices for analysis         
      #f_name = "./Analysis_Results/"+str(29975)+"/Raw_Fitting/fit_results_4_"+ str(29975) + "_{0:5.3f}_{1:5.3f}_{2:d}.npz".format(0.000, 0.500, ch)#self.par['dtmin']/us, self.par['dtmax']/us, self.par['channel'])
      (dtmin,dtmax) = db.retrieve('dtmin, dtmax', 'Raw_Fitting', 'Shot = '+ str(shot) + ' AND Channel = '+ str(ch))      
      #cdata.par.get_value('input_result_file', str)
      pf_name =  "../Analysis_Results/"+str(shot)+"/Rate_Plotting/rate_results_4_"+ str(shot) + "_{0:5.3f}_{1:5.3f}_{2:d}.npz".format(dtmin, dtmax, ch)#cdata.par.get_value('output_file', str)
      
      #f_name = res_dir + f_name
      #pf_name = res_dir + pf_name
      
      # shot = cdata.par.get_value('shot_number',str)
      
      try:
          d = np.load(pf_name)
          #pd = B.get_file(pf_name)
      except:
          print "cannot open : ", pf_name, " skipping"
          continue
      t = d['t']#B.get_data(pd, 't')
      Ap  = d['Ap']#B.get_data(pd, 'Ap')
      dAp = d['dAp']#B.get_data(pd, 'dAp')
      
      

      # 
      # At  = B.get_data(pd, 'At')
      # dAt = B.get_data(pd, 'dAt')
      # 
      # A  = B.get_data(pd, 'A')
      # dA = B.get_data(pd, 'dA')
      # 
      # Total signal
      B.plot_exp(t, Ap, dAp, color = colors[ channels[i] ], ecolor='grey', label = 'Ch {}'.format(channels[i]), capsize = 0.)
      # B.plot_line(t, A, color = colors[i])
      #
      B.pl.xlabel('t [s]')
      B.pl.ylabel('Rate (p)')
      B.pl.title('Shot : '+ str(shot) )
      #B.pl.ylim((A_min, A_max))
      #B.pl.xlim((t_min, t_max))
    B.pl.legend(loc = 'upper right')
    B.pl.show()
# all done


# get the control file
parser = AG.ArgumentParser()
parser.add_argument("Shot", nargs = '?', help="Control file ", default = 29975)
#parser.add_argument("control_file", nargs = '?', help="Control file ", default = 'control_combine.data')
args = parser.parse_args()
shot=args.Shot
#cdata = pfile(args.control_file)
# get control file names
#c_file = cdata.get_value('control_files').split(',')
 

(channels_str,) = db.retrieve('Channels', 'Combined_Rates', 'Shot = '+str(shot))
#channels_str = cdata.get_value('channels')
channels = map(int, channels_str.split(','))

#detectors_str = cdata.get_value('detectors')
#detectors = map(int, detectors_str.split(','))

#shot = 29975
#shot = cdata.get_value('shot_number',str)


# get directories
#try:
#    rdir = cdata.get_value('result_directory')
#    res_dir = './' + shot + '/' + rdir
#except:
#    res_dir = './' + 'Analysis_Resutls/' + str(shot) + '/results/'
#print 'using results directory : ', res_dir
#
#try:
#    sdir = cdata.get_value('steps_directory')
#    step_dir = './' + shot + '/' + sdir
#except:
#    step_dir = './' + 'Analysis_Resutls/'+str(shot) + '/steps/'
#print 'using steps directory : ', step_dir


#try:
#    cdir = cdata.get_value('control_directory')
#    ctrl_dir = './' + shot + '/' + cdir + '/'
#except:
#    ctrl_dir = './' + shot + '/control/'
#print 'using control directory : ', ctrl_dir



# get plotting limits

(t_min, t_max) = db.retrieve('t_min, t_max','Combined_Rates', 'Shot = '+str(shot))
#t_min = 0.0#cdata.get_value('t_min')
#t_max = 0.5#cdata.get_value('t_max')

(A_min, A_max) = db.retrieve('A_min, A_max','Combined_Rates', 'Shot = '+str(shot))
#A_min = 0. #cdata.get_value('A_min')
#A_max = 150.e3#cdata.get_value('A_max')

(d_time,) = db.retrieve('d_time','Combined_Rates', 'Shot = '+str(shot))
#d_time = 100.e-3#cdata.get_value('slice_width')

colors = ['y', 'r', 'g', 'b', 'm', 'c']

# calculate averages

# get ts_file first
#cf0 = B.get_file(ctrl_dir + c_file[0])
#ts_file = cf0.par.get_value('time_slice_file')



def write_file():
    o = open(o_file,'w')
    
    part1 = \
    """
    # parameter file for further analysis
    #
    # profile data file
    #
    # rates averaged for the ranges given below will be written into this file
    # 
    """
    o.write(part1)
    # detector and channel information
    o.write(('#\ channels = ' + channels_str + '\n'))
    #o.write(('#\ detectors = ' + detectors_str + '\n'))
    
    o.write('#\ pro_file = pro_' + str(shot) + '.data\n') 
    part2 = \
    """
    # time slices for rate averaging
    #
    # these time limits are used to determine average rates to be fitted
    # with profile functions
    
    #! t_min[f,0]/  t_max[f,1]/ comment[s,2]/ neutron_rate[f,3]/ 
    """
    o.write(part2)
    
                    
    print \
    """
        Add hight edge - left click
        
        Add low edge - right click
        
        Finish and save to file - middle click
        
        Load existing TS file -
            ST.load_ts_file( file_name )
        
        Show slices:
            ST.show_slices()
    """
#    
#    """
#    Add crash data:
#        ST.add_crash_high_edges()
#        ST.add_crash_low_edges()
#    Clear data:
#        ST.clear_points_before()
#        ST.clear_points_after()
#    Add data to TS file:
#        ST.add_slices() ( write to temporary file )
#        ST.finish_all() ( close temp. file)
#    Load existing TS file
#        ST.load_ts_file( file_name)
#    Save to final file:
#        save_all() (move to final location)
#    Show slices:
#        ST.show_slices()
#    """
    plot_data(channels)
    ST = SawTeeth(fig, file_obj = o )
    ST.add_crash_high_edges()
    
o_file='../Analysis_Results/'+str(shot)+'/Step_Dir/ts_file.data' #ts_file = step_dir + ts_file
fig = B.pl.figure(1)
if  not os.path.exists(os.path.dirname(o_file)):
    os.makedirs(os.path.dirname(o_file))
if os.path.isfile(o_file):
    inp = raw_input("Do you want to overwrite the results file? (y)es or (n)o: ") 
    if inp == "yes" or inp == "y": 
        os.remove(o_file)
        print 'Old file removed.'
        write_file()
    else:
        plot_data(channels)
       #    elif inp == "no" or inp == "n": 
#       return    

