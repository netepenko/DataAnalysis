# giving initial parameters
# fit exp. data with errors
#
# this version is for orbit205 output

import fileinput as FI
import numpy as np
import LT.box as B
import database_operations as db
#from LT.parameterfile import pfile

#import match_arrays as MA

import matplotlib.pyplot as pl

# special version for orbit data
import orbit_view_data as vd

# import module for transp calculation
import TRANSP as T

import sys
# parse arguments

import argparse as AG


#args = parser.parse_args()
parser = AG.ArgumentParser()
parser.add_argument("Shot", nargs = '?', help="Control file ", default = 29975)
args = parser.parse_args()
shot=args.Shot

figure_size = (8,6)
figure_size_2 = (6,8)

colors = ['r','g','b','y','c','m','k']

#----------------------------------------------------------------------
# clear all figures
def cl_all(n):
    for i in range(1,n+1):
        pl.figure(i);pl.clf();
    # that is all
#----------------------------------------------------------------------
def mag(r):
    return np.sqrt(np.inner(r,r))
# find the first mid-plane crossing of trajectory v
def get_zero_crossing(v):
    if v.is_NC :
        r0 = get_impact_parameter(v)
    else:
        i_n = np.where(v.zt<=0.)[0][0]
        i_p = i_n - 1
        zp = v.zt[i_p]
        rp = v.rt[i_p]
        zn = v.zt[i_n]
        rn = v.rt[i_n]
        m = (zp -zn)/(rp - rn)
        r0 = rn - zn/m
    return r0
    
def get_impact_parameter(v):
    # calculate impact parameter
    d_r_n = np.array([v.dx[0], v.dy[0]])
    d_r_n = d_r_n/mag(d_r_n)
    r_front = np.array([v.xt[0], v.yt[0]])
    alpha_min = -np.dot(d_r_n, r_front)
    r_min = r_front + alpha_min*d_r_n
    r_min_mag = mag(r_min)
    return r_min_mag

    return 1.
 
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
    
def save_matrix(m, o, write_header = True, write_header_only = False):
    """
    save a matrix in a data file format
    """
    # create header line
    nr = m.shape[0]
    nc = m.shape[1]
    if write_header:
        counter = 0
        header = '#! '
        for ic in range(nc):
            name = 'mc{0:d}[f,{1:d}]/ '.format(ic, counter)
            counter += 1
            header += name
        o.write(header + '\n')
    if write_header_only:
        return
    for ir in range(nr):
        line = ''
        for ic in range(nc):
            name = '{} '.format(m[ir, ic])
            line += name
        o.write(line + '\n')
    # all done   

#----------------------------------------------------------------------
# get a polar angle for a 2d vector
def pol_angle(rx,ry):
    cphi = rx/np.sqrt(rx**2+ry**2)
    phi = np.arccos(cphi)
    phic = np.where(ry>0, phi, 2.*np.pi - phi) # set all values where rx  < 0 to 2pi - phi
    return phic
#----------------------------------------------------------------------


#----------------------------------------------------------------------
# model functions
#----------------------------------------------------------------------

# these are the functions used in orbit to generate the data
# use these for tests

em_par_1 = np.array([1., 11.45])
def Em_mod_1(f, par = em_par_1):
    # simple power law profile
    S = par[0]*f**par[1]
    return S


em_par_4 = [1., 1.0, 0.08, 0.0, 0.15]
def Em_mod_4(r, z, par=em_par_4):
    # offset gauss in R,z
    g1 = par[0]*np.exp( -0.5*((r-par[1])/par[2])**2)
    g2 = np.exp( -0.5*((z-par[3])/par[4])**2)
    return g1*g2    

#----------------------------------------------------------------------
# fit functions
#----------------------------------------------------------------------
# simple power law
#----------------------------------------------------------------------
def Em_pow(x, r, z):
    # simple power law
    s = alpha()*(x)**lam()
    return s

def Em_pow_der(x, r, z):
    return np.array([x**lam(), alpha()*np.log(x)*x**lam()])
#    return np.array([alpha()*np.log(x)*x**lam()])


#----------------------------------------------------------------------
# power law with modulation (requires psi, r, z)
#----------------------------------------------------------------------
# from orbit output: rmaxis = 1.071290730999
#                    zmaxis = -1.46114150999999997E-004
r_maxis = 1.0
z_maxis = 0.

def Em_pow_mod(psi, r, z ):
    # r_maxis : location of magnetic axis
    rho_x = r - r_maxis
    rho_z = z - z_maxis
    phi = pol_angle(rho_x,rho_z)  # polar angle
    s = alpha()*psi**lam()*(1.+ A1()*np.cos(phi) )
    return s

def Em_pow_mods(psi, r, z ):
    # r_maxis : location of magnetic axis
    rho_x = r - r_maxis
    rho_z = z - z_maxis
    phi = pol_angle(rho_x,rho_z)  # polar angle
    s = alpha()*psi**lam()*(1.+ A1()*np.sin(phi) )
    return s

def Em_pow_mods_der(psi, r, z ):
    return np.ones_like(r)*0.1

def Em_pow_mod_der(psi, r, z ):
    return np.ones_like(r)*0.1

def Em_pow_eff(psi, r, z ):
    # calculate efficiency
    # this is programmed into orbit
    # r_maxis : location of magnetic axis
    rho_x = r - r_maxis
    rho_z = z - z_maxis
    phi = pol_angle(rho_x,rho_z)  # polar angle
    s = psi**lam()*(1.+ A1()*np.cos(phi) )
    return s

#----------------------------------------------------------------------
# simple gaussian
#----------------------------------------------------------------------
def Em_g(psi, r, z):
    # offset gauss in R,z
    R2 = (r-(r_maxis +pos()) )**2 + (z - z_maxis)**2
    # R2 = (r-(r_maxis +pos()) )**2
    g = A()*np.exp( -R2/sig()**2)
    return g
      
def Em_g_der(psi, r, z):
    return np.ones_like(r)

#----------------------------------------------------------------------
# simple asym. gaussian
#----------------------------------------------------------------------
def Em_ag(psi, r, z):
    # offset asym. gauss in r,z
    x = r-pos()
    xn = (x <= 0.)
    xp = (x > 0.)
    g = np.zeros_like(r)
    g[xn] = A()*np.exp( -(x[xn]**2 + z[xn]**2)/sign()**2)
    g[xp] = A()*np.exp( -(x[xp]**2 + z[xp]**2)/sigp()**2)
    return g
      
def Em_ag_der(psi, r, z):
    return np.ones_like(r)
                                                                                                                                                        
#----------------------------------------------------------------------
# two gaussians (Input)
#----------------------------------------------------------------------
def Em_2g(psi, r, z):
    # offset gauss in R,z
    g1 = A()*np.exp( -((r-(r_maxis+pos()))/sig())**2)
    # g2 = np.exp( -((z-posz())/sigz())**2)
    # vertical width 2 hor
    sigz = 2.*sig()
    g2 = np.exp( -((z-z_maxis)/sigz)**2)
    return g1*g2    

def Em_2g_der(psi, r, z):
    # offset gauss in R,z
    return np.ones_like(r) * 0.1
    
#----------------------------------------------------------------------
# TRANSP, fit scale factor
#----------------------------------------------------------------------
def Em_T(psi, r, z):
    return A()*transp.EM_tot( r, z)

def Em_T_der(psi, r, z):
    # offset gauss in R,z
    return transp.EM_tot(r, z)

#----------------------------------------------------------------------
# the real fitting part is here
#----------------------------------------------------------------------

# get parameters from file


#cd = pfile(args.control_file)
def TruFal(a):
    if a == 'True':
        return True
    elif a == 'False':
        return False


# Default setting is that Em depends only on psirel
(use_all_variables,) = db.retrieve('use_all_variables', 'Combined_Rates', 'Shot = '+str(shot))#cd.get_value('use_all_variables', var_type = cd.Bool)
use_all_variables = TruFal(use_all_variables)
# calculate rates
(calc_rate,) = db.retrieve('calc_rate', 'Combined_Rates', 'Shot = '+str(shot))#cd.get_value('calc_rate', var_type = cd.Bool)
calc_rate=TruFal(calc_rate)
# read the model
(model,) = db.retrieve('model', 'Combined_Rates', 'Shot = '+str(shot))#cd.get_value('fit_model')


# set the initial values for the model chosen
if model == 'pow':
    # simple power law
    alpha = 2.e5
    lam = 10.
    #lam = B.Parameter( cd.get_value('lam', var_type = float), 'lam')
    #alpha = B.Parameter( cd.get_value('alpha', var_type = float), 'alpha')
    Em_func = Em_pow
    Em_func_der = Em_pow_der
    print 'initial parameters :', alpha,'\n', lam,'\n'
    current_fit_par = [lam, alpha]

elif model == 'pow_mod':
    # modulated power law cosine
    alpha = 2.e5
    lam = 10.
    A1 = 0.1
#    alpha = B.Parameter(cd.get_value('alpha') , 'alpha')
#    lam   = B.Parameter(cd.get_value('lam') , 'lam')
#    A1    = B.Parameter(cd.get_value('A1') , 'A1')
    use_all_variables = True
    Em_func = Em_pow_mod
    Em_func_der = Em_pow_mod_der
    print 'initial parameters :', alpha(), lam(), A1()
    current_fit_par = [alpha, lam, A1]

elif model == 'pow_mods':
    # modulated power law sine
    
    alpha = B.Parameter(cd.get_value('alpha') , 'alpha')
    lam   = B.Parameter(cd.get_value('lam') , 'lam')
    A1    = B.Parameter(cd.get_value('A1') , 'A1')
    use_all_variables = True
    Em_func = Em_pow_mods
    Em_func_der = Em_pow_mods_der
    print 'initial parameters :', alpha(), lam(), A1()
    current_fit_par = [alpha, lam, A1]
    
            
elif model =='simple_gauss':
    A=B.Parameter(0.1, 'A')
    pos=B.Parameter(1.2, 'r0')
    sig=B.Parameter(0.5, 'sig')
    #A = B.Parameter(cd.get_value('A') , 'A')
    #pos   = B.Parameter(cd.get_value('r0') ,'r0')
    #sig    = B.Parameter(cd.get_value('sig') ,'sig')
    current_fit_par = [A, pos, sig]
    Em_func = Em_g
    Em_func_der = Em_g_der
    
elif model =='asym_gauss':
    A = B.Parameter(cd.get_value('A') , 'A')
    pos   = B.Parameter(cd.get_value('r0') ,'r0')
    sign    = B.Parameter(cd.get_value('sign') ,'sign')
    sigp    = B.Parameter(cd.get_value('sigp') ,'sigp')
    current_fit_par = [A, sign, sigp]
    Em_func = Em_ag
    Em_func_der = Em_ag_der    

elif model == 'two_gauss':
    # modulated power law
    A = B.Parameter(cd.get_value('A') , 'A')
    pos   = B.Parameter(cd.get_value('r0') ,'r0')
    sig    = B.Parameter(cd.get_value('sig_r') ,'sig_r')
    #posz   = B.Parameter(cd.get_value('z0') ,'z0')
    #sigz    = B.Parameter(cd.get_value('sig_z') ,'sig_z')
    use_all_variables = True
    Em_func = Em_2g
    Em_func_der = Em_2g_der
    print 'initial parameters :', A,'\n', pos,'\n', sig,'\n'
    current_fit_par = [A, pos, sig]
    
elif model == 'TRANSP':
    # use TRANSP result fit overall scale factor
    # get necessary parameters for proper initialization
    TRANSP_dir = cd.get_value('TRANSP_dir')
    BT_file = cd.get_value('BT_file')
    BB_file = cd.get_value('BB_file')
    # initialize transp
    transp = T.T_data(TRANSP_dir, BT_file, BB_file)
    transp.EM_transp_init()
    #
    A = A = B.Parameter(cd.get_value('A') , 'A')
    use_all_variables = True
    Em_func = Em_T
    Em_func_der = Em_T_der
    print 'initial parameters :', A,'\n'
    current_fit_par = [A]
else:
    print"unknown model", model
    sys.exit()

#
# define the emissivity model
#----------------------------------------------------------------------
view_files = []
view_channels = []
mag_axis = []

det_exp = []
ch_exp = []

R_exp = []
dR_exp = []

R_exp_n = []
dR_exp_n = []

is_NC = []
# get data

# used for the output file of fit results and parameters
use_data_set = 'NSTX_29975'#cd.get_value('use_data_set')

# decide if rates are normalized to total neutron detector rate
fit_normalized = True#cd.get_value('fit_normalized')

# do a combined NC PD fit
NC_combined = False#cd.get_value('NC_combined')

if NC_combined:
    NC_eff = B.Parameter(cd.get_value('NC_efficiency') , 'NC_eff')
    current_fit_par.append(NC_eff)

# every data file contains 2 data sets: before and after (typically when something happens)
# use data
use_before = True#cd.get_value('use_before', var_type = cd.Bool)

(view_dir,)= db.retrieve('view_dir', 'Combined_Rates', 'Shot = '+str(shot))
#view_dir = cd.get_value('view_dir')
(view_name,) = db.retrieve('view_names', 'Combined_Rates', 'Shot = '+str(shot))
#view_names, v_chan = get_names_ch( cd.get_value('views') )
(channels_str,) = db.retrieve('Channels', 'Combined_Rates', 'Shot = '+str(shot))
v_chan_f = map(int, channels_str.split(','))
v_chan=[]
view_names=[]
view_names.append(view_name)
v_chan.append(v_chan_f)
#view_dir = cd.get_value('view_dir')
#view_names, v_chan = get_names_ch( cd.get_value('views') )

rate_dir = './Analysis_Results/'  + str(shot) + '/Step_Dir/'# cd.get_value('rate_dir')
# r_det are the detector numbers
#rate_names,r_det =
rate_names , r_det = get_names_ch('step_1_29975/2/3, step_2_29975/2/3')# cd.get_value('rate_data'))


# assemble the PD information
# orbit view data and rate data MUST match
for i, v_d in enumerate(view_names):
    # loop over directories
    v_f = view_dir + '/' + v_d + '/'
    # get the magneti axis data
    f_magnetic_axis = v_f + 'orbit_output'
    rm,zm = get_magnetic_axis( f_magnetic_axis)
    # get the data and combine them
    f_rate_data = rate_dir + '/' + rate_names[i] + '.data'
    r_d = B.get_file(f_rate_data)
    det = B.get_data(r_d, 'det')
    det_l = list(det)
    chan = B.get_data(r_d, 'ch')
    if use_before:
        R = B.get_data(r_d, 'Rb')
        dR = B.get_data(r_d, 'dRb')
        n_r = r_d.par.get_value('n_r_b')
    else:
        R = B.get_data(r_d, 'Ra')
        dR = B.get_data(r_d, 'dRa')
        n_r = r_d.par.get_value('n_r_a')
    
    for j, n in enumerate(v_chan[i]):
        # loop over detectors in views
        view_f = v_f + 'track_{0:1d}1111.data'.format(n)
        view_data = B.get_file(view_f)
        # get the channel number for the view
        n_view_chan = view_data.par.get_value('channel_number', int)
        view_channels.append(n_view_chan)
        view_files.append(view_f)
        mag_axis.append((rm,zm))
        print 'PD view detector : ', n, ' using track file : ', view_f, ' view_channel : ', n_view_chan
        for k, n_det in enumerate(r_det[i]):
            # loop over detectors in rate data to match the channel numbers
            # find the corresponding channel number in the data
            try:
                c_i = det_l.index(n_det)
            except:
                # not found
                continue
            # make sure the channel numbers match
            if n_view_chan == chan[c_i]:
                print 'saving PD channel : ', n_view_chan
                ch_exp.append(chan[c_i])
                det_exp.append(n)
                R_exp.append(R[c_i])
                dR_exp.append(dR[c_i])
                R_exp_n.append(R[c_i]/n_r)
                dR_exp_n.append(dR[c_i]/n_r)
                is_NC.append(False)
    



# map NC views and channels
if NC_combined: 
    # NC information
    NC_view_dir = cd.get_value('NC_view_dir')
    NC_view_names, NC_v_chan = get_names_ch( cd.get_value('NC_views') )

    NC_rate_dir = cd.get_value('NC_rate_dir')
    NC_rate_names, NC_r_det = get_names_ch( cd.get_value('NC_rate_data'))

# assemble the NC information
# NC orbit view data and rate data MUST match
    for i, v_d in enumerate(NC_view_names):
        # loop over directories
        v_f = NC_view_dir + '/' + v_d 
        # get the magneti axis data
        rm,zm = get_magnetic_axis( f_magnetic_axis)
        # get the data and combine them
        f_rate_data = NC_rate_dir + '/' + NC_rate_names[i] + '.data'
        r_d = B.get_file(f_rate_data)
        chan = B.get_data(r_d, 'ch')
        det = B.get_data(r_d, 'det')
        det_l = list(det)

        if use_before:
            R = B.get_data(r_d, 'Rb')
            dR = B.get_data(r_d, 'dRb')
            n_r = r_d.par.get_value('n_r_b')
        else:
            R = B.get_data(r_d, 'Ra')
            dR = B.get_data(r_d, 'dRa')
            n_r = r_d.par.get_value('n_r_a')

        for j, n in enumerate(NC_v_chan[i]):
            # loop over detectors in views
            view_f = v_f + '{0:1d}.data'.format(n)
            view_data = B.get_file(view_f)
            # get the channel number for the view
            n_view_chan = view_data.par.get_value('channel_number', int)
            view_channels.append(n_view_chan)
            view_files.append(view_f)
            mag_axis.append((rm,zm))
            print 'NC view detector : ', n, ' using track file : ', view_f, ' view_channel : ', n_view_chan
            for k, n_det in enumerate(NC_r_det[i]):
                # loop over detectors in rate data to match the channel numbers
                # find the corresponding channel number in the data
                try:
                    c_i = det_l.index(n_det)
                except:
                    # not found
                    continue
                # make sure the channel numbers match
                if n_view_chan == chan[c_i]:
                    print 'saving NC channel : ', n_view_chan
                    ch_exp.append(chan[c_i])
                    det_exp.append(n)
                    R_exp.append(R[c_i])
                    dR_exp.append(dR[c_i])
                    R_exp_n.append(R[c_i]/n_r)
                    dR_exp_n.append(dR[c_i]/n_r)
                    is_NC.append(True)
    

    # setting up exp. data

exp_data = R_exp
exp_err = dR_exp

if fit_normalized:
    exp_data = R_exp_n
    exp_err = dR_exp_n


# now read the data

views = []

for i,f in enumerate(view_files):
    print i
    print f
    vv = vd.view(f, neutron_camera = False) # is_NC[i]) my mod Alex
    views.append(vv)
xv = np.arange(len(views))

# get zero crossings
R0 = np.array([ get_zero_crossing(v) for v in views ])


# define the fitting function:
def S_f(x):
    eff = []
    for i in x:
        global r_maxis, z_maxis
        r_maxis, z_maxis= mag_axis[i]
        v = views[i]
        if v.is_NC:
            eff.append(NC_eff()*v.get_eff(Em_func, use_all = use_all_variables, get_rate = calc_rate) )
        else:
            eff.append(v.get_eff(Em_func, use_all = use_all_variables, get_rate = calc_rate) )
    return np.array(eff)

# to calculate efficiencies
def S_eff(x):
    eff = []
    for i in x:
        global r_maxis, z_maxis
        r_maxis, z_maxis= mag_axis[i]        
        eff.append(views[i].get_eff(Em_func, use_all = use_all_variables, get_rate = False) )
    return np.array(eff)    

print '-------------------------------------------------------------------'
print 'fitting  data set : ', use_data_set, ' with mode : ', model
print '-------------------------------------------------------------------'

orbit_fit = B.genfit(S_f, current_fit_par ,\
              y = exp_data, \
              x = xv, \
              y_err = exp_err, \
              full_output=1,\
                nplot = 0, \
              ftol = 0.001, maxfev = 2000
                  )
# fitting using analytically calculated derivatives does not work well
#                  deriv = Em_func_der, \
#
# get fit values
stat = orbit_fit.stat
fitted_rate = stat['fitted values']
# get covariance matrix
if orbit_fit.covar == None:
    print "fit did not converge !' "
    for k in stat.keys():
        print '------------------------------------------------------'
        print k, ' = ', stat[k]
    print '------------------------------------------------------'
    sys.exit()
#
# take chi square into account for error estimate
mcov = orbit_fit.covar *  orbit_fit.chi2_red

# ratio between exp. and fit
r = exp_data/fitted_rate
dr = exp_err/fitted_rate

# plot the results:
# first the data with error bars
pl.ioff()
# what the fit produced
fig1 = pl.figure(1, figsize = figure_size_2)
# compare the fitted eff. with the exp. one
pl.subplot(2,1,1)
pl.errorbar(R0, exp_data, yerr = exp_err, marker = 'o', color='r', ls = 'None')
pl.plot(R0, fitted_rate, 'bD')
pl.ylabel('counts')
#ymin,ymax = pl.ylim()
#if ymin < 0.:
#    pl.ylim( (0., ymax) )
# done
pl.subplot(2,1,2)
# compare the fitted emissivity with the model (input) one
psirel = np.linspace(0.001, 1., 101)
# calculate errors for the fittet emissivity at each psi value
# using the uncertainty of the fit
if use_all_variables:
    for i,vv in enumerate(views):
        ic = i%7
        Em_fit_view = Em_func(vv.pos_psirel, vv.rt[vv.in_pos_psirel], vv.zt[vv.in_pos_psirel])
        Em_diff_view = (Em_fit_view - vv.Ema)/vv.Ema* 100.
        pl.plot(vv.pos_psirel, Em_fit_view, color = colors[ic] )
        # pl.plot(vv.pos_psirel, vv.Ema, color = colors[ic], ls = '--' )
else:
    dEm_dpar = Em_func_der(psirel)
    der = np.matrix( dEm_dpar )
    Em_fit = Em_func(psirel)
    ss=(der.transpose()*mcov)*der
    # these are the errors for each view
    de_err = np.sqrt( np.diag(ss) )
    # fitted function
    Em_high = Em_fit + de_err
    Em_low =  Em_fit - de_err
    pl.fill_between(psirel, Em_high, Em_low, color = 'b')
    pl.plot(psirel, Em_fit, color = 'c', lw = 2.)
# use the model from Orbit
#for vv in views:
#    pl.plot(vv.pos_psirel, vv.Ema)
#
pl.xlabel(r'$\psi_{rel}$')
pl.ylabel(r'$S(\psi_{rel})$')
# done
pl.subplots_adjust(left=0.20, bottom = 0.12)
pl.show()

fig2 = pl.figure(2, figsize = figure_size)
pl.errorbar(R0, r, yerr = dr, marker = 'o', color='r', ls = 'None')
ymin,ymax = pl.ylim()
pl.xlabel('R_0 (m)')
pl.ylabel('ratio exp/fit')
if ymin < 0.8:
    pl.ylim( (0.8, ymax) )
ymin,ymax = pl.ylim()
if ymax > 1.2:
    pl.ylim( (ymin, 1.2) )

pl.subplots_adjust(left=0.15, bottom = 0.15)
pl.show()

# write results into results file
# output file name
# fit results
res_f = use_data_set+'_'+model+'_res.data'
out_res = open(res_f, 'w')
# write exp. values and fitted values
#
out_res.write('# fit results and exp. rates\n')
out_res.write('#! R[f,0]/ R_exp[f,1]/ dR_exp[f,2]/ R_fit{f,3]/\n')

for i, R_val in enumerate(R0):
    out_res.write('{} {} {} {}\n'.format(R_val, exp_data[i], exp_err[i], fitted_rate[i]))
out_res.close()

# parameter results
out_f = use_data_set+'_'+model+'.data'
out = open(out_f, 'w')


print 'Model used : ', model
out.write('#\ Model = {}\n\n'.format( model))

out.write('#\ control_file = {}\n\n'.format('no_ctrlfiel'))#cd.filename))


print 'reduced chi square: ', orbit_fit.chi2_red
out.write('#\ reduced_chi_square = {}\n\n'.format( orbit_fit.chi2_red))

for i,p_val in enumerate(current_fit_par):
    print 'final parameter : ',i, ' = ', p_val
    out.write('# final parameter : {} = {}\n'.format(i, p_val))
# that's it


out.write('\n\n')

for p in current_fit_par:
    print p.name, '=', p()
    name, value, error = p.get()
    out.write('#\ {} = {}; sig_{} = {} \n'.format(name, value, name, error))

out.write('\n\n')

# output for nml file
for i,p in enumerate(current_fit_par):
    print '    par({}) = {}'.format(i+1, p())
    out.write('#    par({}) = {}\n'.format(i+1, p()))

# calculate total rates
#
# efficiencies with fitted values (this should be the same as in the orbit output)
#
# to do this orbit need to have been recalculated with the final parameters
eff_c = S_eff(xv)

total_rate = fitted_rate/eff_c
out.write('\n\n')

print "Total rate : ", total_rate
out.write("#\ total_rate = {} \n\n".format(total_rate[0]))

# save covariance matrix

out.write('# Covariance matrix, scaled by chi2_red\n\n')
out.write('#    mc0, mc1 ...: the column vectors, each line is a row \n\n')
save_matrix(mcov, out)

out.close()

