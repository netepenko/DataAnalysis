#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 17:29:48 2023

calculate RP positions

@author: boeglinw
"""

import numpy as np
import argparse as AG
import sys


#%% to handle the air flag
class ToggleAction(AG.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[2:4] != 'no')
        

#%% all positions in mm

P0 = 1200.


Pref = 1976.535094

# Standard probe length
Std_diag = 165.

# measured distance closest collimator to shaft end

Coll_dist = 149.78

Range = 92.4

PD_offset = Coll_dist - Std_diag

# air status

air_status = {True:'on', False:'off'}

#%%
parser = AG.ArgumentParser(prog = 'RP_pos', 
                           #prefix_chars='-+',
                           description = 'Calculate RP setpoints (default) or get the RP positions from setpoint values',
                           formatter_class=AG.ArgumentDefaultsHelpFormatter)
# nargs = '?' use 1 argument and if missing assign defaul value
#parser.add_argument("file", nargs='?', help="my help text for this", default = 'data.dat')
# nargs = '*' use all positional argumensts e.g. wildcards
parser.add_argument("value", nargs='*', help="enter a value", type = float)

parser.add_argument('-p', '--get_RP', help="calculate RP position", action='store_true')

parser.add_argument('-C', '--Coll_dist', help="collimator distance to shaft end (mm)", type = float, default = Coll_dist )

parser.add_argument('--air', '--no-air', help = 'air on, air off', action=ToggleAction, nargs=0, default=True)


parser.set_defaults(get_setpoint=True)
parser.set_defaults(get_RP=False)

# add an option

args = parser.parse_args()

values = np.array(args.value)


# calculate RP positions if selected

if args.get_RP:
    Setpoint_value = values
    Shaft_End_Radius = (Setpoint_value + Pref)/1000.
    if args.air:
        RP = Shaft_End_Radius + Range/1000 - args.Coll_dist/1000.
    else:
        RP = Shaft_End_Radius  - args.Coll_dist/1000.
    print(f'RP position (air {air_status[args.air]}) = {RP}')
else:
    RP = values
    if args.air:
        Probe_Top_Radius_Air_Off = RP - Range/1000
    else:
        Probe_Top_Radius_Air_Off = RP
    Shaft_End_Radius = Probe_Top_Radius_Air_Off + args.Coll_dist/1000.
    
    SV = Shaft_End_Radius*1000 - Pref
    print(f'Setpoint (air {air_status[args.air]}) = {SV}')
    

    
    
