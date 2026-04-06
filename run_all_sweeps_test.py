"""
exec(open("run_all_sweeps_test.py").read())
"""

import os
import sys
sys.path.insert(0, './helper_libs')
import numpy as np
#import matplotlib.pyplot as plt
import runge_kutta_fourth_order_coupled_oscillators as rk
from datetime import datetime
import pandas as pd
#import pickle
import pickle_helpers as ph
import matplotlib as mpl
import argparse
import re
import matplotlib.pyplot as plt
mpl.rcParams['agg.path.chunksize'] = 10000

tic = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--num_t_shift_steps',default=3,type=int)
parser.add_argument('--af_block_lim',nargs='+',default=['0','1']) # format: 1 2
parser.add_argument('--t_shift_block_lim',nargs='+',default=['0','3'])
parser.add_argument('--af_block_size',default=1,type=int)
parser.add_argument('--t_shift_block_size',default=1,type=int)
parser.add_argument('--af_list',nargs='+',default=['1000.0'])
parser.add_argument('--signal_mag',default=0,type=float)
parser.add_argument('--total_t',default=10.0,type=float)
parser.add_argument('--max_t_shift',default=300.0,type=float)
parser.add_argument('--min_run_index',default=1,type=int)
parser.add_argument('--max_run_index',default=6,type=int)
parser.add_argument('--data_folder_num',default=58,type=int)
parser.add_argument('--phi1',default=0,type=float)
parser.add_argument('--phi3',default=np.pi,type=float)
parser.add_argument('--frequency',default=0.2,type=float)
parser.add_argument('--spect_num',default=5000,type=int)
parser.add_argument('--save_verbosity_index',default=500,type=int)
parser.add_argument('--input_filename',default='../mad_data_12-4-25/prepped_data/flights1_mod1.csv')
parser.add_argument('--del_t_ff',default=0.004,type=float)
parser.add_argument('--x10',default=0,type=float)
parser.add_argument('--z10',default=0,type=float)
parser.add_argument('--x20',default=0,type=float)
parser.add_argument('--z20',default=0,type=float)
parser.add_argument('--x30',default=0,type=float)
parser.add_argument('--z30',default=0,type=float)
parser.add_argument('--t_sig_start',default=0,type=float)
parser.add_argument('--logistic_parameter',default=100.0,type=float)
parser.add_argument('--noise_flag',action='store_false')
parser.add_argument('--A1',default=10000.0,type=float)
parser.add_argument('--sig_repeat_timestep_factor',default=1,type=int)
parser.add_argument('--sensor_increment',default=0,type=int)
parser.add_argument('--pulse1_height',default=0,type=float)
parser.add_argument('--pulse1_center',default=40.0)
parser.add_argument('--pulse1_width',default=(1.0/135))
parser.add_argument('--pulse1_type',default=0)
parser.add_argument('--pulse2_height',default=1.0,type=float)
parser.add_argument('--pulse2_center',default=80.0)
parser.add_argument('--pulse2_width',default=(1.0/135))
parser.add_argument('--pulse2_type',default=5)
parser.add_argument('--pulse3_height',default=0,type=float)
parser.add_argument('--pulse3_center',default=40.0)
parser.add_argument('--pulse3_width',default=(1.0/135))
parser.add_argument('--pulse3_type',default=0)
parser.add_argument('--Cff1',default=0)
parser.add_argument('--Cff2',default=1.0)
parser.add_argument('--Cff3',default=0)
parser.add_argument('--t_offset1',default=0)
parser.add_argument('--t_offset2',default=0)
parser.add_argument('--t_offset3',default=0)
parser.add_argument('--pulse_filename',default='../weak_signals/sensor_data/prepped_data/pulse1.csv')
parser.add_argument('--gamma1',default=0.7)
parser.add_argument('--gamma2',default=1.1)
parser.add_argument('--alpha1',default=1e4)
parser.add_argument('--alpha2',default=1e4)
parser.add_argument('--beta1',default=0)
parser.add_argument('--C1',default=1.0)
parser.add_argument('--kappa12',default=300)
parser.add_argument('--abs_pulse_location_flag',default=1)
parser.add_argument('--snap_flag',default=0)
parser.set_defaults(noise_flag=True) 
args = parser.parse_args()

if (args.af_block_lim != None) and (args.af_block_lim != 'None'):
    amp_factor_block_lim = [int(x) for x in args.af_block_lim]
else:
    amp_factor_block_lim = None
    
if (args.t_shift_block_lim != None) and (args.t_shift_block_lim != 'None'):
    t_shift_block_lim = [int(x) for x in args.t_shift_block_lim]
else:
    t_shift_block_lim = None

if not os.path.isdir('./data/data'+str(args.data_folder_num)):
    os.system('mkdir data/data'+str(args.data_folder_num))

gamma1 = float(args.gamma1)
gamma2 = float(args.gamma2)
gamma3 = gamma1
alpha1 = float(args.alpha1)
alpha2 = float(args.alpha2)
alpha3 = alpha1
beta1 = float(args.beta1)
beta2 = beta1
beta3 = beta1
C1 = float(args.C1)
C2 = C1
C3 = C1
Cff1 = float(args.Cff1)
Cff2 = float(args.Cff2)
Cff3 = float(args.Cff3)
kappa12 = float(args.kappa12)
kappa13 = 0
kappa21 = kappa12
kappa23 = kappa12
kappa31 = 0
kappa32 = kappa12
A1 = float(args.A1)
A3 = A1

f = float(args.frequency)
omega1 = 2*np.pi*f
omega2 = omega1
omega3 = omega1
phi1 = float(args.phi1)
phi2 = 0
phi3 = float(args.phi3)

t_sig_start = float(args.t_sig_start)
logistic_parameter = float(args.logistic_parameter)

amp_factor_list = np.array([float(x) for x in args.af_list])
amp_factor_block_size = int(args.af_block_size)
num_amp_factor_blocks = int((len(amp_factor_list))/amp_factor_block_size)
save_verbosity_index = int(args.save_verbosity_index)
noise_ff_flag = bool(args.noise_flag)
sig_repeat_timestep_factor = int(args.sig_repeat_timestep_factor)
sensor_increment = int(args.sensor_increment)
pulse1_height = float(args.pulse1_height)
pulse1_center = float(args.pulse1_center)
pulse1_width = float(args.pulse1_width)
pulse1_type = int(args.pulse1_type)
pulse2_height = float(args.pulse2_height)
pulse2_center = float(args.pulse2_center)
pulse2_width = float(args.pulse2_width)
pulse2_type = int(args.pulse2_type)
pulse3_height = float(args.pulse3_height)
pulse3_center = float(args.pulse3_center)
pulse3_width = float(args.pulse3_width)
pulse3_type = int(args.pulse3_type)
t_offset1 = float(args.t_offset1)
t_offset2 = float(args.t_offset2)
t_offset3 = float(args.t_offset3)
abs_pulse_location_flag = bool(int(args.abs_pulse_location_flag))
snap_flag = bool(int(args.snap_flag))

if amp_factor_block_lim != None:
    amp_factor_block_list = list(range(amp_factor_block_lim[0],amp_factor_block_lim[1]))
    for i in range(len(amp_factor_block_list)): amp_factor_block_list[i]=int(amp_factor_block_list[i])
else:
    amp_factor_block_list = list(range(num_amp_factor_blocks))
    
amp_factor_lists = []
for i in range(num_amp_factor_blocks):
    if i in amp_factor_block_list:
        temp_list = amp_factor_list[i*amp_factor_block_size:(i+1)*amp_factor_block_size] 
        amp_factor_lists.append(temp_list)

del_t_ff = float(args.del_t_ff)
if args.input_filename[-4:] == '.pkl':
    df = ph.load_pickle(args.input_filename)
    total_t_all = del_t_ff*len(df['ff'])
else:
    df = np.loadtxt(str(args.input_filename),delimiter=',')
    total_t_all = del_t_ff*len(df[:,0])

if str(args.pulse_filename) != 'None':
    dfp = np.loadtxt(str(args.pulse_filename),delimiter=',')
    t_pulse_loaded = np.array(dfp[:,0])
    pulse_loaded = np.array(dfp[:,1])
else:
    t_pulse_loaded = np.array([])
    pulse_loaded = np.array([])

num_t_shift_steps = int(args.num_t_shift_steps)
t_shift_start = 0
if snap_flag:
    t_shift_end = (1/f)*int(np.floor(float(args.max_t_shift)/(1/f)))
    t_shift_step = round( ( np.round(t_shift_end/(1/f)) - np.round(t_shift_start/(1/f)) )/num_t_shift_steps )*(1/f)
else:
    t_shift_end = float(args.max_t_shift)
    t_shift_step = ( t_shift_end - t_shift_start )/num_t_shift_steps
t_shift_end = t_shift_start + num_t_shift_steps*t_shift_step
total_t = np.floor( (total_t_all - t_shift_end)/del_t_ff )*del_t_ff
if total_t > float(args.total_t):
    total_t = np.floor( (args.total_t)/del_t_ff )*del_t_ff

t_shift_block_size = int(args.t_shift_block_size)
num_t_shift_blocks = int(num_t_shift_steps/t_shift_block_size)

if t_shift_block_lim != None:
    t_shift_block_list = list(range(t_shift_block_lim[0],t_shift_block_lim[1]))
    for i in range(len(t_shift_block_list)): t_shift_block_list[i]=int(t_shift_block_list[i])
else:
    t_shift_block_list = list(range(num_t_shift_blocks))
    
t_shift_lists = []
for i in range(num_t_shift_blocks):
    if i in t_shift_block_list:
        temp_list = np.linspace(t_shift_start+i*t_shift_block_size*t_shift_step,t_shift_start+(i+1)*t_shift_block_size*t_shift_step-t_shift_step,t_shift_block_size)
        t_shift_lists.append(temp_list)

print("Starting run with parameters "+str(args))

#interp_type = "trigonometric"
interp_type = "cubic spline"

x10 = float(args.x10)
z10 = float(args.z10)
x20 = float(args.x20)
z20 = float(args.z20)
x30 = float(args.x30)
z30 = float(args.z30)

for run_index in range(int(args.min_run_index),int(args.max_run_index)):
    
    if (args.signal_mag != 0) or (args.pulse2_height != 0) or (args.pulse1_height != 0):
        save_name = "data/data"+str(args.data_folder_num)+"/sweep_run" + str(run_index)+"_sig"
    else:
        save_name = "data/data"+str(args.data_folder_num)+"/sweep_run" + str(run_index)

    del_t = del_t_ff/4

    t = np.arange(0,total_t,del_t)

    if noise_ff_flag:
        if args.input_filename[-4:] == '.pkl':
            ff = df['ff']
        else:
            ff = df[:,run_index]
    else:
        if args.input_filename[-4:] == '.pkl':
            ff = np.zeros(len(df['ff']))
        else:
            ff = np.zeros(len(df[:,run_index]))

    t_ff = np.linspace(0,(len(ff)-1)*del_t_ff,len(ff))

    t_shift = 0

    del_t_ratio = int(del_t_ff/del_t)

    t_shift_ind = np.argmin(np.abs(t_shift-t_ff))
    tmax_ind = t_shift_ind + int(np.floor(total_t/del_t_ff))
    t = np.arange(t_ff[0],t_ff[int(np.floor(total_t/del_t_ff))],del_t)
    t_ff2_clip = t_ff[t_shift_ind:tmax_ind]
    t_ff13_clip = t_ff[0:len(t_ff2_clip)]
    ff_clip = ff[t_shift_ind:tmax_ind]
    
    logistic_fxn = 1/(1+np.exp(-1*logistic_parameter*(t_ff13_clip-t_sig_start)))
    if np.sum(np.isnan(logistic_fxn)) > 0:
        print("ERRROR: NaN values present in signal")
    
    A2 = 0
    signal_mag = args.signal_mag

    print('Signal magnitude = '+str(signal_mag)+'\n')

    for n in range(len(amp_factor_block_list)):
        
        for i in range(len(amp_factor_lists[n])):
            
            if np.abs(signal_mag) > 0:
                A2 = float(signal_mag)*amp_factor_lists[n][i]
            
            for m in range(len(t_shift_block_list)):
                
                pkl_name = './'+save_name+'_amp_factor_block_'+str(amp_factor_block_list[n])+'_'+str(i)+'_t_shift_block_'+str(t_shift_block_list[m])+'.pkl'
                if not os.path.isfile(pkl_name):
                    
                    z1_ff_freq_list = []
                    z2_ff_freq_list = []
                    z3_ff_freq_list = []
                    
                    x1_list = []
                    x2_list = []
                    x3_list = []
                    
                    z1_list = []
                    z2_list = []
                    z3_list = []
                    
                    for j in range(len(t_shift_lists[m])):
                    
                        if noise_ff_flag:
                            if args.input_filename[-4:] == '.pkl':
                                ff = df['ff']
                            else:
                                ff = df[:,run_index+sensor_increment*m]
                        
                        t_shift = t_shift_lists[m][j]
                        
                        t_shift_ind = np.argmin(np.abs(t_shift-t_ff))
                        tmax_ind = t_shift_ind + int(np.floor(total_t/del_t_ff))
                        
                        t_offset1_ind = int(np.floor(t_offset1/del_t_ff))
                        t_offset2_ind = int(np.floor(t_offset2/del_t_ff))
                        t_offset3_ind = int(np.floor(t_offset3/del_t_ff))
                        
                        t_ff2_clip = t_ff[t_shift_ind:tmax_ind]
                        t_ff13_clip = t_ff[0:len(t_ff2_clip)]
                        ff1_clip_raw = amp_factor_lists[n][i]*Cff1*ff[(t_offset1_ind+t_shift_ind):(t_offset1_ind+tmax_ind)]
                        ff2_clip_raw = amp_factor_lists[n][i]*Cff2*ff[(t_offset2_ind+t_shift_ind):(t_offset2_ind+tmax_ind)]
                        ff3_clip_raw = amp_factor_lists[n][i]*Cff3*ff[(t_offset3_ind+t_shift_ind):(t_offset3_ind+tmax_ind)]
                        if sig_repeat_timestep_factor > 1:
                            t_sig_clip = np.repeat(t_ff2_clip,sig_repeat_timestep_factor)
                            t_sig_clip = t_sig_clip[0:len(t_ff2_clip)]
                            omega_sig = sig_repeat_timestep_factor*omega2
                        else:
                            t_sig_clip = t_ff2_clip
                            omega_sig = omega2
                        
                        if abs_pulse_location_flag:
                            pulse_offset = 0
                        else:
                            pulse_offset = t_sig_clip[0]
                        
                        signal1 = np.zeros(len(t_sig_clip))
                        if pulse1_type == 0:
                            signal1 += amp_factor_lists[n][i]*(pulse1_height)*np.exp(-((t_sig_clip-(pulse_offset+pulse1_center))**2)/(2*pulse1_width**2))
                        elif pulse1_type == 1:
                            new_pulse = np.sqrt(24.0/(5*np.pi)) * (1 - (5.0/3)*((t_sig_clip-(pulse_offset+pulse1_center))/pulse1_width)**2) / (1 + ((t_sig_clip-(pulse_offset+pulse1_center))/pulse1_width)**2)**2.5
                            signal1 += amp_factor_lists[n][i]*new_pulse*(pulse1_height)/np.max(new_pulse)
                        elif pulse1_type == 2:
                            new_pulse = np.sqrt(128.0/(5*np.pi)) * ((t_sig_clip-(pulse_offset+pulse1_center))*pulse1_width) / (1 + ((t_sig_clip-(pulse_offset+pulse1_center))/pulse1_width)**2)**2.5
                            signal1 += amp_factor_lists[n][i]*new_pulse*(pulse1_height)/np.max(new_pulse)
                        elif pulse1_type == 3:
                            new_pulse = np.sqrt(128.0/(3*np.pi)) * (((t_sig_clip-(pulse_offset+pulse1_center))/pulse1_width)**2) / (1 + ((t_sig_clip-(pulse_offset+pulse1_center))/pulse1_width)**2)**2.5
                            signal1 += amp_factor_lists[n][i]*new_pulse*(pulse1_height)/np.max(new_pulse)
                        elif pulse1_type == 4:
                            signal1 = amp_factor_lists[n][i]*(pulse1_height)*np.exp(-((t_sig_clip-(pulse_offset+pulse1_center))**2)/(2*pulse1_width**2))*np.sin(omega_sig*t_sig_clip+phi2)
                        elif pulse1_type == 5:
                            t_pulse_loaded_clip = t_pulse_loaded + del_t_ff*np.round( ( (pulse_offset+pulse1_center) - ((np.max(t_pulse_loaded)-np.min(t_pulse_loaded))/2)  )/del_t_ff)

                            pulse_loaded_clip = amp_factor_lists[n][i]*(pulse1_height/np.max(pulse_loaded))*pulse_loaded[t_pulse_loaded_clip<=t_sig_clip[-1]]
                            t_pulse_loaded_clip = t_pulse_loaded_clip[t_pulse_loaded_clip<=t_sig_clip[-1]]
                            pulse_loaded_clip = pulse_loaded_clip[t_pulse_loaded_clip>=t_sig_clip[0]]
                            t_pulse_loaded_clip = t_pulse_loaded_clip[t_pulse_loaded_clip>=t_sig_clip[0]]
                            
                            if len(t_pulse_loaded_clip) > 1:
                                ind_p_start = np.argmin(np.abs(t_sig_clip-t_pulse_loaded_clip[0]))
                                ind_p_end = np.argmin(np.abs(t_sig_clip-t_pulse_loaded_clip[-1])) + 1
                                signal1[ind_p_start:ind_p_end] += pulse_loaded_clip
                        
                        
                        signal2 = np.zeros(len(t_sig_clip))
                        if pulse2_type == 0:
                            signal2 += amp_factor_lists[n][i]*(pulse2_height)*np.exp(-((t_sig_clip-(pulse_offset+pulse2_center))**2)/(2*pulse2_width**2))
                        elif pulse2_type == 1:
                            new_pulse = np.sqrt(24.0/(5*np.pi)) * (1 - (5.0/3)*((t_sig_clip-(pulse_offset+pulse2_center))/pulse2_width)**2) / (1 + ((t_sig_clip-(pulse_offset+pulse2_center))/pulse2_width)**2)**2.5
                            signal2 += amp_factor_lists[n][i]*new_pulse*(pulse2_height)/np.max(new_pulse)
                        elif pulse2_type == 2:
                            new_pulse = np.sqrt(128.0/(5*np.pi)) * ((t_sig_clip-(pulse_offset+pulse2_center))*pulse2_width) / (1 + ((t_sig_clip-(pulse_offset+pulse2_center))/pulse2_width)**2)**2.5
                            signal2 += amp_factor_lists[n][i]*new_pulse*(pulse2_height)/np.max(new_pulse)
                        elif pulse2_type == 3:
                            new_pulse = np.sqrt(128.0/(3*np.pi)) * (((t_sig_clip-(pulse_offset+pulse2_center))/pulse2_width)**2) / (1 + ((t_sig_clip-(pulse_offset+pulse2_center))/pulse2_width)**2)**2.5
                            signal2 += amp_factor_lists[n][i]*new_pulse*(pulse2_height)/np.max(new_pulse)
                        elif pulse2_type == 4:
                            signal2 = amp_factor_lists[n][i]*(pulse2_height)*np.exp(-((t_sig_clip-(pulse_offset+pulse2_center))**2)/(2*pulse2_width**2))*np.sin(omega_sig*t_sig_clip+phi2)
                        elif pulse2_type == 5:
                            t_pulse_loaded_clip = t_pulse_loaded + del_t_ff*np.round( ( (pulse_offset+pulse2_center) - ((np.max(t_pulse_loaded)-np.min(t_pulse_loaded))/2)  )/del_t_ff)

                            pulse_loaded_clip = amp_factor_lists[n][i]*(pulse2_height/np.max(pulse_loaded))*pulse_loaded[t_pulse_loaded_clip<=t_sig_clip[-1]]
                            t_pulse_loaded_clip = t_pulse_loaded_clip[t_pulse_loaded_clip<=t_sig_clip[-1]]
                            pulse_loaded_clip = pulse_loaded_clip[t_pulse_loaded_clip>=t_sig_clip[0]]
                            t_pulse_loaded_clip = t_pulse_loaded_clip[t_pulse_loaded_clip>=t_sig_clip[0]]
                            
                            if len(t_pulse_loaded_clip) > 1:
                                ind_p_start = np.argmin(np.abs(t_sig_clip-t_pulse_loaded_clip[0]))
                                ind_p_end = np.argmin(np.abs(t_sig_clip-t_pulse_loaded_clip[-1])) + 1
                                signal2[ind_p_start:ind_p_end] += pulse_loaded_clip
                        
                        if t_sig_start == 0:
                            signal2 += (A2)*np.sin(omega_sig*t_sig_clip+phi2)
                        else:
                            signal2 += logistic_fxn*(A2)*np.sin(omega_sig*t_sig_clip+phi2)
                            
                        signal3 = np.zeros(len(t_sig_clip))
                        if pulse3_type == 0:
                            signal3 += amp_factor_lists[n][i]*(pulse3_height)*np.exp(-((t_sig_clip-(pulse_offset+pulse3_center))**2)/(2*pulse3_width**2))
                        elif pulse3_type == 1:
                            new_pulse = np.sqrt(24.0/(5*np.pi)) * (1 - (5.0/3)*((t_sig_clip-(pulse_offset+pulse3_center))/pulse3_width)**2) / (1 + ((t_sig_clip-(pulse_offset+pulse3_center))/pulse3_width)**2)**2.5
                            signal3 += amp_factor_lists[n][i]*new_pulse*(pulse3_height)/np.max(new_pulse)
                        elif pulse3_type == 2:
                            new_pulse = np.sqrt(128.0/(5*np.pi)) * ((t_sig_clip-(pulse_offset+pulse3_center))*pulse3_width) / (1 + ((t_sig_clip-(pulse_offset+pulse3_center))/pulse3_width)**2)**2.5
                            signal3 += amp_factor_lists[n][i]*new_pulse*(pulse3_height)/np.max(new_pulse)
                        elif pulse3_type == 3:
                            new_pulse = np.sqrt(128.0/(3*np.pi)) * (((t_sig_clip-(pulse_offset+pulse3_center))/pulse3_width)**2) / (1 + ((t_sig_clip-(pulse_offset+pulse3_center))/pulse3_width)**2)**2.5
                            signal3 += amp_factor_lists[n][i]*new_pulse*(pulse3_height)/np.max(new_pulse)
                        elif pulse3_type == 4:
                            signal3 = amp_factor_lists[n][i]*(pulse3_height)*np.exp(-((t_sig_clip-(pulse_offset+pulse3_center))**2)/(2*pulse3_width**2))*np.sin(omega_sig*t_sig_clip+phi2)
                        elif pulse3_type == 5:
                            t_pulse_loaded_clip = t_pulse_loaded + del_t_ff*np.round( ( (pulse_offset+pulse3_center) - ((np.max(t_pulse_loaded)-np.min(t_pulse_loaded))/2)  )/del_t_ff)

                            pulse_loaded_clip = amp_factor_lists[n][i]*(pulse3_height/np.max(pulse_loaded))*pulse_loaded[t_pulse_loaded_clip<=t_sig_clip[-1]]
                            t_pulse_loaded_clip = t_pulse_loaded_clip[t_pulse_loaded_clip<=t_sig_clip[-1]]
                            pulse_loaded_clip = pulse_loaded_clip[t_pulse_loaded_clip>=t_sig_clip[0]]
                            t_pulse_loaded_clip = t_pulse_loaded_clip[t_pulse_loaded_clip>=t_sig_clip[0]]
                            
                            if len(t_pulse_loaded_clip) > 1:
                                ind_p_start = np.argmin(np.abs(t_sig_clip-t_pulse_loaded_clip[0]))
                                ind_p_end = np.argmin(np.abs(t_sig_clip-t_pulse_loaded_clip[-1])) + 1
                                signal3[ind_p_start:ind_p_end] += pulse_loaded_clip
                        
                        ff1_clip = ff1_clip_raw + (A1)*np.sin(omega1*t_ff13_clip+phi1) + signal1
                        ff2_clip = ff2_clip_raw + signal2
                        ff3_clip = ff3_clip_raw + (A3)*np.sin(omega3*t_ff13_clip+phi3) + signal3
                        
                        print("\nff mod block "+str(amp_factor_block_list[n])+" of "+str(amp_factor_block_list[-1])+" | iteration "+str(i)+" of "+str(len(amp_factor_lists[n]))+" | amp_factor = "+str(amp_factor_lists[n][i]))
                        print("t_shift block "+str(t_shift_block_list[m])+" of "+str(t_shift_block_list[-1])+" | iteration "+str(j)+" of "+str(len(t_shift_lists[m]))+" | t_shift = "+str(t_shift))
                        
                        """
                        tici = datetime.now()
                        analytic_sinusoid_flag = False
                        x1,z1,x2,z2,x3,z3,t_interp_half_step,ff_interp_half_step = rk.rk_solve(t,del_t,x10,z10,x20,z20,x30,z30,gamma1,gamma2,gamma3,\
                                                                                                      alpha1,alpha2,alpha3,beta1,beta2,beta3,C1,C2,C3,\
                                                                                                      1.0,1.0,1.0,kappa12,kappa13,kappa21,kappa23,kappa31,kappa32,\
                                                                                                      omega1,A1,phi1,omega2,A2,phi2,omega3,A3,phi3,\
                                                                                                      ff1_clip,ff2_clip,ff3_clip,t_ff13_clip,del_t_ff,\
                                                                                                      interp_type,analytic_sinusoid_flag)
                        toci = datetime.now()
                        print("R-K analytic ff solve execution time: ")
                        print(toci-tici)
                        
                        x1 = x1[::del_t_ratio]
                        x2 = x2[::del_t_ratio]
                        x3 = x3[::del_t_ratio]
                        
                        z1 = z1[::del_t_ratio]
                        z2 = z2[::del_t_ratio]
                        z3 = z3[::del_t_ratio]
                        
                        z1_spect = np.abs( np.fft.fft(z1[-1*int(args.spect_num)-1:-1]) )/(int(args.spect_num)/2)
                        z2_spect = np.abs( np.fft.fft(z2[-1*int(args.spect_num)-1:-1]) )/(int(args.spect_num)/2)
                        z3_spect = np.abs( np.fft.fft(z3[-1*int(args.spect_num)-1:-1]) )/(int(args.spect_num)/2)
                        
                        #if n==0 and i==0:
                        freq = np.fft.fftfreq(len(z2_spect),del_t_ff)
                        idx_ff_freq = (np.abs(freq - f)).argmin()
                        

                        z1_ff_freq = z1_spect[idx_ff_freq]
                        z2_ff_freq = z2_spect[idx_ff_freq]
                        z3_ff_freq = z3_spect[idx_ff_freq]

                        z1_ff_freq_list.append(z1_ff_freq)
                        z2_ff_freq_list.append(z2_ff_freq)
                        z3_ff_freq_list.append(z3_ff_freq)
                        
                        x1_list.append(x1)
                        x2_list.append(x2)
                        x3_list.append(x3)
                        z1_list.append(z1)
                        z2_list.append(z2)
                        z3_list.append(z3)
                        
                        toc = datetime.now()
                        print("Total elapsed time = "+str(toc-tic))
                    
                    if save_verbosity_index > 0:
                        tici = datetime.now()
                        if save_verbosity_index >= 1000:
                            ph.pickle_all(dir(),globals(),pkl_name)
                        elif save_verbosity_index >= 700:                            
                            var_list = ['t_shift_lists','m','n','amp_factor_lists','z1_ff_freq_list','z3_ff_freq_list','A2',\
                                'z1_list','z2_list','z3_list','x1_list','x2_list','x3_list','del_t_ff']
                            ph.pickle_var_list(var_list,globals(),pkl_name)
                        elif save_verbosity_index >= 500:                            
                            var_list = ['t_shift_lists','m','n','amp_factor_lists','z1_ff_freq_list','z3_ff_freq_list','A2',\
                                'z1_list','z2_list','z3_list','del_t_ff']
                            ph.pickle_var_list(var_list,globals(),pkl_name)
                        elif save_verbosity_index >= 400:                            
                            var_list = ['t_shift_lists','m','n','amp_factor_lists','z1_ff_freq_list','z3_ff_freq_list','A2',\
                                'z1_list','z3_list','del_t_ff']
                            ph.pickle_var_list(var_list,globals(),pkl_name)
                        else:
                            var_list = ['t_shift_lists','m','n','amp_factor_lists','z1_ff_freq_list','z3_ff_freq_list','A2']
                            ph.pickle_var_list(var_list,globals(),pkl_name)
                        print("\nA2 block "+str(amp_factor_block_list[n])+" ("+str(i)+") of "+str(amp_factor_block_list[-1])+", ")
                        print("\nt_shift block "+str(t_shift_block_list[m])+" of "+str(t_shift_block_list[-1])+" saved")
                        toci = datetime.now()
                        print("save time = "+str(toci-tici))
                else:
                    print('skipping '+pkl_name+' (already exists)')
"""

plt.show(block=False)

toc = datetime.now()
print("Execution time: ")
print(toc-tic)










