#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:07:20 2022

@author: gregg
"""

import os
import numpy as np
import sys
import time
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-rl','--min_run_index',default=1,type=int) 
parser.add_argument('-ru','--max_run_index',default=13,type=int)
parser.add_argument('-d','--data_folder_num',default=1,type=int)
parser.add_argument('-ts','--timestep',default=30,type=int)
parser.add_argument('-t1','--timeoffset1',default=0,type=int)
parser.add_argument('-t2','--timeoffset2',default=0,type=int)
parser.add_argument('-as','--a2_superblock_size',default=1,type=int)
parser.add_argument('-rf','--run_flag',default=1,type=int)
args = parser.parse_args()

min_run_index = int(args.min_run_index)
max_run_index = int(args.max_run_index)
data_folder_num = int(args.data_folder_num)
timestep = int(args.timestep)
timeoffset1 = int(args.timeoffset1)
timeoffset2 = int(args.timeoffset2)
A2_superblock_size = int(args.a2_superblock_size)
run_flag = bool(args.run_flag)

# Nomenclature: a block is a set of amplification factor values or time shift values that are saved in the same file. A superblock is the number of blocks that is run in each terminal tab (tabs are run in parallel)
num_t_steps = 10 # number of time samplings
t_shift_superblock_size = 10 # number of time steps in a superblock

phi1 = 0.0 # phase of the forcing sinusoid on oscillator 1
max_t_shift = 1000.0 # maximum time shift for time samplings. The shift for each time sampling is max_t_shift/num_t_steps
frequency = 15.625 # Frequency of the forcing sinusoids (Hz)
af_list = '0.1 0.5 1.0 1.5 2.0 3.0 4.0 5.0' # List of amplification factors to use. Entries separated by spaces.
spect_num = 100000 # number of data points to include when calculating the spectra to use in the detection coefficient calculation
save_verbosity_index = 500 # number between 0 and 1000 that determines how much data is saved with each run. Refer to run_all_sweeps.py to see what saves with each verbosity index level.
signal_mag = 0.2 # sinusoidal signal amplitude applied to oscillator 2
input_filename = '../weak_signals/sensor_data/prepped_data/flights1_mod.csv' # string to specify csv file for noise
pulse_filename = '../weak_signals/sensor_data/prepped_data/pulse1.csv' # string to specify file for pulse time series (optional)
total_t = 500.0 # Total time to run simulation, for each sampling
x10 = 0.028998442540967034 # initial condition, oscillator 1
z10 = -84.42093768595821 # initial condition, time derivative of oscillator 1
x20 = -0.169103092562322 # initial condition, oscillator 2
z20 = 5.6747251829963075# initial condition, time derivative of oscillator 2
x30 = 0.10030426818849783 # initial condition, oscillator 3
z30 = 77.7816827284428# initial condition, time derivative of oscillator 3
t_sig_start = 0.0 # If >0, the sinusoidal signal will start at this time relative to the beginning of the sampling
noise_flag = True # Whether or not to include noise
gamma1 = 0.7 # damping term on oscillator 1, gamma3 is set equal to gamma1
gamma2 = 1.1 # damping term on oscillator 2
alpha1 = 1e4 # magnitude of sinusoidal term in equation for oscillator 1. The corresponding term in oscillator 3 is set equal to this.
alpha2 = 1e4 # magnitude of sinusoidal term in equation for oscillator 2.
beta1 = 0 # magnitude of the cubic term in oscillator 1. This term is set equal to this for all oscillators. 
C1 = 1.0 # Constant term in the forcing function (set equally for all oscillators)
kappa12 = 300 # coupling term between oscillators 1 and 2 (set equal for term between oscillators 2 and 3)
A1 = 1000.0 # Amplitude of forcing sinusoid on oscillator 1 (set equal for oscillator 3)
sig_repeat_timestep_factor = 1 # rarely used feature. If >1, overrides frequency for signal on oscillator 2, giving it a frequency with a period of this many time steps.
sensor_increment = 0 # If >0, switches to a different column in the input noise file for each time sampling (name assumes columns represent sensors)
phi3 = np.pi # phase of forcing sinusoid on oscillator 3
del_t_ff = 0.004 # time step for noise data (ff stands for forcing function)
Cff1 = 0 # Constant multiplier for noise on oscillator 1 (normally zero)
Cff2 = 1 # Constant multiplier for noise on oscillator 2 (normally one)
Cff3 = 0 # Constant multiplier for noise on oscillator 3 (normally zero)
t_offset1 = 0 # If >0, time starts at this location in the noise file for noise on oscillator 1.
t_offset2 = 0 # If >0, time starts at this location in the noise file for noise on oscillator 2.
t_offset3 = 0 # If >0, time starts at this location in the noise file for noise on oscillator 3.
pulse1_height = 0 # amplitude of pulse to apply to oscillator 1. Normally 0. Generally, a pulse is only applied to oscillator 2.
pulse1_type = 1 # index in range 0-5 to choose the type of pulse to apply to oscillator 1. 0 = Gaussian pulse. 1-3 = mathematical functions intended to mimic magnetic pulses (see run_all_sweeps.py). 4 = wave packet, namely sinusoid*Gaussian pulse. 5 = load pulse from file. 
pulse1_width = 10/(2*np.sqrt(2*np.log(2))) # pulse width on oscillator 1. If Gaussian, this is sigma (standard deviation). If loading a pulse from a file, this is ignored.
pulse1_center = 300 # Time to apply pulse to oscillator 1 (center point of pulse).
pulse2_height = 10 # amplitude of pulse to apply to oscillator 2. 
pulse2_type = 1 # index in range 0-5 to choose the type of pulse to apply to oscillator 2. 0 = Gaussian pulse. 1-3 = mathematical functions intended to mimic magnetic pulses (see run_all_sweeps.py). 4 = wave packet, namely sinusoid*Gaussian pulse. 5 = load pulse from file. 
pulse2_width = 5/(2*np.sqrt(2*np.log(2))) # pulse width on oscillator 2. If Gaussian, this is sigma (standard deviation). If loading a pulse from a file, this is ignored.
pulse2_center = 300 # Time to apply pulse to oscillator 2 (center point of pulse).
pulse3_height = 0 # amplitude of pulse to apply to oscillator 3. Normally 0. Generally, a pulse is only applied to oscillator 2.
pulse3_type = 1 # index in range 0-5 to choose the type of pulse to apply to oscillator 3. 0 = Gaussian pulse. 1-3 = mathematical functions intended to mimic magnetic pulses (see run_all_sweeps.py). 4 = wave packet, namely sinusoid*Gaussian pulse. 5 = load pulse from file. 
pulse3_width = 10/(2*np.sqrt(2*np.log(2))) # pulse width on oscillator 3. If Gaussian, this is sigma (standard deviation). If loading a pulse from a file, this is ignored.
pulse3_center = 30 # Time to apply pulse to oscillator 3 (center point of pulse).
abs_pulse_location_flag = 0 # If 0, one pulse is applied for every time sampling. The timing of pulse*_center is relative to the beginning of the sampling. If 1, one pulse is applied for all time. The timing of pulse*_center is relative to the beginning of the noise file.
snap_flag = 1 # If 1, the beginning of each time sampling will snap to the nearest multiple of 1/frequency. This guaruntees the phase remains the same between samplings.

if data_folder_num == 1:
    input_filename = './Gaussian_noise_4-8-26.csv'
    pulse_filename = 'None'
    A1 = 1e4
    total_t = 20.0
    signal_mag = 0
    af_list = '1 1000 10000 30000 50000 70000'
    pulse1_height = 0
    pulse1_type = 0
    pulse1_center = 0
    pulse2_height = 1.0
    pulse2_type = 0
    pulse2_center = 45
    pulse3_height = 0
    pulse3_type = 0
    spect_num = 1250
    Cff1 = 0
    Cff2 = 1
    Cff3 = 0
    t_offset2 = 0
    max_t_shift = 50
    num_t_steps = 10
    t_shift_superblock_size = 10
    x10 = -0.35855048462514466
    z10 = 107.86692450322923
    x20 = 0.00010210034673662672
    z20 = -0.00010738032412920047
    x30 = 0.358839140293869
    z30 = -107.86400005602802
    frequency = 0.2
    alpha1 = 0.1
    alpha2 = 1.33
    kappa12 = 0.14
    gamma1 = 1.5
    gamma2 = 0.75
    abs_pulse_location_flag = 1
    snap_flag = 1


#A2_block_list = list(range(0,len(af_list.split(' ')),A2_superblock_size))
af_len = len([float(x) for x in af_list.split(' ')])
A2_block_list = list(range(0,af_len,A2_superblock_size))

t_shift_block_list = list(range(num_t_steps))

num_t_shift_superblocks = int(np.ceil(len(t_shift_block_list)/t_shift_superblock_size))
t_shift_superblock_lists = []
for i in range(num_t_shift_superblocks):
    t_shift_superblock_list = list(range(i*t_shift_superblock_size,(i+1)*t_shift_superblock_size))
    t_shift_superblock_lists.append(t_shift_superblock_list)
    
timestep_list = list(range(0, 2*len(A2_block_list)*len(t_shift_superblock_lists)*timestep,timestep))
random.shuffle(timestep_list)

if not os.path.isdir('./data/data'+str(data_folder_num)):
    os.system('mkdir data/data'+str(data_folder_num))

ind=0
for i in range(len(A2_block_list)):
    for j in range(len(t_shift_superblock_lists)):
        command_string = ''
        command_string += 'gnome-terminal -x sh -c "'
        command_string += 'sleep '+str(timestep_list[ind]+timeoffset1)+'s; '
        command_string += 'python run_all_sweeps.py '
        command_string += '--af_block_lim '+str(A2_block_list[i])+' '+str(A2_block_list[i]+A2_superblock_size)+' '
        command_string += '--t_shift_block_lim '+str(t_shift_superblock_lists[j][0])+' '+str(int(t_shift_superblock_lists[j][-1])+1)+' '
        command_string += '--num_t_shift_steps '+str(len(t_shift_block_list))+' '
        command_string += '--phi1 '+str(phi1)+' '
        command_string += '--signal_mag 0 '
        command_string += '--min_run_index '+str(min_run_index)+' '
        command_string += '--max_run_index '+str(max_run_index)+' '
        command_string += '--data_folder_num '+str(data_folder_num)+' '
        command_string += '--max_t_shift '+str(max_t_shift)+' '
        command_string += '--frequency '+str(frequency)+' '
        command_string += '--af_list '+af_list+' '
        command_string += '--spect_num '+str(spect_num)+' '
        command_string += '--save_verbosity_index '+str(save_verbosity_index)+' '
        command_string += '--input_filename '+input_filename+' '
        command_string += '--pulse_filename '+pulse_filename+' '
        command_string += '--total_t '+str(total_t)+' '
        command_string += '--x10 '+str(x10)+' '
        command_string += '--z10 '+str(z10)+' '
        command_string += '--x20 '+str(x20)+' '
        command_string += '--z20 '+str(z20)+' '
        command_string += '--x30 '+str(x30)+' '
        command_string += '--z30 '+str(z30)+' '
        command_string += '--t_sig_start '+str(t_sig_start)+' '
        if not noise_flag:
            command_string += '--noise_flag '
        command_string += '--A1 '+str(A1)+' '
        command_string += '--pulse1_width '+str(pulse1_width)+' '
        command_string += '--pulse1_center '+str(pulse1_center)+' '
        command_string += '--pulse1_height 0 '
        command_string += '--pulse1_type '+str(pulse1_type)+' '
        command_string += '--pulse2_width '+str(pulse2_width)+' '
        command_string += '--pulse2_center '+str(pulse2_center)+' '
        command_string += '--pulse2_height 0 '
        command_string += '--pulse2_type '+str(pulse2_type)+' '
        command_string += '--pulse3_width '+str(pulse3_width)+' '
        command_string += '--pulse3_center '+str(pulse3_center)+' '
        command_string += '--pulse3_height 0 '
        command_string += '--pulse3_type '+str(pulse3_type)+' '
        command_string += '--sensor_increment '+str(sensor_increment)+' '
        command_string += '--phi3 '+str(phi3)+' '
        command_string += '--del_t_ff '+str(del_t_ff)+' '
        command_string += '--Cff1 '+str(Cff1)+' '
        command_string += '--Cff2 '+str(Cff2)+' '
        command_string += '--Cff3 '+str(Cff3)+' '
        command_string += '--t_offset1 '+str(t_offset1)+' '
        command_string += '--t_offset2 '+str(t_offset2)+' '
        command_string += '--t_offset3 '+str(t_offset3)+' '
        command_string += '--gamma1 '+str(gamma1)+' '
        command_string += '--gamma2 '+str(gamma2)+' '
        command_string += '--alpha1 '+str(alpha1)+' '
        command_string += '--alpha2 '+str(alpha2)+' '
        command_string += '--beta1 '+str(beta1)+' '
        command_string += '--C1 '+str(C1)+' '
        command_string += '--kappa12 '+str(kappa12)+' '
        command_string += '--abs_pulse_location_flag '+str(abs_pulse_location_flag)+' '
        command_string += '--snap_flag '+str(snap_flag)+' '
        
        command_string += ' | tee ./data/data'+str(data_folder_num)+'/logfile_all_runs_'+str(A2_block_list[i])+\
                      '_'+str(t_shift_superblock_lists[j][0])+'_'+str(t_shift_superblock_lists[j][-1])+'.txt"'
        
        if run_flag:
            os.system(command_string)
        else:
            print(str(ind+1))
            print(command_string)
            print("\n")
        ind += 1
        
for i in range(len(A2_block_list)):
    for j in range(len(t_shift_superblock_lists)):
        command_string = ''
        command_string += 'gnome-terminal -x sh -c "'
        command_string += 'sleep '+str(timestep_list[ind]+timeoffset1)+'s; '
        command_string += 'python run_all_sweeps.py '
        command_string += '--af_block_lim '+str(A2_block_list[i])+' '+str(A2_block_list[i]+A2_superblock_size)+' '
        command_string += '--t_shift_block_lim '+str(t_shift_superblock_lists[j][0])+' '+str(int(t_shift_superblock_lists[j][-1])+1)+' '
        command_string += '--num_t_shift_steps '+str(len(t_shift_block_list))+' '
        command_string += '--phi1 '+str(phi1)+' '
        command_string += '--signal_mag '+str(signal_mag)+' '
        command_string += '--min_run_index '+str(min_run_index)+' '
        command_string += '--max_run_index '+str(max_run_index)+' '
        command_string += '--data_folder_num '+str(data_folder_num)+' '
        command_string += '--max_t_shift '+str(max_t_shift)+' '
        command_string += '--frequency '+str(frequency)+' '
        command_string += '--af_list '+af_list+' '
        command_string += '--spect_num '+str(spect_num)+' '
        command_string += '--save_verbosity_index '+str(save_verbosity_index)+' '
        command_string += '--input_filename '+input_filename+' '
        command_string += '--pulse_filename '+pulse_filename+' '
        command_string += '--total_t '+str(total_t)+' '
        command_string += '--x10 '+str(x10)+' '
        command_string += '--z10 '+str(z10)+' '
        command_string += '--x20 '+str(x20)+' '
        command_string += '--z20 '+str(z20)+' '
        command_string += '--x30 '+str(x30)+' '
        command_string += '--z30 '+str(z30)+' '
        command_string += '--t_sig_start '+str(t_sig_start)+' '
        if not noise_flag:
            command_string += '--noise_flag '
        command_string += '--A1 '+str(A1)+' '
        command_string += '--pulse1_width '+str(pulse1_width)+' '
        command_string += '--pulse1_center '+str(pulse1_center)+' '
        command_string += '--pulse1_height '+str(pulse1_height)+' '
        command_string += '--pulse1_type '+str(pulse1_type)+' '
        command_string += '--pulse2_width '+str(pulse2_width)+' '
        command_string += '--pulse2_center '+str(pulse2_center)+' '
        command_string += '--pulse2_height '+str(pulse2_height)+' '
        command_string += '--pulse2_type '+str(pulse2_type)+' '
        command_string += '--pulse3_width '+str(pulse3_width)+' '
        command_string += '--pulse3_center '+str(pulse3_center)+' '
        command_string += '--pulse3_height '+str(pulse3_height)+' '
        command_string += '--pulse3_type '+str(pulse3_type)+' '
        command_string += '--sensor_increment '+str(sensor_increment)+' '
        command_string += '--phi3 '+str(phi3)+' '
        command_string += '--del_t_ff '+str(del_t_ff)+' '
        command_string += '--Cff1 '+str(Cff1)+' '
        command_string += '--Cff2 '+str(Cff2)+' '
        command_string += '--Cff3 '+str(Cff3)+' '
        command_string += '--t_offset1 '+str(t_offset1)+' '
        command_string += '--t_offset2 '+str(t_offset2)+' '
        command_string += '--t_offset3 '+str(t_offset3)+' '
        command_string += '--gamma1 '+str(gamma1)+' '
        command_string += '--gamma2 '+str(gamma2)+' '
        command_string += '--alpha1 '+str(alpha1)+' '
        command_string += '--alpha2 '+str(alpha2)+' '
        command_string += '--beta1 '+str(beta1)+' '
        command_string += '--C1 '+str(C1)+' '
        command_string += '--kappa12 '+str(kappa12)+' '
        command_string += '--abs_pulse_location_flag '+str(abs_pulse_location_flag)+' '
        command_string += '--snap_flag '+str(snap_flag)+' '
        
        command_string += ' | tee ./data/data'+str(data_folder_num)+'/logfile_all_runs_sig_'+str(A2_block_list[i])+\
                      '_'+str(t_shift_superblock_lists[j][0])+'_'+str(t_shift_superblock_lists[j][-1])+'.txt"'
        
        if run_flag:
            os.system(command_string)
        else:
            print(str(ind+1))
            print(command_string)
            print("\n")
        ind += 1
            
            
            
            
            
            
