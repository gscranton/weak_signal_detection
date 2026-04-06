"""
exec(open("sweep_analysis_multirun.py").read())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Thu Dec 22 08:48:30 2022

@author: gregg

command example
python sweep_analysis_multirun.py -p -sp -srd -d 45 -r 1 2 3 4 5 6 7 8 9 10 11 12
python sweep_analysis_multirun.py -p -sp -srd -d 47 -r 1 2 3 4 5 6
python sweep_analysis_multirun.py -p -sp -srd -d 43 -r 7 8 9 10 11 12
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import re
sys.path.insert(0, './helper_libs')
import pickle_helpers as ph
from datetime import datetime
import contract_list as cl
import argparse
import os
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_folder_num',default=38,type=int) #22
parser.add_argument('-r','--run_nums',nargs='+',default=['1','2','3'],type=int)
parser.add_argument('-p','--plot_block',action='store_true')
parser.add_argument('-snf','--custom_samp_num_flag',action='store_true')
parser.add_argument('-sn','--samp_num',default=5200,type=int)
parser.add_argument('-ss','--samp_start',default=69800,type=int) # 120000
parser.add_argument('-f','--frequency',default=0.2,type=float) # 15.0
parser.add_argument('-dt','--del_t_ff',default=0.004,type=float)
parser.add_argument('-ds','--downsample_factor',default=1,type=int)
parser.add_argument('-pmx','--plt_max',default=1000,type=int)
parser.add_argument('-pmn','--plt_min',default=0,type=int)
parser.add_argument('-ps','--plot_spectra_flag',action='store_true')
parser.add_argument('-pss','--plot_spectra_with_signal_flag',action='store_true')
parser.add_argument('-sp','--save_plots_flag',action='store_true')
parser.add_argument('-x2m','--x2_marker_flag',action='store_true')
parser.add_argument('-x2f','--x2_frequency',default=0.2)
parser.add_argument('-rf','--root_folder',default="./") # "./" or "Z:\gr075391\A2_sweep_2\\"
parser.add_argument('-srd','--save_result_data',action='store_true')
parser.add_argument('-tel','--t0_num_exclude_list',nargs='+',default='None')
#parser.add_argument('-eut','--exclude_upper_threshold',default=0)
parser.set_defaults(plot_block=False,custom_samp_num_flag=False,plot_spectra_flag=False,\
    plot_spectra_with_signal_flag=False,x2_marker_flag=False,save_result_data=False,save_plots_flag=False) # custom_samp_num_flag=False x2_marker_flag=False
args = parser.parse_args()

run_nums = np.array([int(x) for x in args.run_nums])
data_folder_num = int(args.data_folder_num)
custom_samp_num_flag = bool(args.custom_samp_num_flag)
samp_num = int(args.samp_num)
samp_start = int(args.samp_start)
frequency = float(args.frequency)
del_t_ff = float(args.del_t_ff)
plot_block = bool(args.plot_block)
downsample_factor = int(args.downsample_factor)
plt_max = int(args.plt_max)
plt_min = int(args.plt_min)
save_plots_flag = bool(args.save_plots_flag)
x2_marker_flag = bool(args.x2_marker_flag)
x2_frequency = float(args.x2_frequency)
root_folder = args.root_folder
plot_spectra_flag = bool(args.plot_spectra_flag)
plot_spectra_with_signal_flag = bool(args.plot_spectra_with_signal_flag)
save_result_data = bool(args.save_result_data)
t0_num_exclude_list = args.t0_num_exclude_list
#exclude_upper_threshold = args.exclude_upper_threshold
if t0_num_exclude_list != 'None':
    t0_num_exclude_list = np.array([int(x) for x in t0_num_exclude_list])

if root_folder[-1] != '_':
    root_folder_seg_num = len(re.split('_',root_folder))-1
else:
    root_folder_seg_num = len(re.split('_',root_folder))

tic = datetime.now()

file_list = []
for run_num in run_nums:
    load_prefix = root_folder+"data/data"+str(data_folder_num)+"/sweep_run"+str(run_num)+"_sig_amp"
    file_list += glob.glob(load_prefix+"*.pkl")

if t0_num_exclude_list != 'None':
    for i in range(len(file_list)-1,-1,-1):
        temp = int(re.split('\.',re.split('_',file_list[i])[root_folder_seg_num+11])[0])
        if temp in t0_num_exclude_list:
            file_list.pop(i)

amp_factor_num_list = []
t0_num_list = []
for i in range(len(file_list)):
    temp = int(re.split('\.',re.split('_',file_list[i])[root_folder_seg_num+6])[0])
    amp_factor_num_list.append(temp)
    temp = int(re.split('\.',re.split('_',file_list[i])[root_folder_seg_num+11])[0])
    t0_num_list.append(temp)

num_t0 = np.max(t0_num_list)+1
total_num_list = list( num_t0*np.array(amp_factor_num_list) + np.array(t0_num_list) )

amp_factor_num_list = np.array([x for _,x in sorted(zip(total_num_list,amp_factor_num_list))])
t0_num_list = np.array([x for _,x in sorted(zip(total_num_list,t0_num_list))])
file_list = np.array([x for _,x in sorted(zip(total_num_list,file_list))])
total_num_list = np.array(sorted(total_num_list))

if downsample_factor > 1:
    #file_list = file_list[::downsample_factor]
    #amp_factor_num_list = amp_factor_num_list[::downsample_factor]
    #t0_num_list = t0_num_list[::downsample_factor]
    #total_num_list = total_num_list[::downsample_factor]
    file_list = file_list[(np.array(t0_num_list)%downsample_factor == 0)]
    amp_factor_num_list = amp_factor_num_list[(np.array(t0_num_list)%downsample_factor == 0)]
    total_num_list = total_num_list[(np.array(t0_num_list)%downsample_factor == 0)]
    t0_num_list = t0_num_list[(np.array(t0_num_list)%downsample_factor == 0)]

amp_factor_list = []
t0_list = []
z1_ff_freq_list1 = []
z3_ff_freq_list1 = []
x13_diff_list = []
A2_list = []
z1_spect1_list = []
z2_spect1_list = []
z3_spect1_list = []
z2_x2_freq_list1 = []
z1_sig_list = []
z2_sig_list = []
z3_sig_list = []

for i in range(len(file_list)):
    try:
        d = (ph.load_pickle(file_list[i]))
    except:
        print("could not load "+file_list[i])
    
    if "t0_lists" in d:
        t0_list += d["t0_lists"][d["m"]].tolist()
    else:
        t0_list += d["t_shift_lists"][d["m"]].tolist()
    
    
    if custom_samp_num_flag:
        for j in range(len(d['z1_list'])):
            z1_temp = d['z1_list'][j]
            z1_sig_list.append(z1_temp)
            spect = np.fft.fft(z1_temp[samp_start:samp_start+samp_num]) /(samp_num/2)
            freq = np.fft.fftfreq(len(spect),del_t_ff)
            idx_ff_freq = (np.abs(freq - frequency)).argmin()
            z1_ff_freq_list1 += [spect[idx_ff_freq]]
            z1_spect1_list.append(spect)
            
            z2_temp = d['z2_list'][j]
            z2_sig_list.append(z2_temp)
            spect = np.fft.fft(z2_temp[samp_start:samp_start+samp_num]) /(samp_num/2)
            freq = np.fft.fftfreq(len(spect),del_t_ff)
            z2_spect1_list.append(spect)
            idx_x2_freq = (np.abs(freq - x2_frequency)).argmin()
            #idx_x2_freq = np.abs(spect[freq>0]).argmax()
            z2_x2_freq_list1 += [spect[idx_x2_freq]]
            
            z3_temp = d['z3_list'][j]
            z3_sig_list.append(z3_temp)
            spect = np.fft.fft(z3_temp[samp_start:samp_start+samp_num]) /(samp_num/2)
            freq = np.fft.fftfreq(len(spect),del_t_ff)
            idx_ff_freq = (np.abs(freq - frequency)).argmin()
            z3_ff_freq_list1 += [spect[idx_ff_freq]]
            z3_spect1_list.append(spect)
    else:
        z1_ff_freq_list1 += [(d["z1_ff_freq_list"])]
        z3_ff_freq_list1 += [(d["z3_ff_freq_list"])]
    
    amp_factor_list += d["amp_factor_lists"][d["n"]].tolist()
    

    A2_list += [d["A2"]]
    #A2_list += [d["A2"][0]]
    
if custom_samp_num_flag:
    print("spect len = "+str(len(spect)))
    print("Frequency = "+str(freq[idx_ff_freq]))
    print("Frequency error = "+str(np.abs(freq[idx_ff_freq]-frequency)))

A2_list = list(set(list(A2_list)))
A2_list.sort()

amp_factor_unique_list = cl.cl(amp_factor_list) #list(set(amp_factor_list))
amp_factor_num_unique_list = cl.cl(amp_factor_num_list) #list(set(amp_factor_num_list))
amp_factor_num_unique_list = [x for _,x in sorted(zip(amp_factor_unique_list,amp_factor_num_unique_list))]
amp_factor_unique_list.sort()

parameter_avg_list1 = []
parameter_list1 = np.array([])
for i in range(len(amp_factor_num_unique_list)):
    inds_cur = [j for j,x in enumerate(amp_factor_num_list) if x==amp_factor_num_unique_list[i]]
    z1_temp = np.array( [z1_ff_freq_list1[k] for k in inds_cur] )
    z3_temp = np.array( [z3_ff_freq_list1[k] for k in inds_cur] )
    for k in inds_cur:
        parameter_list1 = np.append(parameter_list1, np.abs(z1_ff_freq_list1[k]) - np.abs(z3_ff_freq_list1[k]))
    temp = np.mean( np.abs(z1_temp) - np.abs(z3_temp) )
    parameter_avg_list1.append(temp)

file_list = []
for run_num in run_nums:
    load_prefix = root_folder+"data/data"+str(data_folder_num)+"/sweep_run"+str(run_num)+"_amp"
    file_list += glob.glob(load_prefix+"*.pkl")

if t0_num_exclude_list != 'None':
    for i in range(len(file_list)-1,-1,-1):
        temp = int(re.split('\.',re.split('_',file_list[i])[root_folder_seg_num+10])[0])
        if temp in t0_num_exclude_list:
            file_list.pop(i)

amp_factor_num_list = []
t0_num_list = []
for i in range(len(file_list)):
    temp = int(re.split('\.',re.split('_',file_list[i])[root_folder_seg_num+5])[0])
    amp_factor_num_list.append(temp)
    temp = int(re.split('\.',re.split('_',file_list[i])[root_folder_seg_num+10])[0])
    t0_num_list.append(temp)

num_t0 = np.max(t0_num_list)+1
total_num_list = list( num_t0*np.array(amp_factor_num_list) + np.array(t0_num_list) )

amp_factor_num_list = np.array([x for _,x in sorted(zip(total_num_list,amp_factor_num_list))])
t0_num_list = np.array([x for _,x in sorted(zip(total_num_list,t0_num_list))])
file_list = np.array([x for _,x in sorted(zip(total_num_list,file_list))])
total_num_list = np.array(sorted(total_num_list))

if downsample_factor > 1:
    file_list = file_list[(np.array(t0_num_list)%downsample_factor == 0)]
    amp_factor_num_list = amp_factor_num_list[(np.array(t0_num_list)%downsample_factor == 0)]
    total_num_list = total_num_list[(np.array(t0_num_list)%downsample_factor == 0)]
    t0_num_list = t0_num_list[(np.array(t0_num_list)%downsample_factor == 0)]

amp_factor_list = []
t0_list = []
z1_ff_freq_list0 = []
z3_ff_freq_list0 = []
x13_diff_list = []
#ff_list = []
z1_spect0_list = []
z2_spect0_list = []
z3_spect0_list = []
z2_x2_freq_list0 = []
z1_list = []
z2_list = []
z3_list = []

for i in range(len(file_list)):
    try:
        d = (ph.load_pickle(file_list[i]))
    except:
        print("could not load "+file_list[i])
    
    if "t0_lists" in d:
        t0_list += d["t0_lists"][d["m"]].tolist()
    else:
        t0_list += d["t_shift_lists"][d["m"]].tolist()
    amp_factor_list += d["amp_factor_lists"][d["n"]].tolist()
    
    if custom_samp_num_flag:
        for j in range(len(d['z1_list'])):
            z1_temp = d['z1_list'][j]
            z1_list.append(z1_temp)
            spect = np.fft.fft(z1_temp[samp_start:samp_start+samp_num]) /(samp_num/2)
            freq = np.fft.fftfreq(len(spect),del_t_ff)
            idx_ff_freq = (np.abs(freq - frequency)).argmin()
            z1_ff_freq_list0 += [spect[idx_ff_freq]]
            z1_spect0_list.append(spect)
            
            z2_temp = d['z2_list'][j]
            z2_list.append(z2_temp)
            spect = np.fft.fft(z2_temp[samp_start:samp_start+samp_num]) /(samp_num/2)
            freq = np.fft.fftfreq(len(spect),del_t_ff)
            z2_spect0_list.append(spect)
            idx_x2_freq = (np.abs(freq - x2_frequency)).argmin()
            #idx_x2_freq = np.abs(spect[freq>0]).argmax()
            z2_x2_freq_list0 += [spect[idx_x2_freq]]
            
            z3_temp = d['z3_list'][j]
            z3_list.append(z3_temp)
            spect = np.fft.fft(z3_temp[samp_start:samp_start+samp_num]) /(samp_num/2)
            freq = np.fft.fftfreq(len(spect),del_t_ff)
            idx_ff_freq = (np.abs(freq - frequency)).argmin()
            z3_ff_freq_list0 += [spect[idx_ff_freq]]
            z3_spect0_list.append(spect)
    else:
        z1_ff_freq_list0 += [(d["z1_ff_freq_list"])]
        z3_ff_freq_list0 += [(d["z3_ff_freq_list"])]
    
amp_factor_unique_list = cl.cl(amp_factor_list) #list(set(amp_factor_list))
amp_factor_num_unique_list = cl.cl(amp_factor_num_list) #list(set(amp_factor_num_list))
amp_factor_num_unique_list = [x for _,x in sorted(zip(amp_factor_unique_list,amp_factor_num_unique_list))]
amp_factor_unique_list.sort()

parameter_avg_list0 = []
parameter_list0 = np.array([])
for i in range(len(amp_factor_num_unique_list)):
    inds_cur = [j for j,x in enumerate(amp_factor_num_list) if x==amp_factor_num_unique_list[i]]
    #inds_cur = inds_cur[0:-2] #DEBUG
    z1_temp = np.array( [z1_ff_freq_list0[k] for k in inds_cur] )
    z3_temp = np.array( [z3_ff_freq_list0[k] for k in inds_cur] )
    for k in inds_cur:
        parameter_list0 = np.append(parameter_list0, np.abs(z1_ff_freq_list0[k]) - np.abs(z3_ff_freq_list0[k]))
    temp = np.mean( np.abs(z1_temp) - np.abs(z3_temp) )
    parameter_avg_list0.append(temp)

if x2_marker_flag:
    x2_parameter_avg_list = []
    x2_parameter_avg_list0 = []
    x2_parameter_avg_list1 = []
    x2_parameter_list = np.array([])
    x2_parameter_list0 = np.array([])
    x2_parameter_list1 = np.array([])
    for i in range(len(amp_factor_num_unique_list)):
        inds_cur = [j for j,x in enumerate(amp_factor_num_list) if x==amp_factor_num_unique_list[i]]
        z2_temp = np.array( [z2_x2_freq_list0[k] for k in inds_cur] )
        z2_sig_temp = np.array( [z2_x2_freq_list1[k] for k in inds_cur] )
        for k in inds_cur:
            x2_parameter_list = np.append(x2_parameter_list, np.abs(z2_x2_freq_list1[k]) - np.abs(z2_x2_freq_list0[k]))
            x2_parameter_list0 = np.append(x2_parameter_list0, np.abs(z2_x2_freq_list0[k]))
            x2_parameter_list1 = np.append(x2_parameter_list1, np.abs(z2_x2_freq_list1[k]))
        temp = np.mean( np.abs(z2_sig_temp) - np.abs(z2_temp) )
        x2_parameter_avg_list.append(temp)
        x2_parameter_avg_list0.append(np.mean(np.abs(z2_temp)))
        x2_parameter_avg_list1.append(np.mean(np.abs(z2_sig_temp)))

pltlen = np.min([len(amp_factor_unique_list),len(parameter_avg_list0),len(parameter_avg_list1)])
if plt_max > pltlen:
    plt_max = pltlen

pltcolors = ['tab:blue','tab:orange','tab:green']
plt.figure()
plt.plot(amp_factor_unique_list[plt_min:plt_max],parameter_avg_list0[plt_min:plt_max],color=pltcolors[0])
plt.plot(amp_factor_unique_list[plt_min:plt_max],parameter_avg_list1[plt_min:plt_max],color=pltcolors[1])
plt.plot(amp_factor_unique_list[plt_min:plt_max],parameter_avg_list0[plt_min:plt_max],'o',color=pltcolors[0])
plt.plot(amp_factor_unique_list[plt_min:plt_max],parameter_avg_list1[plt_min:plt_max],'o',color=pltcolors[1])
plt.grid(visible=True,which="both",axis="both")
plt.xlabel("Scaling Factor")
plt.ylabel("Detection Coefficient")
#plt.ylabel("x1-x3 at driving frequency")
plt.legend(["noise","noise+signal"])
#plt.xlim([plt_min-0.25,plt_max+0.25]) # TEMPORARY

fig_dir_top = 'figs'
fig_data_dir = '/figs_data'+str(data_folder_num)
fig_run_dir = '/figs_run'+"_".join([str(x) for x in run_nums])
if not os.path.isdir(fig_dir_top):
    os.system('mkdir '+fig_dir_top)
if not os.path.isdir(fig_dir_top+fig_data_dir):
    os.system('mkdir '+fig_dir_top+fig_data_dir)
if not os.path.isdir(fig_dir_top+fig_data_dir+fig_run_dir):
    os.system('mkdir '+fig_dir_top+fig_data_dir+fig_run_dir)
fig_dir = fig_dir_top+fig_data_dir+fig_run_dir+'/'
if save_plots_flag:
    plt.savefig(fig_dir+'detection_coeff_avg.png',bbox_inches='tight')

ratio = int(round(len(parameter_list0)/len(parameter_avg_list0)))
plt2_min = plt_min*ratio
plt2_max = plt_max*ratio

plt.figure()
plt.plot(parameter_list0[plt2_min:plt2_max],color=pltcolors[0])
plt.plot(parameter_list1[plt2_min:plt2_max],color=pltcolors[1])
plt.plot(parameter_list0[plt2_min:plt2_max],'x',color=pltcolors[0])
plt.plot(parameter_list1[plt2_min:plt2_max],'x',color=pltcolors[1])
plt.grid(visible=True,which="both",axis="both")
plt.xticks(list(np.linspace(0,len(parameter_list0[plt2_min:plt2_max]),(len(amp_factor_unique_list[plt_min:plt_max])+1))))
plt.legend(["noise","noise+signal"])
plt.xlabel("Data point number")
plt.ylabel("Detection Coefficient (without averaging)")
if save_plots_flag:
    plt.savefig(fig_dir+'detection_coeff_all.png',bbox_inches='tight')

if save_result_data:
    hf = h5py.File(fig_dir+'/figs_run'+str(data_folder_num)+'_results.hdf','w')
    hf.create_dataset('amp_factor_unique_list',data=amp_factor_unique_list,compression='gzip',compression_opts=9)
    hf.create_dataset('parameter_avg_list0',data=parameter_avg_list0,compression='gzip',compression_opts=9)
    hf.create_dataset('parameter_avg_list'+str(data_folder_num),data=parameter_avg_list1,compression='gzip',compression_opts=9)
    hf.create_dataset('parameter_list0',data=parameter_list0,compression='gzip',compression_opts=9)
    hf.create_dataset('parameter_list'+str(data_folder_num),data=parameter_list1,compression='gzip',compression_opts=9)
    if x2_marker_flag:
        hf.create_dataset('x2_parameter_avg_list0',data=x2_parameter_avg_list0,compression='gzip',compression_opts=9)
        hf.create_dataset('x2_parameter_avg_list'+str(data_folder_num),data=x2_parameter_avg_list1,compression='gzip',compression_opts=9)
        hf.create_dataset('x2_parameter_list0',data=x2_parameter_list0,compression='gzip',compression_opts=9)
        hf.create_dataset('x2_parameter_list'+str(data_folder_num),data=x2_parameter_list1,compression='gzip',compression_opts=9)
    hf.close()

if x2_marker_flag:
    if False:
        plt.figure()
        plt.plot(amp_factor_unique_list[plt_min:plt_max],x2_parameter_avg_list[plt_min:plt_max],color=pltcolors[0])
        plt.plot(amp_factor_unique_list[plt_min:plt_max],x2_parameter_avg_list[plt_min:plt_max],'o',color=pltcolors[0])
        plt.grid(visible=True,which="both",axis="both")
        plt.xlabel("Scaling Factor")
        plt.ylabel("x2 Detection Coefficient difference")

        plt.figure()
        plt.plot(x2_parameter_list[plt2_min:plt2_max],color=pltcolors[0])
        plt.plot(x2_parameter_list[plt2_min:plt2_max],'x',color=pltcolors[0])
        plt.grid(visible=True,which="both",axis="both")
        plt.xlabel("data point number")
        plt.ylabel("x2 Detection Coefficient difference (without averaging)")
    
    plt.figure()
    plt.plot(amp_factor_unique_list[plt_min:plt_max],x2_parameter_avg_list0[plt_min:plt_max],color=pltcolors[0])
    plt.plot(amp_factor_unique_list[plt_min:plt_max],x2_parameter_avg_list1[plt_min:plt_max],color=pltcolors[1])
    plt.plot(amp_factor_unique_list[plt_min:plt_max],x2_parameter_avg_list0[plt_min:plt_max],'o',color=pltcolors[0])
    plt.plot(amp_factor_unique_list[plt_min:plt_max],x2_parameter_avg_list1[plt_min:plt_max],'o',color=pltcolors[1])
    plt.grid(visible=True,which="both",axis="both")
    plt.xlabel("Scaling Factor")
    plt.ylabel("x2 Detection Coefficient")
    plt.legend(["noise","noise+signal"])
    if save_plots_flag:
        plt.savefig(fig_dir+'x2_detection_coeff.png',bbox_inches='tight')
    
    plt.figure()
    plt.plot(x2_parameter_list0[plt2_min:plt2_max],color=pltcolors[0])
    plt.plot(x2_parameter_list1[plt2_min:plt2_max],color=pltcolors[1])
    plt.plot(x2_parameter_list0[plt2_min:plt2_max],'x',color=pltcolors[0])
    plt.plot(x2_parameter_list1[plt2_min:plt2_max],'x',color=pltcolors[1])
    plt.grid(visible=True,which="both",axis="both")
    plt.xticks(list(np.linspace(0,len(parameter_list0[plt2_min:plt2_max]),(len(amp_factor_unique_list[plt_min:plt_max])+1))))
    plt.legend(["noise","noise+signal"])
    plt.xlabel("data point number")
    plt.ylabel("x2 Detection Coefficient (without averaging)")
    if save_plots_flag:
        plt.savefig(fig_dir+'x2_detection_coeff_all.png',bbox_inches='tight')

if False:
    print("Average numbers:")
    print("Parmeter for noise: "+str(np.mean(parameter_avg_list0[plt_min:plt_max])))
    print("Parmeter for signal+noise: "+str(np.mean(parameter_avg_list1[plt_min:plt_max])))

if plot_spectra_flag:
    # plot all oscillator spectra
    # noise only
    freq_lims = [10,20] # [10, 20]
    #y_lims = [0,8] # TEMPORARY
    y_lims = []
    for i in range(len(amp_factor_num_unique_list)):
        inds_cur_long = [j for j,x in enumerate(amp_factor_num_list) if x==amp_factor_num_unique_list[i]]
        
        num_subplots = 3
        num_plots = int(np.ceil(len(inds_cur_long)/num_subplots))
        
        #freq = np.fft.fftfreq(len(z1_spect0_list[0]),del_t_ff)
        
        for m in range(num_plots):
            #num_subplots = int(len(inds_cur_long)/num_plots)
            max_ind = min([(m+1)*num_subplots,len(inds_cur_long)])
            inds_cur = inds_cur_long[m*num_subplots:max_ind]
            
            figure, axis = plt.subplots(num_subplots,3)
            figure.set_size_inches(16,10)
            figure.suptitle("Scaling factor "+str(amp_factor_unique_list[i])+"\n t_shift indicies: "+str([ind%len(inds_cur_long) for ind in inds_cur]))
            
            plt.setp(axis,xticks=list(range(freq_lims[0],freq_lims[1])))
            
            #for ko,k in enumerate(inds_cur):
            for k in inds_cur:
                ko = k-inds_cur[0]
                
                #figure, axis = plt.subplots(1,3)
                #figure.set_size_inches(14,5)
                #figure.suptitle("Amplification factor "+str(amp_factor_list[k])+", t shift "+str(t0_list[k]))
                
                axis[ko,0].plot(freq,np.abs(z1_spect0_list[k]))
                axis[ko,0].grid(visible=True,which="both",axis="both")
                if ko==len(inds_cur)-1:
                    axis[ko,0].set_xlabel("frequency (Hz)")
                if ko==int(round(len(inds_cur)/2))-1:
                    axis[ko,0].set_ylabel("Spectrum magnitude")
                if ko==0:
                    axis[ko,0].set_title("dx1/dt")
                axis[ko,0].set_xlim(freq_lims)
                if len(y_lims) == 2:
                    axis[ko,0].set_ylim(y_lims)
                
                num_annotations = 3
                #maxinds = np.argpartition(np.abs(z1_spect0_list[k]*(freq>=0)),-1*num_annotations)[-1*num_annotations:]
                maxinds = [np.argmax(np.abs(z1_spect0_list[k]*(freq>=0))),
                    np.argmax(np.abs(z1_spect0_list[k]*(freq>15.1))),
                    np.argmax(np.abs(z1_spect0_list[k]*(freq<14.9)*(freq>0)))]
                for n in range(num_annotations):
                    freq_temp = freq[maxinds[n]]
                    z_temp = np.abs(z1_spect0_list[k])[maxinds[n]]
                    axis[ko,0].annotate("("+"{:.2f}".format(freq_temp)+","+"{:.2f}".format(z_temp)+")",(freq_temp,z_temp)\
                        ,size=8)
                    axis[ko,0].plot(freq_temp,z_temp,'x',color='b')
                
                axis[ko,1].plot(freq,np.abs(z2_spect0_list[k]))
                axis[ko,1].grid(visible=True,which="both",axis="both")
                if ko==len(inds_cur)-1:
                    axis[ko,1].set_xlabel("frequency (Hz)")
                if ko==int(round(len(inds_cur)/2))-1:
                    axis[ko,1].set_ylabel("Spectrum magnitude")
                if ko==0:
                    axis[ko,1].set_title("dx2/dt")
                axis[ko,1].set_xlim(freq_lims)
                if len(y_lims) == 2:
                    axis[ko,1].set_ylim(y_lims)
                
                num_annotations = 1
                maxinds = [np.argmax(np.abs(z2_spect0_list[k]*(freq>=0)))]
                for n in range(num_annotations):
                    freq_temp = freq[maxinds[n]]
                    z_temp = np.abs(z2_spect0_list[k])[maxinds[n]]
                    axis[ko,1].annotate("("+"{:.2f}".format(freq_temp)+","+"{:.2f}".format(z_temp)+")",(freq_temp,z_temp)\
                        ,size=8)
                    axis[ko,1].plot(freq_temp,z_temp,'x',color='b')
                
                axis[ko,2].plot(freq,np.abs(z3_spect0_list[k]))
                axis[ko,2].grid(visible=True,which="both",axis="both")
                if ko==len(inds_cur)-1:
                    axis[ko,2].set_xlabel("frequency (Hz)")
                if ko==int(round(len(inds_cur)/2))-1:
                    axis[ko,2].set_ylabel("Spectrum magnitude")
                if ko==0:
                    axis[ko,2].set_title("dx3/dt")
                axis[ko,2].set_xlim(freq_lims)
                if len(y_lims) == 2:
                    axis[ko,2].set_ylim(y_lims)
                
                num_annotations = 3
                maxinds = [np.argmax(np.abs(z3_spect0_list[k]*(freq>=0))),
                    np.argmax(np.abs(z3_spect0_list[k]*(freq>15.1))),
                    np.argmax(np.abs(z3_spect0_list[k]*(freq<14.9)*(freq>0)))]
                for n in range(num_annotations):
                    freq_temp = freq[maxinds[n]]
                    z_temp = np.abs(z3_spect0_list[k])[maxinds[n]]
                    axis[ko,2].annotate("("+"{:.2f}".format(freq_temp)+","+"{:.2f}".format(z_temp)+")",(freq_temp,z_temp)\
                        ,size=8)
                    axis[ko,2].plot(freq_temp,z_temp,'x',color='b')
            if save_plots_flag:
                t_shift_inds = [ind%len(inds_cur_long) for ind in inds_cur]
                spect_save_name = 'oscillator_spectra_scale_'+str(amp_factor_unique_list[i])+'_tshift_inds_'+str(t_shift_inds[0])+'_to_'+str(t_shift_inds[-1])+'.png'
                plt.savefig(fig_dir+spect_save_name,bbox_inches='tight')
if plot_spectra_with_signal_flag:
    # plot all oscillator spectra
    # signal+noise
    freq_lims = [10,20]
    y_lims = [] #[0,130]
    for i in range(len(amp_factor_num_unique_list)):
        inds_cur_long = [j for j,x in enumerate(amp_factor_num_list) if x==amp_factor_num_unique_list[i]]
        
        num_subplots = 3
        num_plots = int(np.ceil(len(inds_cur_long)/num_subplots))
        
        for m in range(num_plots):
            #num_subplots = int(len(inds_cur_long)/num_plots)
            max_ind = min([(m+1)*num_subplots,len(inds_cur_long)])
            inds_cur = inds_cur_long[m*num_subplots:max_ind]
            
            figure, axis = plt.subplots(num_subplots,3)
            figure.set_size_inches(16,10)
            figure.suptitle("Scaling factor "+str(amp_factor_unique_list[i])+" (signal present)\n t_shift indicies: "+str([ind%len(inds_cur_long) for ind in inds_cur]))
            
            plt.setp(axis,xticks=list(range(freq_lims[0],freq_lims[1])))
            
            #for ko,k in enumerate(inds_cur):
            for k in inds_cur:
                ko = k-inds_cur[0]
                
                #figure, axis = plt.subplots(1,3)
                #figure.set_size_inches(14,5)
                #figure.suptitle("Amplification factor "+str(amp_factor_list[k])+", t shift "+str(t0_list[k]))
                
                axis[ko,0].plot(freq,np.abs(z1_spect1_list[k]))
                axis[ko,0].grid(visible=True,which="both",axis="both")
                if ko==len(inds_cur)-1:
                    axis[ko,0].set_xlabel("frequency (Hz)")
                if ko==int(round(len(inds_cur)/2))-1:
                    axis[ko,0].set_ylabel("Spectrum magnitude")
                if ko==0:
                    axis[ko,0].set_title("dx1/dt")
                axis[ko,0].set_xlim(freq_lims)
                if len(y_lims) == 2:
                    axis[ko,0].set_ylim(y_lims)
                
                num_annotations = 3
                #maxinds = np.argpartition(np.abs(z1_spect1_list[k]*(freq>=0)),-1*num_annotations)[-1*num_annotations:]
                maxinds = [np.argmax(np.abs(z1_spect1_list[k]*(freq>=0))),
                    np.argmax(np.abs(z1_spect1_list[k]*(freq>15.1))),
                    np.argmax(np.abs(z1_spect1_list[k]*(freq<14.9)*(freq>0)))]
                for n in range(num_annotations):
                    freq_temp = freq[maxinds[n]]
                    z_temp = np.abs(z1_spect1_list[k])[maxinds[n]]
                    axis[ko,0].annotate("("+"{:.2f}".format(freq_temp)+","+"{:.2f}".format(z_temp)+")",(freq_temp,z_temp)\
                        ,size=8)
                    axis[ko,0].plot(freq_temp,z_temp,'x',color='b')
                
                axis[ko,1].plot(freq,np.abs(z2_spect1_list[k]))
                axis[ko,1].grid(visible=True,which="both",axis="both")
                if ko==len(inds_cur)-1:
                    axis[ko,1].set_xlabel("frequency (Hz)")
                if ko==int(round(len(inds_cur)/2))-1:
                    axis[ko,1].set_ylabel("Spectrum magnitude")
                if ko==0:
                    axis[ko,1].set_title("dx2/dt")
                axis[ko,1].set_xlim(freq_lims)
                if len(y_lims) == 2:
                    axis[ko,1].set_ylim(y_lims)
                
                num_annotations = 1
                maxinds = [np.argmax(np.abs(z2_spect1_list[k]*(freq>=0)))]
                for n in range(num_annotations):
                    freq_temp = freq[maxinds[n]]
                    z_temp = np.abs(z2_spect1_list[k])[maxinds[n]]
                    axis[ko,1].annotate("("+"{:.2f}".format(freq_temp)+","+"{:.2f}".format(z_temp)+")",(freq_temp,z_temp)\
                        ,size=8)
                    axis[ko,1].plot(freq_temp,z_temp,'x',color='b')
                
                axis[ko,2].plot(freq,np.abs(z3_spect1_list[k]))
                axis[ko,2].grid(visible=True,which="both",axis="both")
                if ko==len(inds_cur)-1:
                    axis[ko,2].set_xlabel("frequency (Hz)")
                if ko==int(round(len(inds_cur)/2))-1:
                    axis[ko,2].set_ylabel("Spectrum magnitude")
                if ko==0:
                    axis[ko,2].set_title("dx3/dt")
                axis[ko,2].set_xlim(freq_lims)
                if len(y_lims) == 2:
                    axis[ko,2].set_ylim(y_lims)
                
                num_annotations = 3
                maxinds = [np.argmax(np.abs(z3_spect1_list[k]*(freq>=0))),
                    np.argmax(np.abs(z3_spect1_list[k]*(freq>15.1))),
                    np.argmax(np.abs(z3_spect1_list[k]*(freq<14.9)*(freq>0)))]
                for n in range(num_annotations):
                    freq_temp = freq[maxinds[n]]
                    z_temp = np.abs(z3_spect1_list[k])[maxinds[n]]
                    axis[ko,2].annotate("("+"{:.2f}".format(freq_temp)+","+"{:.2f}".format(z_temp)+")",(freq_temp,z_temp)\
                        ,size=8)
                    axis[ko,2].plot(freq_temp,z_temp,'x',color='b')
            if save_plots_flag:
                t_shift_inds = [ind%len(inds_cur_long) for ind in inds_cur]
                spect_save_name = 'oscillator_spectra_scale_'+str(amp_factor_unique_list[i])+'_tshift_inds_'+str(t_shift_inds[0])+'_to_'+str(t_shift_inds[-1])+'_sig.png'
                plt.savefig(fig_dir+spect_save_name,bbox_inches='tight')

if False:
    print("x10: "+str(d['x1_list'][-1][-1]))
    print("z10: "+str(d['z1_list'][-1][-1]))
    print("x20: "+str(d['x2_list'][-1][-1]))
    print("z20: "+str(d['z2_list'][-1][-1]))
    print("x30: "+str(d['x3_list'][-1][-1]))
    print("z30: "+str(d['z3_list'][-1][-1]))

if False:
    t0_step = 1.225
    t0_list = t0_step*np.linspace(0,9,10)
    inds_list = list(range(10)) #[0,6,9]
    for i in inds_list:
        inds_cur = [j for j,x in enumerate(amp_factor_num_list) if x==amp_factor_num_unique_list[i]]
        plt.figure()
        plt.plot(t0_list,parameter_list0[inds_cur],color=pltcolors[0])
        plt.plot(t0_list,parameter_list1[inds_cur],color=pltcolors[1])
        plt.plot(t0_list,parameter_list0[inds_cur],'x',color=pltcolors[0])
        plt.plot(t0_list,parameter_list1[inds_cur],'x',color=pltcolors[1])
        plt.grid(visible=True,which="both",axis="both")
        #plt.xticks(list(np.linspace(0,20,21)))
        plt.legend(["noise","noise+signal"])
        plt.title("Amplification factor = "+str(amp_factor_unique_list[i]))
        plt.xlabel("noise/signal time shift")
        plt.ylabel("Parameter value")

if False:
    x1 = d['x1_list'][-1]
    t = np.linspace(0,del_t_ff*len(x1),len(x1))
    plt.figure()
    plt.plot(t,x1)
    plt.xlabel('time (s)')
    plt.ylabel('x1')
    plt.title('x1')
    plt.grid(visible=True,which="both",axis="both")
    
    x2 = d['x2_list'][-1]
    plt.figure()
    plt.plot(t,x2)
    plt.xlabel('time (s)')
    plt.ylabel('x2')
    plt.title('x2')
    plt.grid(visible=True,which="both",axis="both")
    
    x3 = d['x3_list'][-1]
    plt.figure()
    plt.plot(t,x3)
    plt.xlabel('time (s)')
    plt.ylabel('x3')
    plt.title('x3')
    plt.grid(visible=True,which="both",axis="both")

if False:
    z2_sig = z2_sig_list[0]
    t = np.linspace(0,del_t_ff*len(z2_sig),len(z2_sig))
    plt.figure()
    plt.plot(t,z2_sig)
    plt.xlabel('time (s)')
    plt.ylabel('z2')
    plt.title('z2 (w signal)')
    plt.grid(visible=True,which="both",axis="both")
    
    z2 = z2_list[0]
    t = np.linspace(0,del_t_ff*len(z2),len(z2))
    plt.figure()
    plt.plot(t,z2)
    plt.xlabel('time (s)')
    plt.ylabel('z2')
    plt.title('z2')
    plt.grid(visible=True,which="both",axis="both")


if False:
    # This is hard coded for data33
    noise_filename = '..\IA2_PAS_DB_dat_ACO_NB#01_nom_filtered_notched_corrected_0s_to_5s_cat.fdb'
    df = np.loadtxt(noise_filename)
    ind_low = int(114.0/del_t_ff)
    ind_high = int(154.0/del_t_ff)
    nn = 1e3*df[ind_low:ind_high,0]
    
    x1 = d['x1_list'][-1]
    x2 = d['x2_list'][-1]
    x3 = d['x3_list'][-1]
    t = np.linspace(0,del_t_ff*len(x1),len(x1))
    omega = 2*np.pi*frequency
    A1 = 1e3
    A2 = 1.45
    signal = A2*np.sin(omega*t)
    ff1 = A1*np.sin(omega*t)
    ff2 = nn + signal
    ff3 = -1*A1*np.sin(omega*t)
    
    pltcolors = ['tab:blue','tab:orange','tab:green','tab:blue','xkcd:gold','tab:green','tab:red','tab:purple','tab:orange','tab:brown']
    lw = 2.0
    
    #t_lims = [40.0-2.0/frequency, 40.0]
    t_lims = [35.0, 40.0]
    y_lims = [-1.6, 1.6]
    ff_lims = [-1100,1100]
    
    plt.figure()
    plt.plot(t,ff1,linewidth=lw,color=pltcolors[0])
    plt.xlabel('time (s)')
    plt.ylabel('ff1')
    plt.title('ff1')
    plt.xlim(t_lims)
    plt.ylim(ff_lims)
    
    plt.figure()
    plt.plot(t,ff2,linewidth=lw,color=pltcolors[0])
    plt.xlabel('time (s)')
    plt.ylabel('ff2')
    plt.title('ff2')
    plt.xlim(t_lims)
    plt.ylim(ff_lims)
    
    plt.figure()
    plt.plot(t,ff3,linewidth=lw,color=pltcolors[0])
    plt.xlabel('time (s)')
    plt.ylabel('ff3')
    plt.title('ff3')
    plt.xlim(t_lims)
    plt.ylim(ff_lims)
    
    plt.figure()
    plt.plot(t,x1,linewidth=lw,color=pltcolors[1])
    plt.xlabel('time (s)')
    plt.ylabel('x1')
    plt.title('x1')
    plt.xlim(t_lims)
    plt.ylim(y_lims)
    
    plt.figure()
    plt.plot(t,x2,linewidth=lw,color=pltcolors[1])
    plt.xlabel('time (s)')
    plt.ylabel('x2')
    plt.title('x2')
    plt.xlim(t_lims)
    plt.ylim(y_lims)
    
    plt.figure()
    plt.plot(t,x3,linewidth=lw,color=pltcolors[1])
    plt.xlabel('time (s)')
    plt.ylabel('x3')
    plt.title('x3')
    plt.xlim(t_lims)
    plt.ylim(y_lims)

toc = datetime.now()
print("execution time: ")
print(toc-tic)

plt.show(block=plot_block)




















