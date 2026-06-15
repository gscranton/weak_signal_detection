"""
exec(open("prepdata3.py").read())

To process data in:
Shared Technical info > Flight Data > mad_wahoo_flt30_data.zip
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime
import alare_mad_helpers as amh
import findint as fi

def moving_average_subtraction(data, window_size):
    window = np.ones(window_size) / window_size
    
    moving_avg = np.convolve(data, window, mode='same')
    
    detrended_data = data - moving_avg
    
    return detrended_data
    
tic = datetime.now()


cut_num = 3

csv_list = ['./639_2026_01_16_18_37_39_qtfm_gen2.csv']
wvm_csv_list = ['./639_2026_01_16_18_37_39_WVM.csv']
if cut_num==1:
    tmin_list = [[1990, 2082, 2166, 2261, 2342, 2430, 2532, 2616, 2708, 2792, 2887, 2968]]
    tmax_list = [[2036, 2124, 2212, 2300, 2390, 2477, 2574, 2670, 2750, 2847, 2929, 3024]]
elif cut_num==2:
    tmin_list = [[2006.209, 2092.089, 2181.765, 2267.805, 2357.485, 2443.901,\
       2534.865, 2642.149, 2712.813, 2819.417, 2890.185, 2997.609]]
    tmax_list = [[2032.209, 2118.089, 2207.765, 2293.805, 2383.485, 2469.901,\
       2560.865, 2668.149, 2738.813, 2845.417, 2916.185, 3023.609]]
elif cut_num==3:
    tmin_list = [[1919.209, 2005.089, 2094.765, 2180.805, 2270.485, 2356.901,\
       2447.865, 2555.149, 2625.813, 2732.417, 2803.185, 2910.609]]
    tmax_list = [[2049.209, 2135.089, 2224.765, 2310.805, 2400.485, 2486.901,\
       2577.865, 2685.149, 2755.813, 2862.417, 2933.185, 3040.609]]

tlens = [ np.array(tmax_list[n])-np.array(tmin_list[n]) for n in range(np.shape(tmax_list)[0]) ]

save_flag = True
plot_spect_flag = True
single_run_flag = False
use_min_max_flag = True
filter_flag = True # True for saved data
MAS_flag = True
freq_shift_flag = False

MAS_num = 1000
shift_f = 15.625

if filter_flag == False:
    if cut_num==1:
        fout = './prepped_data/flights30_v3_unfiltered_mod1.csv'
    elif cut_num==2:
        fout = './prepped_data/flights30_v3_unfiltered_mod2.csv'
    elif cut_num==3:
        fout = './prepped_data/flights30_v3_unfiltered_mod3.csv'
else:
    if cut_num==1:
        fout = './prepped_data/flights30_v3_mod1.csv'
    elif cut_num==2:
        fout = './prepped_data/flights30_v3_mod2.csv'
    elif cut_num==3:
        fout = './prepped_data/flights30_v3_mod3.csv'

#fout = './prepped_data/flights30_highpass_only_mod1.csv' # used in a special case where lowpass_cutoff was set to 1000

#highpass_cutoff = 0.018
#highpass_cutoff = 0.09
highpass_cutoff = 0.1 #1e-6
lowpass_cutoff = 3.0 # 40 for saved data
butterworth_flag = True
N_butterworth = 5

scale_factor = 1.0

n = 2
dt = 0.004
if single_run_flag:
    datalen = 5200
    samp_start = (300/dt)-5200
    
    datalen = int(300/dt)
    samp_start = 0
    
    samp_step = 12.992/dt
    
    #datalen = int(1e3/dt)
    num_step = 3

t_list = []
d_list = []
freq_list = []
spect_list = []

wt_list = []
wx_list = []
wy_list = []
wz_list = []

for ind in range(len(csv_list)):
    for tind in range(len(tmin_list[ind])):
        csv_name = csv_list[ind]
        wvm_csv = wvm_csv_list[ind] 
        df = pd.read_csv(csv_name, encoding='latin1')
        
        wdf = pd.read_csv(wvm_csv)
        wdf.dropna()
        wdf, _ = amh.trim_after_gps(wdf, sys_time_header='system_time_ns')
        wdf = wdf.dropna()
        wt = wdf['QTFM:Last Timestamp'].to_numpy()
        wx = wdf['NED X (m)'].to_numpy()
        wy = wdf['NED Y (m)'].to_numpy()
        wz = wdf['NED Z (m)'].to_numpy()
        
        if single_run_flag:
            tmin = tmin_list[ind][tind] + samp_start*dt + num_step*samp_step*dt
            tmax = tmin + datalen*dt
        else:
            tmin = tmin_list[ind][tind]
            tmax = tmax_list[ind][tind]
        if use_min_max_flag:
            wx = wx[wt<tmax*1e3]
            wy = wy[wt<tmax*1e3]
            wz = wz[wt<tmax*1e3]
            wt = wt[wt<tmax*1e3]
            wx = wx[wt>tmin*1e3]
            wy = wy[wt>tmin*1e3]
            wz = wz[wt>tmin*1e3]
            wt = wt[wt>tmin*1e3]
        
        wt_list.append(wt)
        wx_list.append(wx)
        wy_list.append(wy)
        wz_list.append(wz)
        
        indstart = 0
	
        traw = np.array( df.iloc[:,11] )*1e-3
        traw = traw[indstart:-2]
        t = traw[traw!=0]
    	
        I = np.array( df.iloc[:,n] )
        I = I[indstart:-2]
        I = I[traw!=0]

        #t = t-t[0]
    
        tm = np.arange(t[0],np.ceil((t[-1]-t[0])/dt)*dt+t[0],dt)
        cs = interpolate.CubicSpline(t,I)
        Ip = cs(tm)
        
        if MAS_flag:
            Ip = moving_average_subtraction(Ip, MAS_num)
    
        if freq_shift_flag:
            Ip = Ip*np.exp(1j*2*np.pi*shift_f*tm)
    
        spect = np.fft.fftshift( np.fft.fft(Ip)/(len(tm)/2) )
        freq = np.fft.fftshift( np.fft.fftfreq(len(tm),dt) )
    
        if filter_flag:
            #
            if butterworth_flag:
                butterworth_highpass = ( 1/np.sqrt( 1 + (highpass_cutoff/freq)**(2*N_butterworth) ) )
                butterworth_lowpass = ( 1/np.sqrt( 1 + (freq/lowpass_cutoff)**(2*N_butterworth) ) )
                spect = spect*butterworth_highpass*butterworth_lowpass
            else:
                spect[np.abs(freq)<highpass_cutoff] = 0
                spect[np.abs(freq)>lowpass_cutoff] = 0
        If = np.real( np.fft.ifft(np.fft.fftshift(spect*(len(tm)/2))) )
    
        if use_min_max_flag:
            If = If[tm<tmax]
            tm = tm[tm<tmax]
            If = If[tm>tmin]
            tm = tm[tm>tmin]
    
        spect = np.fft.fftshift( np.fft.fft(If)/(len(tm)/2) )
        freq = np.fft.fftshift( np.fft.fftfreq(len(tm),dt) )
    
        t_list.append(tm)
        d_list.append(scale_factor*If)
        #t_list.append(tm)
        #d_list.append(Ip)
        spect_list.append(scale_factor*spect)
        freq_list.append(freq)

trajectories = [np.concatenate((np.expand_dims(np.array(wx_list[n]),axis=1),np.expand_dims(np.array(wy_list[n]),axis=1)),axis=1) for n in range(len(wt_list))]

est_x, est_y = fi.find_intersection(trajectories)

center = np.array([est_x,est_y])

cent_t = []
for n in range(len(trajectories)):
    dist = trajectories[n] - center
    absdist = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
    cent_ind = np.argmin(absdist)
    cent_t.append(wt_list[n][cent_ind])

cent_t = np.array(cent_t)*1e-3
lowermarg = cent_t-np.array(tmin_list[0])
uppermarg = np.array(tmax_list[0])-cent_t

for ind in range(len(t_list)):
    if single_run_flag:
        plot_title = "Sensor data file "+str(ind+1)+" time shift number "+str(num_step)
    else:
        plot_title = "Sensor data "+str(ind+1)
    plt.figure()
    plt.plot(t_list[ind],d_list[ind])
    plt.title(plot_title)
    plt.xlabel("Time (s)")
    #plt.ylim([-1e5,1e5])
    #plt.legend(['1','2','3','4','5'])
    #plt.plot(dfo[0,:],np.transpose(dfo[1:,:]),'.')

if plot_spect_flag:
    for ind in range(len(t_list)):
        if single_run_flag:
            plot_title = "Sensor data file "+str(ind+1)+" time shift number "+str(num_step)
        else:
            plot_title = "Sensor data "+str(ind+1)
        plt.figure()
        plt.plot(freq_list[ind],np.abs(spect_list[ind]))
        plt.yscale('log')
        plt.xlim([0,10])
        plt.title(plot_title)
#plt.legend(['1','2','3','4','5'])

plt.show(block=False)

lens = [len(x) for x in t_list]
mli = np.argmax(lens)
ml = np.max(lens)

dfo = np.nan*np.ones((len(t_list)+1,ml))
dfo[0,:] = t_list[mli]
for ind in range(len(d_list)):
    dfo[ind+1,:len(d_list[ind])] = d_list[ind]

if save_flag:
    np.savetxt(fout,np.transpose(dfo),delimiter=",")

toc = datetime.now()
print("Execution time: ")
print(toc-tic)






