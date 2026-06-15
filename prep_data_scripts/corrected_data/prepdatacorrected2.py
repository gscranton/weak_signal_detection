"""
exec(open("prepdatacorrected2.py").read())

To process data in:
Shared Technical Info > Yehuda_and_Gregg > corrected_qtfm_data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime

tic = datetime.now()

corrected_flag = True

if corrected_flag:
    
    run_num = 15
    if run_num == 15:
        csv_list = ['./corrected_qtfm_data/15_corr_qtfm_df.csv']
    elif run_num == 16:
        csv_list = ['./corrected_qtfm_data/21_corr_qtfm_df.csv']
    elif run_num == 23:
        csv_list = ['./corrected_qtfm_data/23_corr_qtfm_df.csv']
    elif run_num == 25:
        csv_list = ['./corrected_qtfm_data/25_corr_qtfm_df.csv']
    elif run_num == 26:
        csv_list = ['./corrected_qtfm_data/26_corr_qtfm_df.csv']
    
    if run_num == 15:
        tmin_list = [[1631, 1795, 1980, 2113, 2309, 2476, 2653, 2783]]
        tmax_list = [[1760, 1858, 2076, 2228, 2441, 2537, 2751, 2903]]
    elif run_num == 16:
        tmin_list = [[1037, 1173, 1350, 1480, 1663, 1845, 2015, 2147]]
        tmax_list = [[1155, 1245, 1448, 1538, 1803, 1898, 2118, 2257]]
    elif run_num == 23:
        tmin_list = [[986, 1130, 1232, 1380, 1490, 1630, 1782, 1925, 2050, 2173, 2310, 2457]]
        tmax_list = [[1093, 1202, 1343, 1454, 1596, 1701, 1896, 2000, 2150, 2253, 2430, 2536]]
    elif run_num == 25:
        tmin_list = [[786, 912, 1079, 1203, 1315, 1450, 1606, 1760, 1887, 2037, 2173, 2326]]
        tmax_list = [[877, 998, 1172, 1290, 1420, 1530, 1732, 1830, 2009, 2110, 2297, 2402]]
    elif run_num == 26:
        tmin_list = [[1315, 1475, 1576, 1727, 1833, 1991, 2146, 2308, 2479, 2630, 2764, 2917]]
        tmax_list = [[1438, 1550, 1689, 1800, 1950, 2062, 2280, 2384, 2600, 2705, 2887, 2997]]
    
else:
    csv_list = ['./20251001_MAD_Flt15_SGM0/773_1970_01_01_00_00_53_qtfm_gen2.csv',
    './20251001_MAD_Flt16_SGM0/774_2025_08_26_00_31_22_qtfm_gen2.csv',
    './20251016_MAD_Flt21_SGM0/923_2025_08_26_00_31_21_qtfm_gen2.csv',
    './20251112_MAD_Flt23_SGM0/1048_2025_08_26_00_31_21_qtfm_gen2.csv',
    './20251112_MAD_Flt25_SGM0/1054_1970_01_01_00_00_39_qtfm_gen2.csv',
    './20251112_MAD_Flt26_SGM0/1056_2025_08_26_00_31_21_qtfm_gen2.csv']
    tmin_list = [[1631, 1795, 1980, 2113, 2309, 2476, 2653, 2783],\
	  [1037, 1173, 1350, 1480, 1663, 1845, 2015, 2147],\
	  [935, 1029, 1212, 1351],\
	  [986, 1130, 1232, 1380, 1490, 1630, 1782, 1925, 2050, 2173, 2310, 2457],\
	  [786, 912, 1079, 1203, 1315, 1450, 1606, 1760, 1887, 2037, 2173, 2326],
	  [1315, 1475, 1576, 1727, 1833, 1991, 2146, 2308, 2479, 2630, 2764, 2917]]
	  
    tmax_list = [[1760, 1858, 2076, 2228, 2441, 2537, 2751, 2903],\
	  [1155, 1245, 1448, 1538, 1803, 1898, 2118, 2257],\
	  [996, 1129, 1311, 1441],\
	  [1093, 1202, 1343, 1454, 1596, 1701, 1896, 2000, 2150, 2253, 2430, 2536],\
	  [877, 998, 1172, 1290, 1420, 1530, 1732, 1830, 2009, 2110, 2297, 2402],
	  [1438, 1550, 1689, 1800, 1950, 2062, 2280, 2384, 2600, 2705, 2887, 2997]]



fout = './prepped_data/flights1_run_'+str(run_num)+'_mod2.csv'
save_flag = False
single_run_flag = False
use_min_max_flag = True
filter_flag = False
freq_shift_flag = False
shift_f = 15.625

highpass_cutoff = 0.018
highpass_cutoff = 0.09
highpass_cutoff = 0.1
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
for ind in range(len(csv_list)):
    for tind in range(len(tmin_list[ind])):
        csv_name = csv_list[ind] 
        df = pd.read_csv(csv_name, encoding='latin1')
    
        indstart = 0
	
        if corrected_flag:
            traw = np.array( df['timestamp_ms'] )*1e-3
        else:
            traw = np.array( df.iloc[:,11] )*1e-3
        traw = traw[indstart:-2]
        t = traw[traw!=0]
    	
        if corrected_flag:
            I = np.array( df['corr_scalar_field_nT'] )
        else:
            I = np.array( df.iloc[:,n] )
        I = I[indstart:-2]
        I = I[traw!=0]

        #t = t-t[0]
    
        tm = np.arange(t[0],np.ceil((t[-1]-t[0])/dt)*dt+t[0],dt)
        cs = interpolate.CubicSpline(t,I)
        Ip = cs(tm)
    
        if freq_shift_flag:
            Ip = Ip*np.exp(1j*2*np.pi*shift_f*tm)
    
        spect = np.fft.fftshift( np.fft.fft(Ip)/(len(tm)/2) )
        freq = np.fft.fftshift( np.fft.fftfreq(len(tm),dt) )
    
        if filter_flag:
            #spect[np.abs(freq)>100] = 0
            if butterworth_flag:
                butterworth = ( 1/np.sqrt( 1 + (highpass_cutoff/freq)**(2*N_butterworth) ) )
                spect = spect*butterworth
            else:
                spect[np.abs(freq)<highpass_cutoff] = 0
        If = np.real( np.fft.ifft(np.fft.fftshift(spect*(len(tm)/2))) )
    
        if single_run_flag:
            tmin = tmin_list[ind][tind] + samp_start*dt + num_step*samp_step*dt
            tmax = tmin + datalen*dt
        else:
            tmin = tmin_list[ind][tind]
            tmax = tmax_list[ind][tind]
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

"""
for ind in range(len(t_list)):
    if single_run_flag:
        plot_title = "Sensor data file "+str(ind+1)+" time shift number "+str(num_step)
    else:
        plot_title = "Sensor data "+str(ind+1)
    plt.figure()
    plt.plot(freq_list[ind],np.abs(spect_list[ind]))
    plt.yscale('log')
    plt.xlim([0,125])
    plt.title(plot_title)
#plt.legend(['1','2','3','4','5'])
"""
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






