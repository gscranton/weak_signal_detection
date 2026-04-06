"""
exec(open("testpulse.py").read())
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pulse_filename = '../weak_signals/sensor_data/prepped_data/pulse1.csv'

del_t_ff = 0.004
t_sig_clip = np.arange(0,300,del_t_ff)
signal = np.zeros(np.shape(t_sig_clip))
pulse1_center = 470
pulse1_height = 1.0

dfp = np.loadtxt(str(pulse_filename),delimiter=',')
t_pulse_loaded = np.array(dfp[:,0])
pulse_loaded = np.array(dfp[:,1])

t_pulse_loaded_clip = t_pulse_loaded + del_t_ff*np.round( ( (t_sig_clip[0]+pulse1_center) - ((np.max(t_pulse_loaded)-np.min(t_pulse_loaded))/2)  )/del_t_ff)

pulse_loaded_clip = (pulse1_height/np.max(pulse_loaded))*pulse_loaded[t_pulse_loaded_clip<=t_sig_clip[-1]]
t_pulse_loaded_clip = t_pulse_loaded_clip[t_pulse_loaded_clip<=t_sig_clip[-1]]
pulse_loaded_clip = pulse_loaded_clip[t_pulse_loaded_clip>=t_sig_clip[0]]
t_pulse_loaded_clip = t_pulse_loaded_clip[t_pulse_loaded_clip>=t_sig_clip[0]]

ind_p_start = np.argmin(np.abs(t_sig_clip-t_pulse_loaded_clip[0]))
ind_p_end = np.argmin(np.abs(t_sig_clip-t_pulse_loaded_clip[-1])) + 1
signal1[ind_p_start:ind_p_end] += pulse_loaded_clip

plt.figure()
plt.plot(t_sig_clip,signal)
plt.show(block=False)







