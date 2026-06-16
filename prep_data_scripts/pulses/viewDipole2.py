"""
exec(open("viewDipole2.py").read())

To process data in:
Shared Technical Info > Dipole Modeling
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime
from scipy.signal import savgol_filter
import copy

tic = datetime.now()

save_flag = False
fname = './Example_Dipole_Modeling_Results.xlsx'
fout = './pulse2.csv'

df = pd.read_excel(fname,header=1,parse_dates=True)

traw = np.array(df['Unnamed: 0'])
tr = np.array((traw-traw[0])*1e-9,dtype=np.float64)
t = np.linspace(0,tr[-1]+0.25,len(traw))

latitude = np.array(df['sensorLatitude'])
longitude = np.array(df['sensorLongitude'])
alt = np.array(df['sensorAltitude'])

data2 = np.array(df['7.54 kA-m2, 0 m HCPA'])
data1 = np.array(df['113.1 kA-m2, 300 m HCPA'])

print("\nMean latitude "+str(np.mean(latitude)))
print("Standard deviation latitude "+str(np.std(latitude)))
print("Max deviation latitude "+str(np.max(np.abs(latitude-np.mean(latitude)))))
print("\nMean longitude "+str(np.mean(longitude)))
print("Std longitude "+str(np.std(longitude)))
print("Max deviation longitude "+str(np.max(np.abs(longitude-np.mean(longitude)))))
print("\nMean altitude "+str(np.mean(alt)))
print("Std altitude "+str(np.std(alt)))
print("Max deviation altitude "+str(np.max(np.abs(alt-np.mean(alt)))))

dt = np.mean(np.diff(t))

data1_mod2 = savgol_filter(data1, 15, 3)

freq = np.fft.fftshift( np.fft.fftfreq(len(t),dt) )

spect1 = np.fft.fftshift( np.fft.fft(data1)/(len(t)/2) )
spect1_mod = np.fft.fftshift( np.fft.fft(data1)/(len(t)/2) )
spect1_mod2 = np.fft.fftshift( np.fft.fft(data1_mod2)/(len(t)/2) )

spect1_mod[np.abs(freq)>0.6] = 0

data1_mod = np.real( np.fft.ifft(np.fft.fftshift(spect1_mod*(len(t)/2))) )

plt.figure()
plt.plot(tr)
plt.plot(t)

plt.figure()
plt.plot(t,tr-t)

plt.figure()
#plt.plot(tr,data1)
plt.plot(t,data1)
plt.plot(t,data1_mod)
plt.plot(t,data1_mod2)

plt.figure()
plt.plot(freq,np.abs(spect1))
plt.plot(freq,np.abs(spect1_mod))
plt.plot(freq,np.abs(spect1_mod2))
plt.yscale('log')

plt.figure()
plt.plot(t,data1-data1)
plt.plot(t,data1-data1_mod)
plt.plot(t,data1-data1_mod2)

"""
plt.figure()
plt.plot(tr,data2,'.')
plt.plot(t,data2,'.')
"""

plt.show(block=False)

dfo = np.zeros((len(t),2))
dfo[:,0] = t
dfo[:,1] = data1_mod2

if save_flag:
    np.savetxt(fout,dfo,delimiter=",")

toc = datetime.now()
print("\nExecution time: ")
print(toc-tic)








