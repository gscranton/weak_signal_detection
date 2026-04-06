# exec(open("triginterp2.py").read())

import numpy as np
#import matplotlib.pyplot as plt



def trig_upsample(del_x,x0,y,len_new):
    x = np.arange(0,len(y)*del_x,del_x) + x0
    
    Y = np.fft.fft(y)
    #freq = np.fft.fftfreq(len(x),del_x)
    freq_new = np.fft.fftfreq(len_new,del_x*(len(x)/len_new))
    del_freq = 1/(len(x)*del_x)

    pad_size_l = int(np.ceil((len_new-len(y))/2))
    pad_size_u = int(np.floor((len_new-len(y))/2))
    Ynew = np.append(np.zeros(pad_size_l),np.append(np.fft.fftshift(Y),np.zeros(pad_size_u)))

    Ynew = np.fft.ifftshift(Ynew) 

    ynew = np.fft.ifft(Ynew)*(len(Ynew)/len(Y))

    xnew = np.fft.fftfreq(len(ynew),del_freq)
    xnew = np.fft.fftshift(xnew)
    xnew = xnew + (np.min(x)-np.min(xnew))
    
    return xnew,ynew,freq_new,Ynew

#This is fast, and should be working.
#test it with the script below and with test_triginterp_1val.py
#function is based on the Fourier transform property of time shifting
def triginterp_1val(xi,x,y):
    del_x = x[1]-x[0]
    if xi<x[-1]:
        idx = np.argmin(np.abs(x[x>xi]-xi))
        x_shift = x[x>xi][idx]-xi
    else:
        x_shift = x[-1] + del_x*np.ceil((xi-x[-1])/del_x) - xi
    xnew = x+(del_x-x_shift)
    Y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(x),del_x)
    Ynew = Y*np.exp(-1j*2*np.pi*freq*(x_shift-del_x))
    ynew = np.fft.ifft(Ynew)
    idxy = np.argmin(np.abs(xnew-xi))
    yi = ynew[idxy]
    return yi

def triginterp_1val_debug(xi,x,y):
    del_x = x[1]-x[0]
    if xi<x[-1]:
        idx = np.argmin(np.abs(x[x>xi]-xi))
        x_shift = x[x>xi][idx]-xi
    else:
        x_shift = x[-1] + del_x*np.ceil((xi-x[-1])/del_x) - xi
    #x_shift = x[idx]-xi
    xnew = x+(del_x-x_shift)
    Y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(x),del_x)
    Ynew = Y*np.exp(-1j*2*np.pi*freq*(x_shift-del_x))
    ynew = np.fft.ifft(Ynew)
    idxy = np.argmin(np.abs(xnew-xi))
    yi = ynew[idxy]
    return yi,ynew,xnew

def triginterp_arb_grid(xi,x,y):
    yi = np.zeros(len(xi))
    for i in range(len(xi)):
        yi[i] = triginterp_1val(xi[i],x,y)
    return yi

#This works, but is slow
def triginterp(xi,x,y):
    N = len(x)
    h = 2/N
    scale = (x[1]-x[0])/h
    x=x/scale
    xi=xi/scale
    P=np.zeros(len(xi))
    for k in range(N):
        P = P + y[k]*trigcardinal(xi-x[k],N)
    return P

def trigcardinal(x,N):
    if N%2 == 1:
        tau = np.sin(N*np.pi*x/2) / (N*np.sin(np.pi*x/2))
    else:
        tau = np.sin(N*np.pi*x/2) / (N*np.tan(np.pi*x/2))
    tau[x==0]=1
    return tau
    
"""
del_x = 0.65
x0 = 43.2
x = np.arange(0,50*del_x,del_x) + x0
freq = np.fft.fftfreq(len(x),del_x)
y = np.random.normal(0,0.5,len(x)) + np.sin(x*2*np.pi/(len(x)*del_x)) + np.exp(-((x-min(x)-15)/(2))**2)
Y = np.fft.fft(y)
len_new = 10000

xnew,ynew,freq_new,Ynew = trig_upsample(del_x,x0,y,len_new)

ynew2 = np.real(triginterp(xnew,x,y))

x_logspace = np.logspace(np.log10(np.min(x)),np.log10(np.max(x)),int(len(x)/4))

ynew3 = np.real(triginterp(x_logspace,x,y))

x_shift = del_x/2 
yi,ynew4 = triginterp_1val(x_shift+x0,x,y)

plt.figure()
plt.plot(freq,np.abs(Y),'o')
plt.plot(freq_new,np.abs(Ynew))


plt.figure()
plt.plot(x,y,'o')
plt.plot(x_logspace,ynew3,'x',color='r')
plt.plot(xnew,ynew)


plt.figure()
plt.plot(x,y,'o')
plt.plot(x-x_shift,ynew4,'x',color='r')
plt.plot(xnew,ynew)

plt.show(block=False)
"""




























