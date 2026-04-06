import sys
sys.path.insert(0, './helper_libs')
import numpy as np
import scipy.interpolate as interpolate
import triginterp2 as ti2
from datetime import datetime

def f1(z1):
    return z1

def f2(z2):
    return z2

def f3(z3):
    return z3

def g1(x1,z1,x2,x3,t,gamma1,alpha1,beta1,kappa12,kappa13,C1,Cff1,ff1,A1,omega1,phi1,analytic_sinusoid_flag):
    result = -gamma1*z1 - alpha1*np.sin(x1) - beta1*x1**3 + C1 + kappa12*(x2-x1) + kappa13*(x3-x1) + Cff1*ff1
    if analytic_sinusoid_flag:
        result = result + A1*np.sin(omega1*t+phi1)
    return result
    
def g2(x2,z2,x1,x3,t,gamma2,alpha2,beta2,kappa21,kappa23,C2,Cff2,ff2,A2,omega2,phi2,analytic_sinusoid_flag):
    result = -gamma2*z2 - alpha2*np.sin(x2) - beta2*x2**3 + C2 + kappa21*(x1-x2) + kappa23*(x3-x2) + Cff2*ff2
    if analytic_sinusoid_flag:
        result = result + A2*np.sin(omega2*t+phi2)
    return result

def g3(x3,z3,x1,x2,t,gamma3,alpha3,beta3,kappa31,kappa32,C3,Cff3,ff3,A3,omega3,phi3,analytic_sinusoid_flag):
    result = -gamma3*z3 - alpha3*np.sin(x3) - beta3*x3**3 + C3 + kappa32*(x2-x3) + kappa31*(x1-x3) + Cff3*ff3
    if analytic_sinusoid_flag:
        result = result + A3*np.sin(omega3*t+phi3)
    return result
    

def rk_solve(t,del_t,x10,z10,x20,z20,x30,z30,gamma1,gamma2,gamma3,alpha1,alpha2,alpha3,beta1,beta2,beta3,C1,C2,C3,Cff1,Cff2,Cff3,kappa12,kappa13,kappa21,kappa23,kappa31,kappa32,\
             omega1,A1,phi1,omega2,A2,phi2,omega3,A3,phi3,ff1,ff2,ff3,t_ff,del_t_ff,interp_type="cubic spline",analytic_sinusoid_flag=True):
    
    x1 = np.zeros(len(t))
    z1 = np.zeros(len(t))
    x2 = np.zeros(len(t))
    z2 = np.zeros(len(t))
    x3 = np.zeros(len(t))
    z3 = np.zeros(len(t))
    x1[0] = x10
    z1[0] = z10
    x2[0] = x20
    z2[0] = z20
    x3[0] = x30
    z3[0] = z30
    
    ff1_clip = ff1[(t_ff<=(np.max(t)+del_t/2)) & (t_ff>=np.min(t))]
    ff2_clip = ff2[(t_ff<=(np.max(t)+del_t/2)) & (t_ff>=np.min(t))]
    ff3_clip = ff3[(t_ff<=(np.max(t)+del_t/2)) & (t_ff>=np.min(t))]
    #t_ff_clip = t_ff[(t_ff<=(np.max(t)+del_t/2)) & (t_ff>=np.min(t))]

    t_interp_half_step,_,_,_ = ti2.trig_upsample(del_t_ff,t[0],ff1_clip,int(round((t[-1]+del_t - t[0])/(del_t/2))))
    
    if interp_type == "cubic spline":
        cs1 = interpolate.CubicSpline(t_ff,ff1_clip)
        cs2 = interpolate.CubicSpline(t_ff,ff2_clip)
        cs3 = interpolate.CubicSpline(t_ff,ff3_clip)
        
        ff1_interp_half_step = cs1(t_interp_half_step)
        ff2_interp_half_step = cs2(t_interp_half_step)
        ff3_interp_half_step = cs3(t_interp_half_step)
    elif interp_type == "trigonometric":
        
        _,ff1_interp_half_step,_,_ = ti2.trig_upsample(del_t_ff,t[0],ff1_clip,int(round((t[-1]+del_t - t[0])/(del_t/2))))
        ff1_interp_half_step = np.real(ff1_interp_half_step)
        _,ff2_interp_half_step,_,_ = ti2.trig_upsample(del_t_ff,t[0],ff2_clip,int(round((t[-1]+del_t - t[0])/(del_t/2))))
        ff2_interp_half_step = np.real(ff2_interp_half_step)
        _,ff3_interp_half_step,_,_ = ti2.trig_upsample(del_t_ff,t[0],ff3_clip,int(round((t[-1]+del_t - t[0])/(del_t/2))))
        ff3_interp_half_step = np.real(ff3_interp_half_step)
        
            
        
    for i in range(len(t)-1):
        
        if interp_type == "linear":
            ff1_0 = np.interp(t[i],t_ff,ff1)
            ff1_half = np.interp(t[i]+0.5*del_t,t_ff,ff1)
            ff1_step = np.interp(t[i]+del_t,t_ff,ff1)
            ff2_0 = np.interp(t[i],t_ff,ff2)
            ff2_half = np.interp(t[i]+0.5*del_t,t_ff,ff2)
            ff2_step = np.interp(t[i]+del_t,t_ff,ff2)
            ff3_0 = np.interp(t[i],t_ff,ff3)
            ff3_half = np.interp(t[i]+0.5*del_t,t_ff,ff3)
            ff3_step = np.interp(t[i]+del_t,t_ff,ff3)
        elif interp_type == "cubic spline":
            ff1_0 = cs1(t[i])
            ff1_half = cs1(t[i]+0.5*del_t)
            ff1_step = cs1(t[i]+del_t)
            ff2_0 = cs2(t[i])
            ff2_half = cs2(t[i]+0.5*del_t)
            ff2_step = cs2(t[i]+del_t)
            ff3_0 = cs3(t[i])
            ff3_half = cs3(t[i]+0.5*del_t)
            ff3_step = cs3(t[i]+del_t)
        elif interp_type == "trigonometric":
            ff1_0 = ff1_interp_half_step[2*i]
            ff1_half = ff1_interp_half_step[2*i+1]
            ff1_step = ff1_interp_half_step[2*(i+1)]
            ff2_0 = ff2_interp_half_step[2*i]
            ff2_half = ff2_interp_half_step[2*i+1]
            ff2_step = ff2_interp_half_step[2*(i+1)]
            ff3_0 = ff3_interp_half_step[2*i]
            ff3_half = ff3_interp_half_step[2*i+1]
            ff3_step = ff3_interp_half_step[2*(i+1)]
    
        k10 = del_t*f1(z1[i])
        l10 = del_t*g1(x1[i],z1[i],x2[i],x3[i],t[i],gamma1,alpha1,beta1,kappa12,kappa13,C1,Cff1,ff1_0,A1,omega1,phi1,analytic_sinusoid_flag)
        k20 = del_t*f2(z2[i])
        l20 = del_t*g2(x2[i],z2[i],x1[i],x3[i],t[i],gamma2,alpha2,beta2,kappa21,kappa23,C2,Cff2,ff2_0,A2,omega2,phi2,analytic_sinusoid_flag)
        k30 = del_t*f3(z3[i])
        l30 = del_t*g3(x3[i],z3[i],x1[i],x2[i],t[i],gamma3,alpha3,beta3,kappa31,kappa32,C3,Cff3,ff3_0,A3,omega3,phi3,analytic_sinusoid_flag)
        
        k11 = del_t*f1(z1[i]+0.5*l10)
        l11 = del_t*g1(x1[i]+0.5*k10,z1[i]+0.5*l10,x2[i]+0.5*k20,x3[i]+0.5*k30,t[i]+0.5*del_t,gamma1,alpha1,beta1,kappa12,kappa13,C1,Cff1,ff1_half,A1,omega1,phi1,analytic_sinusoid_flag)
        k21 = del_t*f2(z2[i]+0.5*l20)
        l21 = del_t*g2(x2[i]+0.5*k20,z2[i]+0.5*l20,x1[i]+0.5*k10,x3[i]+0.5*k30,t[i]+0.5*del_t,gamma2,alpha2,beta2,kappa21,kappa23,C2,Cff2,ff2_half,A2,omega2,phi2,analytic_sinusoid_flag)
        k31 = del_t*f3(z3[i]+0.5*l30)
        l31 = del_t*g3(x3[i]+0.5*k30,z3[i]+0.5*l30,x1[i]+0.5*k10,x2[i]+0.5*k20,t[i]+0.5*del_t,gamma3,alpha3,beta3,kappa31,kappa32,C3,Cff3,ff3_half,A3,omega3,phi3,analytic_sinusoid_flag)
        
        k12 = del_t*f1(z1[i]+0.5*l11)
        l12 = del_t*g1(x1[i]+0.5*k11,z1[i]+0.5*l11,x2[i]+0.5*k21,x3[i]+0.5*k31,t[i]+0.5*del_t,gamma1,alpha1,beta1,kappa12,kappa13,C1,Cff1,ff1_half,A1,omega1,phi1,analytic_sinusoid_flag)
        k22 = del_t*f2(z2[i]+0.5*l21)
        l22 = del_t*g2(x2[i]+0.5*k21,z2[i]+0.5*l21,x1[i]+0.5*k11,x3[i]+0.5*k31,t[i]+0.5*del_t,gamma2,alpha2,beta2,kappa21,kappa23,C2,Cff2,ff2_half,A2,omega2,phi2,analytic_sinusoid_flag)
        k32 = del_t*f3(z3[i]+0.5*l31)
        l32 = del_t*g3(x3[i]+0.5*k31,z3[i]+0.5*l31,x1[i]+0.5*k11,x2[i]+0.5*k21,t[i]+0.5*del_t,gamma3,alpha3,beta3,kappa31,kappa32,C3,Cff3,ff3_half,A3,omega3,phi3,analytic_sinusoid_flag)
        
        k13 = del_t*f1(z1[i]+l12)
        l13 = del_t*g1(x1[i]+k12,z1[i]+l12,x2[i]+k22,x3[i]+k32,t[i]+del_t,gamma1,alpha1,beta1,kappa12,kappa13,C1,Cff1,ff1_step,A1,omega1,phi1,analytic_sinusoid_flag)
        k23 = del_t*f2(z2[i]+l22)
        l23 = del_t*g2(x2[i]+k22,z2[i]+l22,x1[i]+k12,x3[i]+k32,t[i]+del_t,gamma2,alpha2,beta2,kappa21,kappa23,C2,Cff2,ff2_step,A2,omega2,phi2,analytic_sinusoid_flag)
        k33 = del_t*f3(z3[i]+l32)
        l33 = del_t*g3(x3[i]+k32,z3[i]+l32,x1[i]+k12,x2[i]+k22,t[i]+del_t,gamma3,alpha3,beta3,kappa31,kappa32,C3,Cff3,ff3_step,A3,omega3,phi3,analytic_sinusoid_flag)
        
        x1[i+1] = x1[i] + (1/6)*(k10 + 2*k11 + 2*k12 + k13)
        z1[i+1] = z1[i] + (1/6)*(l10 + 2*l11 + 2*l12 + l13)
        x2[i+1] = x2[i] + (1/6)*(k20 + 2*k21 + 2*k22 + k23)
        z2[i+1] = z2[i] + (1/6)*(l20 + 2*l21 + 2*l22 + l23)
        x3[i+1] = x3[i] + (1/6)*(k30 + 2*k31 + 2*k32 + k33)
        z3[i+1] = z3[i] + (1/6)*(l30 + 2*l31 + 2*l32 + l33)
        
        if interp_type=="linear":
            ff2_interp_half_step = np.zeros(len(t_interp_half_step))
        
    return x1,z1,x2,z2,x3,z3,t_interp_half_step,ff2_interp_half_step




















