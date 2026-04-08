"""
exec(open("make_noise.py").read())
"""

import numpy as np

filename = 'Gaussian_noise_4-8-26.csv'

dt = 0.004
num_dt = int(10000/dt)
num_columns = 1

np.random.seed(101)
df = np.random.normal(0,1,(num_dt,num_columns+1))

df[:,0] = np.arange(0,dt*num_dt,dt)

np.savetxt(filename,df,delimiter=',')
