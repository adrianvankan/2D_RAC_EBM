import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import levy_stable
#from mpmath import *
import matplotlib as mpl
from matplotlib import rc
from datetime import datetime
#import numba as nb
import h5py
#################################################################################
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#set font sizes
SMALL_SIZE = 22
MEDIUM_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
################################################################################

f = h5py.File('scalar_data/scalar_data_s1/scalar_data_s1_p0.h5','r')
print(list(f.keys()))

dset  = f['tasks']
print(list(dset))
Ek    = dset['Ekin'] 
Tm    = dset['Tmean']

figsize = (6,8)
fig, ax = plt.subplots(2,1,figsize=figsize,layout='constrained',num=1)

max_timestep = 1e-5
dt = max_timestep

time = dt*np.arange(0,len(Ek[:,0,0]))
ax[0].plot(time[::10],(Ek[::10,0,0]),'-',lw=2); ax[0].set_xlabel('$t$'); ax[0].set_ylabel('$E_{kin}$')
ax[0].set_ylim(ymin=0)

ax[1].plot(time[::10],Tm[::10,0,0],'-',lw=2);   ax[1].set_xlabel('$t$'); ax[1].set_ylabel('$\\langle T\\rangle_{glob}$')

T_c = 0.9704793135362177
ax[1].plot(time[::10], np.ones_like(time[::10])*T_c,'k--',lw=2)
ax[1].set_ylim(max(Tm[::10,0,0])*0.9,1.1*max(Tm[::10,0,0]))
plt.savefig('timeseries.png')

plt.show()
