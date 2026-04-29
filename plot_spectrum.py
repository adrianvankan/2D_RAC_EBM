import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import dedalus.public as d3
import logging
import scipy
import matplotlib as mpl
from matplotlib import rc
mpl.use('TkAgg')

################################################################################
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

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

ind = 1
f = h5py.File('snapshots/snapshots_s1.h5')
print(list(f['scales']))

u_phi = f['tasks']['u_phi']
u_theta = f['tasks']['u_theta']

# Parameters
Nphi    = 512
Ntheta  = 256
dealias = 3/2
dtype   = np.float64
R       = 1
nhyper  = 3
ν_E     = 3e-9   #Eddy Viscosity

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist   = d3.Distributor(coords, dtype=dtype)
basis  = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

u = dist.VectorField(coords, name='u',   bases=basis)
#u['g'][0] = u_phi[-1,:,:]
#u['g'][1] = u_theta[-1,:,:]
#u_phi_c   = u['c'][0]
#u_theta_c = u['c'][1]

#######################
Nt         = len(u_phi[:,0,0]);
print('total number of time steps=',Nt)
navg = 10
it_sta     = Nt - navg
it_end     = Nt

E_spec_avg = np.zeros(Ntheta)
for it in range(it_sta,it_end):
  u['g'][0] = u_phi[it,:,:]
  u['g'][1] = u_theta[it,:,:]
  u_c   = u['c'][0]
  v_c   = u['c'][1]
  #u_c    = u_phi_c[it,:,:]
  #v_c    = u_theta_c[it,:,:]
  E_spec = np.zeros(Ntheta)
  for i in range(u_c.shape[0]):
    for j in range(u_c.shape[1]):
      groups = basis.elements_to_groups((False, False), (np.array((i,)),np.array((j,))))
      m   = int(groups[0][0])
      ell = int(groups[1][0])
      E_spec[ell] += 0.5*(u_c[i,j]**2 + v_c[i,j]**2)
  E_spec_avg += E_spec
E_spec_avg /= (it_end-it_sta+1)
############################################################
ells = np.arange(Ntheta)+1
kf = 80
plt.figure(layout='constrained')
plt.loglog(ells,E_spec_avg); plt.ylabel('$E(\\ell)$'); plt.xlabel('$\\ell$')
plt.vlines(kf,1e-10,100,color='k',linestyle='--',lw=3,alpha=0.5)
plt.plot(np.linspace(10,kf),np.linspace(10,kf)**(-5/3)*5000,'b--',lw=3,alpha=0.8);
plt.plot(np.linspace(kf,512),np.linspace(kf,512)**(-5)*5000*kf**5/kf**(5/3),'r--',lw=3,alpha=0.8);
plt.ylim(ymin=4e-6,ymax=100); plt.xlim(1,256);
plt.savefig('E_of_k.pdf')

plt.figure(layout='constrained'); plt.ylabel('$D(\\ell)$'); plt.xlabel('$\\ell$')
plt.loglog(ells,ν_E*E_spec_avg*np.array(ells)**(nhyper+1)*(np.array(ells)+1)**(nhyper+1)); plt.show()
