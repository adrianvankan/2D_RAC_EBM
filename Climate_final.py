import numpy as np
import dedalus.public as d3
import logging
import scipy
from   scipy.ndimage import gaussian_filter
import dedalus.extras.flow_tools as flow_tools
from   mpi4py import MPI
import dedalus.tools.logging as mpi_logging
import matplotlib.pyplot as plt
import pyshtools as pysh
from   pathlib import Path

logger = logging.getLogger(__name__)
figkw = {'figsize':(6,4), 'dpi':100}

###### Numerical Parameters ####
Nφ        = 256 # 512           # Number of l modes
Nθ        = 128 #256           # Number of m modes
dealias   = 3/2
R         = 1.0           #Sphere radius (non-dim)
timestep  = 1e-5          #initial timestep
stop_sim_time = 100       #stop time
dtype     = np.float64
seed0     = 1
max_timestep = 1e-5 #4e-5

# Set restart = True to restart from a given checkpoint
restart    = False
# Specify checkpoint directory
cp_path    = '/global/scratch/users/avankan/DEDALUS/CLIMATE/dedalus_29029009/checkpoints_s3.h5'#'/global/scratch/users/avankan/DEDALUS/CLIMATE/dedalus_26201462/checkpoints/checkpoints_s15.h5'

## Incoming solar radiation profile (mean annual: 'm.a.', perpetual equinox: p.e.) ##
inc_sol = 'm.a.'

##### Dimensional Parameters ####
R_earth = 6.371e6        #Radius of the Earth in [m]
ν_E     = 1e-10          #3.0e-9   #Eddy Viscosity
κ_E     = ν_E            #Eddy Diffusivity
ν_R     = 1e-7           #Rayleigh damping coefficient
C       = 5*10**7        #Ocean mixed layer heat capacity in J/K per m² of surface area
σ_SB    = 5.67*10**(-8)  #Stefan-Boltzmann's constant
ρ_a     = 10**4          #Air mass in kg per m² of surface area of Earth
Omega   = 0*1/24./3600.  #Rotation frequency in Hz
if inc_sol == 'p.e.':
  ε     = 0.591          #Emissivity
elif inc_sol == 'm.a.':
  ε     = 0.567          #Emissivity # computed from 0.591*(0.930874**4/0.940678**4)
α_c     = 0.7            #Globally averaged co-albedo of Earth
S_sun   = 1360           #Solar irradiance

print('ν_E,κ_E,C,ν_R=',str(ν_E),str(κ_E),str(C),str(ν_R))

#### Scales for Nondimensionalization #####
if inc_sol == 'm.a.':
  F0 = 5./4.*S_sun*α_c/4
  T0 = (F0/ε/σ_SB)**(1/4)
elif inc_sol == 'equi':
  F0 = S_sun*α_c/np.pi
  T0   = (F0/ε/σ_SB)**(1/4)

print('T0=',T0)

###### Convection Scheme Parameters #######
T_c_dim    = 301       #temperature threshold for convection in K
T_c        = T_c_dim/T0    #nondimensionalization
print('T_c =',T_c)
τ_c        = 5e-4          #time after which the convective relaxation is done

#Specify forcing range
ls    = np.array([64])    #specify spherical meridional wavenumber(s) to be forced
n_force = int(dealias*Nθ/ls[0])  #standard deviation in #grid points for use in Gaussian filter in forcing function

##### NONDIMENSIONAL GROUPS ##########
if inc_sol == 'm.a.':
  θ_c = np.arcsin(np.sqrt(5./3.*(1-(T_c**4))))
  print('θ_c=',θ_c*180/np.pi)
elif inc_sol == 'equi':
  θ_c        =   np.arccos(T_c**4)
  print('θ_c=',θ_c*180/np.pi)

#COMPUTE NONDIMENSIONAL CONTROL PARAMETERS
trad         = C*T0/F0
u0           = R_earth/trad #np.sqrt(F1/ρ_a/ν_R)
print('trad=',trad,' u0=',u0)

invRo           = 2*Omega*trad
print('1/Ro=',invRo)

Re_R         = 1/(ν_R*trad)
print('Ro_R=1/(ν_R*trad)=',Re_R)
kin_by_therm = ρ_a*u0**2/(C*T0)

# ============================================================
# Compute and save nondimensional parameters
# ============================================================
#from calculate_and_save_parameters import compute_and_save
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
   params = dict(
       R_earth=R_earth,
       Nθ=Nθ,
       dealias=dealias,
       ls=ls,
       ν_E=ν_E,
       κ_E=κ_E,
       ν_R=ν_R,
       τ_c=τ_c,
       C=C,
       T0=T0,
       F0=F0,
       ρ_a=ρ_a,
       Omega=Omega,
       T_c=T_c)

   #compute_and_save(params)
################################################################

#Timestepper settings
timestepper = d3.RK443

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist   = d3.Distributor(coords, dtype=dtype)
basis  = d3.SphereBasis(coords, (Nφ, Nθ), radius=R, dealias=dealias, dtype=dtype)

# Fields
u      = dist.VectorField(coords, name='u',   bases=basis)
f      = dist.VectorField(coords, name='f',   bases=basis)
f_tmp  = dist.VectorField(coords, name='f_tmp',   bases=basis)
p      = dist.Field(              name='p',   bases=basis)
ψ_f    = dist.Field(              name='ψ_f',   bases=basis)
ψ      = dist.Field(              name='ψ',   bases=basis)
τ_p    = dist.Field(              name='τ_p')
const  = dist.Field(              name='const', bases = basis)
g      = dist.Field(              name='g',   bases=basis)
T      = dist.Field(              name='T',   bases=basis)
h      = dist.Field(              name='h',   bases=basis)
temp   = dist.Field(              name='temp', bases=basis)      #temporary fields for computing box averages
temp2  = dist.Field(              name='temp2', bases=basis)
envelope = dist.Field(            name='envelope',bases=basis)
source = dist.Field(              name='source',bases=basis)


# Coordinates
φ, θ = dist.local_grids(basis)
lat = np.pi / 2 - θ + 0*φ
#################################
np.save('lat_mat_full.npy',lat)

# Latitude dependence of incoming solar radiation
if inc_sol == 'equi':
  h['g'] = np.cos(lat)
elif inc_sol == 'm.a.':
  h['g'] = (1-3./5.*np.sin(lat)**2)

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))
def lapn(A): #nth-order Laplacian
  n=3
  for i in range(n):
    A = d3.lap(A)
  A = (-1)**(n+1) * A
  return A

# Problem (nondimensional)
problem = d3.IVP([u, p, T, τ_p], namespace=locals())
problem.add_equation("div(u) + τ_p = 0")
problem.add_equation("dt(u) +  grad(p) + 1/Re_R*u - ν_E*lapn(u) + invRo*zcross(u) =  - u@grad(u) + d3.skew(d3.grad(ψ_f))")
problem.add_equation("dt(T) - κ_E*lapn(T) = -u@grad(T) + h - T**4 - ν_E*kin_by_therm*u@lapn(u) + 1/Re_R*kin_by_therm*u@u")
problem.add_equation("ave(p) = 0")

# Auxiliary Problem
#problem2 = d3.LBVP([envelope], namespace=locals())
#problem2.add_equation("lap(envelope) - lamb2*envelope = source")

# Define unit vectors
ephi     = dist.VectorField(coords, name='ephi', bases = basis)
etheta   = dist.VectorField(coords, name='etheta', bases = basis)
ephi['g'][0] = 1;   ephi['g'][1]   = 0
etheta['g'][0] = 0; etheta['g'][1] = 1

# Solver(s)
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#Initial condition (spherical harmonic |m| <= l)
if restart == False:
  T['g']   = h['g']**(1/4)
  T['g'][T['g']>T_c] = T_c
  file_handler_mode = 'overwrite'
  initial_timestep = max_timestep
  start_time       = 0
elif restart == True:
  write, initial_timestep = solver.load_state(cp_path)
  file_handler_mode = 'append'

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=2e-3, max_writes=10000,mode=file_handler_mode)
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(T, name='Temperature')
snapshots.add_task(u, name='Velocity')

checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=1, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# Scalar Data
analysis1 = solver.evaluator.add_file_handler("scalar_data", sim_dt=max_timestep)
analysis1.add_task(d3.Average(0.5*(u@u),coords), name="Ekin")
analysis1.add_task(d3.Average((-d3.div(d3.skew(u)))**2, coords), name='Enstrophy')
analysis1.add_task(d3.Average(T,coords), name="Tmean")
analysis1.add_task(1/(kin_by_therm)*d3.Average(T,coords)+d3.Average(0.5*(u@u),coords), name="Etot")
analysis1.add_task(u0*np.sqrt(d3.Average(u@u,coords))/(ν_R*R_earth),name="Re_R")

# Flow properties
flow_prop_cad = 10
flow = d3.GlobalFlowProperty(solver, cadence = flow_prop_cad)
flow.add_property(d3.Average(0.5*(u@u),coords), name = 'avgEkin')
flow.add_property(d3.Average(-kin_by_therm*ν_E*u@lapn(u),coords), name='diss_E')
flow.add_property(d3.Average(kin_by_therm*u@u,coords), name='diss_R')
flow.add_property(d3.Average(T,coords), name='glob_avg_temp')
flow.add_property(u0*np.sqrt(d3.Average(u@u,coords))/(ν_R*R_earth),name="Re")

#GlobalArrayReducer
reducer   = flow_tools.GlobalArrayReducer(comm=MPI.COMM_WORLD)

# CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=1, safety=0.8, threshold=0.8, max_change=100000, min_change=0.00001, max_dt=max_timestep)
CFL.add_velocity(u)

# Main loop
cnt_conv   = 0   #counts number of convection steps

# Initialise random seed to seed0
np.random.seed(seed0)

#set MPI rank and size
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

it0 = solver.iteration
t0  = solver.sim_time
#############
# MAIN LOOP #
#############
g.preset_scales(dealias)
h.preset_scales(dealias)
temp.preset_scales(dealias)
temp2.preset_scales(dealias)
ψ_f.preset_scales(dealias)
source.preset_scales(dealias)
envelope.preset_scales(dealias)

dT = 0.1/T0
hs = lambda x: (scipy.special.erf((x-T_c*np.ones_like(x))/dT)+1.)*0.5

fname = Path("theta_full_"+str(Nθ)+".npy")
if fname.exists():
    theta_full = np.load(fname)

try:
    logger.info('Starting main loop')
    while solver.proceed:
        switch = False  #Default: no convective time step

        #INITIALISE ALL ARRAYS TO CORRECT SIZE IN FIRST ITERATION
        if solver.iteration == it0:
          ε_f = 1
          global_random_field = np.ones_like(temp['g'],dtype=np.complex64)
          φ,θ = basis.local_grids(dist=dist,scales=(dealias,dealias));
          if not fname.exists(): np.save("theta_full_"+str(Nθ)+".npy", θ)
          lat = np.pi / 2 - θ + 0*φ
          φ_mat   = φ   + 0*θ
          θ_mat = θ + 0*φ

          # Initialise random seed to seed0
          np.random.seed(seed0+rank)

          fname = Path("lat_mat_full.npy")
          if not fname.exists(): np.save('lat_mat_full.npy',lat); 
          #np.save('lat_mat_full.npy',lat); np.save('phi_mat_full.npy',φ_mat)
          g['g'] = np.cos(lat)
          if inc_sol == 'equi':
            h['g'] = np.cos(lat)
          elif inc_sol == 'm.a.':
            h['g'] = (1-3./5.*np.sin(lat)**2)

        elif solver.iteration > it0:
          ψ_f['g'] = np.zeros_like(ψ_f['g'])
          time = solver.sim_time

          if time-t0 > cnt_conv*τ_c:
             #print(cnt_conv,solver.iteration)
             switch = True
             #CHECK IF TEMPERATURE ANYWHERE EXCEEDS T_c  --> Set T -> T_c where threshold exceeded
             condTgtTc = np.zeros_like(lat,dtype=bool)
             condTgtTc[T['g']>T_c] = True
             condTleqTc = np.zeros_like(lat,dtype=bool)
             condTleqTc[T['g']<=T_c] = True

             #COMPUTE ENERGY INJECTION RATE AS ENERGY RELEASED IN CONVECTION DIVIDED BY CONVECTIVE TIME
             temp['g'] = T['g']
             temp['g'][condTleqTc] = T_c
             deltaH       = reducer.global_mean((temp['g'] - T_c*np.ones_like(temp['g']))*g['g']) / reducer.global_mean(g['g']) #glob. avg. thermal energy released
             ε_f  = 1/kin_by_therm * deltaH / τ_c  #energy injection rate

             #communication between nodes to ensure forcing on "convective scale"
             global_T = T.allgather_data()
             global_envelope = gaussian_filter( hs(global_T), n_force) #len(global_T)/(2*ls[0]) )
             #global_envelope = gaussian_filter( global_T, n_force )   #len(global_T)/(2*ls[0]) )
             envelope.load_from_global_grid_data(global_envelope)

             #Finally set temperature back to threshold: thermal energy released will be injected by random forcing over next tau_c
             T['g'][condTgtTc] = T_c

          global_random_field = np.zeros_like(temp['g'],dtype=np.complex64)
          power = np.zeros(Nθ); power[ls[0]] = 1;
          clm = pysh.SHCoeffs.from_random(power, seed=123+solver.iteration+rank, kind='complex', lmax=int(dealias*Nθ)-1)
          grid_glq = clm.expand(grid='GLQ')
          global_random_field = np.transpose(grid_glq.data)

          i0 = np.where(theta_full==θ[0][0])
          global_random_field = global_random_field[:,i0[1][0]:i0[1][0]+len(θ[0])]
          phase  = 2*np.pi*np.random.rand()
          temp['g'] = np.real(np.exp(1j*phase)*global_random_field*envelope['g'])
          f_temp     = d3.skew(d3.grad(temp)).evaluate()
          f_temp_var = reducer.global_mean(g['g']*(f_temp['g'][0]**2 + f_temp['g'][1]**2))/reducer.global_mean(g['g'])
          f_temp_var += 1e-10                 #Regularization
          ψ_f['g']   = temp['g'] / np.sqrt(f_temp_var) * np.sqrt(2*ε_f/timestep)

          #plt.pcolormesh(ψ_f['g'],cmap='RdBu'); plt.colorbar(); plt.show()
          #if convective event has occurred, increase counter by 1 before continuing
          if switch == True: cnt_conv += 1

        timestep = CFL.compute_timestep()

        solver.step(timestep)
        if (solver.iteration) % flow_prop_cad == 0:
            glob_avg_temp = flow.max('glob_avg_temp')
            avgEkin  = flow.max('avgEkin')
            diss_E   = flow.max('diss_E')
            diss_R   = flow.max('diss_R')
            Re       = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, Ekin=%f, Ekin/time=%f, Diss_E =%f, Diss_R=%f, Re=%f, <T>_glob=%f'  %(solver.iteration, solver.sim_time, timestep,  avgEkin, avgEkin/solver.sim_time, diss_E, diss_R, Re, glob_avg_temp))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

