"""
Editable run configuration for the one-layer advective EBM Dedalus simulation.

Only put independent input parameters here. Derived quantities such as F0, T0,
T_c, Re_R, invRo, kin_by_therm, and n_force are computed in run_climate.py.
"""

# ============================================================
# Numerical parameters
# ============================================================

Nphi = 512
Ntheta = 256
dealias = 3 / 2
R = 1.0

dtype = "float64"  # allowed: "float64", "float32"

initial_timestep = 1e-5
max_timestep = 1e-5
stop_sim_time = 100.0

seed0 = 1

# ============================================================
# Restart / output
# ============================================================

restart = False
checkpoint_path = "../Om0_lf48_wellish_resolved/checkpoints/checkpoints_s9.h5"

snapshot_dt = 2e-3
checkpoint_dt = 1.0
scalar_dt = max_timestep

snapshot_max_writes = 10000
checkpoint_max_writes = 1

# ============================================================
# Solar forcing
# ============================================================

inc_sol = "m.a."  # allowed: "m.a.", "p.e."

# ============================================================
# Dimensional physical parameters
# ============================================================

R_earth = 6.371e6
nu_E = 1e-9
kappa_E = nu_E
nu_R = 1e-7

C = 3e7
sigma_SB = 5.67e-8
rho_a = 1e4

Omega = 0.0
alpha_c = 0.7
S_sun = 1360.0

# ============================================================
# Convective adjustment / stochastic forcing
# ============================================================

T_c_dim = 301.0
tau_c = 5e-4

# Single convective scale: spherical harmonic degree used for stochastic forcing.
l_force = 64

# Gaussian envelope width in dealiased theta-grid points is computed as
# n_force = int(envelope_width_factor * dealias * Ntheta / l_force).
envelope_width_factor = 1.0

# If True, threshold with a sharp mask before Gaussian smoothing.
# If False, use a smooth error-function activation with dimensional width dT_dim.
use_sharp_mask = True
dT_dim = 0.1

# ============================================================
# CFL / diagnostics
# ============================================================

flow_property_cadence = 10
cfl_safety = 0.8
cfl_threshold = 0.8
cfl_max_change = 100000
cfl_min_change = 1e-5
