#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dedalus run script for the one-layer advective EBM.

Usage
-----
    python run_climate.py config_climate.py

The config file contains only independent input parameters. This script computes
all derived nondimensional parameters internally and should normally not be edited
between runs.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import dedalus.public as d3
import dedalus.extras.flow_tools as flow_tools
from mpi4py import MPI
import pyshtools as pysh

logger = logging.getLogger(__name__)
figkw = {"figsize": (6, 4), "dpi": 100}


def load_config(path: str | Path):
    """Load a Python config file as a module."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location("climate_run_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import config file: {path}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def validate_config(cfg) -> None:
    """Basic sanity checks for independent input parameters."""
    if cfg.inc_sol not in ("m.a.", "p.e."):
        raise ValueError("inc_sol must be either 'm.a.' or 'p.e.'.")
    if cfg.dtype not in ("float64", "float32"):
        raise ValueError("dtype must be 'float64' or 'float32'.")
    if cfg.Nphi <= 0 or cfg.Ntheta <= 0:
        raise ValueError("Nphi and Ntheta must be positive.")
    if cfg.dealias < 1:
        raise ValueError("dealias should be >= 1.")
    if cfg.l_force <= 0:
        raise ValueError("l_force must be positive.")
    if cfg.l_force >= cfg.dealias * cfg.Ntheta:
        raise ValueError("l_force must be smaller than dealias*Ntheta.")
    if cfg.max_timestep <= 0 or cfg.initial_timestep <= 0:
        raise ValueError("Timesteps must be positive.")
    if cfg.stop_sim_time <= 0:
        raise ValueError("stop_sim_time must be positive.")
    if cfg.tau_c <= 0:
        raise ValueError("tau_c must be positive.")
    if cfg.envelope_width_factor <= 0:
        raise ValueError("envelope_width_factor must be positive.")
    if cfg.restart and not getattr(cfg, "checkpoint_path", None):
        raise ValueError("restart=True requires checkpoint_path.")


def get_dtype(dtype_name: str):
    if dtype_name == "float64":
        return np.float64
    if dtype_name == "float32":
        return np.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def compute_derived_parameters(cfg):
    """Compute derived dimensional and nondimensional parameters."""
    if cfg.inc_sol == "m.a.":
        epsilon = 0.567
        F0 = 5.0 / 4.0 * cfg.S_sun * cfg.alpha_c / 4.0
    elif cfg.inc_sol == "p.e.":
        epsilon = 0.591
        F0 = cfg.S_sun * cfg.alpha_c / np.pi
    else:
        raise ValueError(f"Unknown inc_sol={cfg.inc_sol!r}")

    T0 = (F0 / (epsilon * cfg.sigma_SB)) ** 0.25
    T_c = cfg.T_c_dim / T0
    trad = cfg.C * T0 / F0
    u0 = cfg.R_earth / trad
    invRo = 2.0 * cfg.Omega * trad
    Re_R = 1.0 / (cfg.nu_R * trad)
    kin_by_therm = cfg.rho_a * u0**2 / (cfg.C * T0)

    l_force = int(cfg.l_force)
    ls = np.array([l_force])
    n_force = int(cfg.envelope_width_factor * cfg.dealias * cfg.Ntheta / l_force)
    n_force = max(n_force, 1)

    if cfg.inc_sol == "m.a.":
        theta_c = np.arcsin(np.sqrt(5.0 / 3.0 * (1.0 - T_c**4)))
    else:
        theta_c = np.arccos(T_c**4)

    return dict(
        epsilon=epsilon,
        F0=F0,
        T0=T0,
        T_c=T_c,
        trad=trad,
        u0=u0,
        invRo=invRo,
        Re_R=Re_R,
        kin_by_therm=kin_by_therm,
        ls=ls,
        n_force=n_force,
        theta_c=theta_c,
    )


def log_parameters(cfg, par, rank: int) -> None:
    if rank != 0:
        return
    print("nu_E,kappa_E,C,nu_R=", cfg.nu_E, cfg.kappa_E, cfg.C, cfg.nu_R)
    print("T0=", par["T0"])
    print("T_c=", par["T_c"])
    print("theta_c=", par["theta_c"] * 180.0 / np.pi)
    print("trad=", par["trad"], " u0=", par["u0"])
    print("1/Ro=", par["invRo"])
    print("Ro_R=1/(nu_R*trad)=", par["Re_R"])
    print("kin_by_therm=", par["kin_by_therm"])
    print("l_force=", cfg.l_force, " n_force=", par["n_force"])


def main() -> None:
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python run_climate.py config_climate.py")

    cfg = load_config(sys.argv[1])
    validate_config(cfg)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dtype = get_dtype(cfg.dtype)
    par = compute_derived_parameters(cfg)
    log_parameters(cfg, par, rank)

    # Unpack frequently used parameters using symbols close to the original code.
    Nφ = cfg.Nphi
    Nθ = cfg.Ntheta
    dealias = cfg.dealias
    R = cfg.R
    timestep = cfg.initial_timestep
    max_timestep = cfg.max_timestep
    stop_sim_time = cfg.stop_sim_time
    seed0 = cfg.seed0

    ν_E = cfg.nu_E
    κ_E = cfg.kappa_E
    ν_R = cfg.nu_R
    T_c = par["T_c"]
    τ_c = cfg.tau_c
    invRo = par["invRo"]
    Re_R = par["Re_R"]
    kin_by_therm = par["kin_by_therm"]
    u0 = par["u0"]
    ls = par["ls"]
    n_force = par["n_force"]

    # Timestepper settings
    timestepper = d3.RK443

    # Bases
    coords = d3.S2Coordinates("phi", "theta")
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nφ, Nθ), radius=R, dealias=dealias, dtype=dtype)

    # Fields
    u = dist.VectorField(coords, name="u", bases=basis)
    p = dist.Field(name="p", bases=basis)
    ψ_f = dist.Field(name="ψ_f", bases=basis)
    τ_p = dist.Field(name="τ_p")
    g = dist.Field(name="g", bases=basis)
    T = dist.Field(name="T", bases=basis)
    h = dist.Field(name="h", bases=basis)
    temp = dist.Field(name="temp", bases=basis)
    envelope = dist.Field(name="envelope", bases=basis)
    source = dist.Field(name="source", bases=basis)

    # Coordinates
    φ, θ = dist.local_grids(basis)
    lat = np.pi / 2.0 - θ + 0.0 * φ
    np.save("lat_mat_full.npy", lat)

    # Latitude dependence of incoming solar radiation
    if cfg.inc_sol == "p.e.":
        h["g"] = np.cos(lat)
    elif cfg.inc_sol == "m.a.":
        h["g"] = 1.0 - 3.0 / 5.0 * np.sin(lat) ** 2

    # Substitutions
    zcross = lambda A: d3.MulCosine(d3.skew(A))

    def lapn(A):
        """nth-order signed hyper-Laplacian used in the original script."""
        n = 3
        for _ in range(n):
            A = d3.lap(A)
        A = (-1) ** (n + 1) * A
        return A

    # Problem (nondimensional)
    # Dedalus evaluates equation strings in the provided namespace. Since this
    # script is organized inside main(), we explicitly include module-level names
    # such as d3, and freeze the namespace before adding equations.
    namespace = locals().copy()
    namespace["d3"] = d3

    problem = d3.IVP([u, p, T, τ_p], namespace=namespace)
    problem.add_equation("div(u) + τ_p = 0")
    problem.add_equation(
        "dt(u) + grad(p) + 1/Re_R*u - ν_E*lapn(u) + invRo*zcross(u) "
        "= - u@grad(u) + d3.skew(d3.grad(ψ_f))"
    )
    problem.add_equation(
        "dt(T) - κ_E*lapn(T) = -u@grad(T) + h - T**4 "
        "- ν_E*kin_by_therm*u@lapn(u) + 1/Re_R*kin_by_therm*u@u"
    )
    problem.add_equation("ave(p) = 0")

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial condition / restart
    if not cfg.restart:
        T["g"] = h["g"] ** 0.25
        T["g"][T["g"] > T_c] = T_c
        file_handler_mode = "overwrite"
    else:
        solver.load_state(cfg.checkpoint_path)
        file_handler_mode = "append"

    # Analysis
    snapshots = solver.evaluator.add_file_handler(
        "snapshots",
        sim_dt=cfg.snapshot_dt,
        max_writes=cfg.snapshot_max_writes,
        mode=file_handler_mode,
    )
    snapshots.add_task(p, name="pressure")
    snapshots.add_task(-d3.div(d3.skew(u)), name="vorticity")
    snapshots.add_task(T, name="Temperature")
    snapshots.add_task(u, name="Velocity")

    checkpoints = solver.evaluator.add_file_handler(
        "checkpoints",
        sim_dt=cfg.checkpoint_dt,
        max_writes=cfg.checkpoint_max_writes,
        mode=file_handler_mode,
    )
    checkpoints.add_tasks(solver.state)

    analysis1 = solver.evaluator.add_file_handler("scalar_data", sim_dt=cfg.scalar_dt)
    analysis1.add_task(d3.Average(0.5 * (u @ u), coords), name="Ekin")
    analysis1.add_task(d3.Average((-d3.div(d3.skew(u))) ** 2, coords), name="Enstrophy")
    analysis1.add_task(d3.Average(T, coords), name="Tmean")
    analysis1.add_task(
        1 / kin_by_therm * d3.Average(T, coords) + d3.Average(0.5 * (u @ u), coords),
        name="Etot",
    )
    analysis1.add_task(
        u0 * np.sqrt(d3.Average(u @ u, coords)) / (ν_R * cfg.R_earth),
        name="Re_R",
    )

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=cfg.flow_property_cadence)
    flow.add_property(d3.Average(0.5 * (u @ u), coords), name="avgEkin")
    flow.add_property(d3.Average(-kin_by_therm * ν_E * u @ lapn(u), coords), name="diss_E")
    flow.add_property(d3.Average(kin_by_therm * u @ u, coords), name="diss_R")
    flow.add_property(d3.Average(T, coords), name="glob_avg_temp")
    flow.add_property(u0 * np.sqrt(d3.Average(u @ u, coords)) / (ν_R * cfg.R_earth), name="Re")

    reducer = flow_tools.GlobalArrayReducer(comm=MPI.COMM_WORLD)

    # CFL
    CFL = d3.CFL(
        solver,
        initial_dt=timestep,
        cadence=1,
        safety=cfg.cfl_safety,
        threshold=cfg.cfl_threshold,
        max_change=cfg.cfl_max_change,
        min_change=cfg.cfl_min_change,
        max_dt=max_timestep,
    )
    CFL.add_velocity(u)

    # Main-loop setup
    cnt_conv = 0
    np.random.seed(seed0)

    it0 = solver.iteration
    t0 = solver.sim_time

    g.preset_scales(dealias)
    h.preset_scales(dealias)
    temp.preset_scales(dealias)
    ψ_f.preset_scales(dealias)
    source.preset_scales(dealias)
    envelope.preset_scales(dealias)

    dT = cfg.dT_dim / par["T0"]
    hs = lambda x: 0.5 * (scipy.special.erf((x - T_c * np.ones_like(x)) / dT) + 1.0)

    theta_file = Path(f"theta_full_{Nθ}.npy")
    if theta_file.exists():
        theta_full = np.load(theta_file)
    else:
        theta_full = None

    try:
        logger.info("Starting main loop")
        while solver.proceed:
            switch = False

            if solver.iteration == it0:
                ε_f = 1.0
                φ, θ = basis.local_grids(dist=dist, scales=(dealias, dealias))
                if not theta_file.exists():
                    np.save(theta_file, θ)
                    theta_full = θ
                lat = np.pi / 2.0 - θ + 0.0 * φ

                np.random.seed(seed0 + rank)

                lat_file = Path("lat_mat_full.npy")
                if not lat_file.exists():
                    np.save(lat_file, lat)

                g["g"] = np.cos(lat)
                if cfg.inc_sol == "p.e.":
                    h["g"] = np.cos(lat)
                elif cfg.inc_sol == "m.a.":
                    h["g"] = 1.0 - 3.0 / 5.0 * np.sin(lat) ** 2

            elif solver.iteration > it0:
                ψ_f["g"] = np.zeros_like(ψ_f["g"])
                time = solver.sim_time

                if time - t0 > cnt_conv * τ_c:
                    switch = True

                    condTgtTc = T["g"] > T_c
                    condTleqTc = T["g"] <= T_c

                    temp["g"] = T["g"]
                    temp["g"][condTleqTc] = T_c
                    deltaH = reducer.global_mean((temp["g"] - T_c) * g["g"]) / reducer.global_mean(g["g"])
                    ε_f = 1.0 / kin_by_therm * deltaH / τ_c

                    # Communication between nodes to ensure forcing on convective scale.
                    global_T = T.allgather_data()
                    if cfg.use_sharp_mask:
                        mask = (global_T > T_c).astype(global_T.dtype)
                    else:
                        mask = hs(global_T)
                    global_envelope = gaussian_filter(mask, n_force)
                    envelope.load_from_global_grid_data(global_envelope)

                    # Relax supercritical temperature to threshold.
                    T["g"][condTgtTc] = T_c

                global_random_field = np.zeros_like(temp["g"], dtype=np.complex64)
                power = np.zeros(Nθ)
                power[ls[0]] = 1.0
                clm = pysh.SHCoeffs.from_random(
                    power,
                    seed=123 + solver.iteration + rank,
                    kind="complex",
                    lmax=int(dealias * Nθ) - 1,
                )
                grid_glq = clm.expand(grid="GLQ")
                global_random_field = np.transpose(grid_glq.data)

                if theta_full is None:
                    theta_full = np.load(theta_file)
                # Locate the local theta block in the global GLQ grid.
                # Use nearest-index matching rather than exact floating-point equality.
                j0 = int(np.argmin(np.abs(theta_full[0, :] - θ[0, 0])))
                global_random_field = global_random_field[:, j0 : j0 + len(θ[0])]

                phase = 2.0 * np.pi * np.random.rand()
                temp["g"] = np.real(np.exp(1j * phase) * global_random_field * envelope["g"])
                f_temp = d3.skew(d3.grad(temp)).evaluate()
                f_temp_var = reducer.global_mean(
                    g["g"] * (f_temp["g"][0] ** 2 + f_temp["g"][1] ** 2)
                ) / reducer.global_mean(g["g"])
                f_temp_var += 1e-10
                ψ_f["g"] = temp["g"] / np.sqrt(f_temp_var) * np.sqrt(2.0 * ε_f / timestep)

                if switch:
                    cnt_conv += 1

            timestep = CFL.compute_timestep()
            solver.step(timestep)

            if solver.iteration % cfg.flow_property_cadence == 0:
                glob_avg_temp = flow.max("glob_avg_temp")
                avgEkin = flow.max("avgEkin")
                diss_E = flow.max("diss_E")
                diss_R = flow.max("diss_R")
                Re = flow.max("Re")
                logger.info(
                    "Iteration=%i, Time=%e, dt=%e, Ekin=%f, Ekin/time=%f, "
                    "Diss_E=%f, Diss_R=%f, Re=%f, <T>_glob=%f",
                    solver.iteration,
                    solver.sim_time,
                    timestep,
                    avgEkin,
                    avgEkin / solver.sim_time,
                    diss_E,
                    diss_R,
                    Re,
                    glob_avg_temp,
                )

    except Exception:
        logger.error("Exception raised, triggering end of main loop.")
        raise
    finally:
        solver.log_stats()


if __name__ == "__main__":
    main()
