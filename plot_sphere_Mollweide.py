#!/usr/bin/env python3
"""
Plot Dedalus sphere outputs in the Mollweide projection.

Usage:
    plot_sphere_Mollweide.py <files>... [--output=<dir>] [--overwrite]

Options:
    --output=<dir>  Output directory [default: ./frames]
    --overwrite     Recreate frames even if the PNG already exists.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from docopt import docopt
from dedalus.tools import post
from dedalus.tools.parallel import Sync
from mpi4py import MPI


TASKS = ("vorticity", "Temperature")
CMAP = plt.cm.RdBu_r
DPI = 100
FIGSIZE = (8, 10)
PROJECTION = ccrs.Mollweide()
TRANSFORM = ccrs.PlateCarree()


def savename_from_write(write_number: int) -> str:
    return f"summary_write_Mollweide_{int(write_number):06d}.png"


def get_lon_lat_matrices(dset):
    phi = dset.dims[1][0][:].ravel()
    theta = dset.dims[2][0][:].ravel()

    lon = (phi - np.pi) * 180.0 / np.pi
    lat = (np.pi / 2.0 - theta) * 180.0 / np.pi

    lon[0] = -180.0
    lon[-1] = 180.0

    return np.meshgrid(lon, lat)


def plot_task(ax, lon_mat, lat_mat, data, task):
    data = np.array(data, copy=True)

    if task == "Temperature":
        ticks = [0.875, 0.9, 0.925, 0.95, 0.975]
        data[0, 0] = ticks[0]
        data[0, 1] = ticks[-1]

        mesh = ax.pcolormesh(
            lon_mat,
            lat_mat,
            data.T,
            transform=TRANSFORM,
            cmap=CMAP,
            shading="auto",
        )
        ax.gridlines(
            draw_labels=True,
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        return mesh, ticks

    if task == "vorticity":
        vmax = 1500.0
    elif task in ("u_phi", "u_theta"):
        vmax = 150.0
    else:
        vmax = float(np.nanmax(np.abs(data)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

    data = np.clip(data, -vmax, vmax)
    data[0, 0] = vmax
    data[-1, -1] = -vmax

    mesh = ax.pcolormesh(
        lon_mat,
        lat_mat,
        data.T,
        transform=TRANSFORM,
        cmap=CMAP,
        shading="auto",
    )
    ax.gridlines(
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    ticks = [-vmax, -vmax / 2, 0, vmax / 2, vmax]
    return mesh, ticks


def main(filename, start, count, output, overwrite=False):
    output = pathlib.Path(output)
    rank = MPI.COMM_WORLD.rank

    with h5py.File(filename, mode="r") as file:
        missing = [task for task in TASKS if task not in file["tasks"]]
        if missing:
            raise KeyError(f"Missing task(s) in {filename}: {missing}")

        times = file["scales/sim_time"][:]
        write_numbers = file["scales/write_number"][:]

        first_task = file["tasks"][TASKS[0]]
        lon_mat, lat_mat = get_lon_lat_matrices(first_task)

        stop = min(start + count, len(times))

        for index in range(start, stop):
            write_number = int(write_numbers[index])
            savepath = output / savename_from_write(write_number)

            if savepath.exists() and not overwrite:
                print(f"Skipping {savepath} (already exists)")
                continue

            print(f"Plotting write {write_number} from {filename}")

            fig, axes = plt.subplots(
                len(TASKS),
                1,
                figsize=FIGSIZE,
                layout="constrained",
                subplot_kw={"projection": PROJECTION},
            )
            axes = np.atleast_1d(axes)

            for ax, task in zip(axes, TASKS):
                data = file["tasks"][task][index, :, :]
                mesh, ticks = plot_task(ax, lon_mat, lat_mat, data, task)
                fig.colorbar(
                    mesh,
                    ax=ax,
                    ticks=ticks,
                    pad=0.08,
                    fraction=0.08,
                    aspect=8,
                )

            time = float(times[index])
            axes[0].set_title(f"Vorticity at t={time:.2f}", fontsize=18)
            axes[1].set_title("Temperature", fontsize=18)
            fig.suptitle(f"t={time:.2f}", fontsize=16)

            # Important: temporary filename must still end in .png,
            # otherwise Matplotlib interprets the format as "tmp".
            tmppath = savepath.with_name(
                f"{savepath.stem}.rank{rank}.tmp.png"
            )

            fig.savefig(str(tmppath), dpi=DPI, format="png")
            tmppath.replace(savepath)
            plt.close(fig)


def cli():
    args = docopt(__doc__)
    output_path = pathlib.Path(args["--output"]).absolute()

    with Sync() as sync:
        if sync.comm.rank == 0:
            output_path.mkdir(parents=True, exist_ok=True)

    post.visit_writes(
        args["<files>"],
        main,
        output=output_path,
        overwrite=bool(args["--overwrite"]),
    )


if __name__ == "__main__":
    cli()
