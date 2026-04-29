"""
Plot sphere outputs in the Mollweide projection.

Usage:
    plot_sphere_Mollweide.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    tasks = ['vorticity','Temperature']
    cmap = plt.cm.RdBu_r
    dpi = 100
    figsize = (8, 10)
    #fig, ax = plt.subplots(1,2,figsize=figsize,layout='constrained',projection=ccrs.Mollweide())
    savename_func = lambda write: 'summary_write_Mollweide_{:06}.png'.format(write)
    # Plot writes
    with h5py.File(filename, mode='r') as file:
      plt.clf()
      fig, ax = plt.subplots(2,1,figsize=figsize,layout='constrained',subplot_kw=dict(projection=ccrs.Mollweide())) #projection=ccrs.Mollweide())
      for itask,task in enumerate(tasks):
        dset = file['tasks'][task]
        times = file['scales']['sim_time']
        phi = dset.dims[1][0][:].ravel()
        theta = dset.dims[2][0][:].ravel()
        lat = (np.pi/2 - theta)*180/np.pi
        phi = (phi - np.pi)*180/np.pi
        phi[0]  = -180
        phi[-1] = 180
        phi_mat,lat_mat = np.meshgrid(phi,lat)
        #fig = plt.figure(1,figsize=figsize)
        #ax = #fig.add_subplot(projection=ccrs.Mollweide())
        for index in range(start, start+count):
          print(index)
          plt.clf()
          fig, ax = plt.subplots(2,1,figsize=figsize,layout='constrained',subplot_kw=dict(projection=ccrs.Mollweide()),num=1)
          for itask,task in enumerate(tasks):
            print(task)
            dset = file['tasks'][task]
            times = file['scales']['sim_time']
            #fig = plt.figure(1,figsize=figsize)
            #fig, ax = plt.subplots(2,1,figsize=figsize,layout='constrained',subplot_kw=dict(projection=ccrs.Mollweide()),num=1)
            #ax = fig.add_subplot(projection=ccrs.Mollweide())            
            data_slices = (index, slice(None), slice(None))
            data = dset[data_slices]
            if task == 'Temperature':
                vmax = 1; vmin =0
                data[0,0] = 0.875; data[0,1]=0.975
                contf = ax[itask].pcolormesh(phi_mat,lat_mat,np.transpose(data),transform=ccrs.PlateCarree(),cmap=cmap)#,levels=np.linspace(-vmax,vmax,60))
                ax[itask].gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
                plt.colorbar(contf,ticks=[0.875,0.9,0.925,0.95,0.975],pad=0.1,fraction=0.1,aspect=5)
            else:
              if task == 'vorticity': vmax = 1500
              elif task == 'u_phi':   vmax = 150
              elif task == 'u_theta': vmax = 150
              data[0,0] = vmax; data[-1,-1] = -vmax
              data[data < -vmax] = - vmax
              data[data > vmax]  =   vmax
              contf = ax[itask].pcolormesh(phi_mat,lat_mat,np.transpose(data),transform=ccrs.PlateCarree(),cmap=cmap)#,levels=np.linspace(-vmax,vmax,60))
              ax[itask].gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
              plt.colorbar(contf,ticks=[-vmax,-vmax//2,0,vmax//2,vmax],pad=0.1,fraction=0.1,aspect=5)
            #SET THIS TO CORRECT VALUE 
            time = times[index]#float(dt*index)

            #if task == 'vorticity':  #plt.suptitle('Vorticity at $t$='+'%.2f' % time, fontsize=18)
            #elif task == 'u_phi':   plt.suptitle('Zonal velocity at $t$=%.2f' % time, fontsize=18)
            #elif task == 'u_theta': plt.suptitle('Meridional velocity at $t$=%.2f' % time, fontsize=18)
            #elif task == 'Temperature': #plt.suptitle('Temperature at $t$='+'%.2f' % time, fontsize=18)
          # Save figure
          time = times[index]
          ax[0].set_title('Vorticity at $t$='+'%.2f' % time, fontsize=18) #at $t$='+'%.2f' % time, fontsize=18)
          ax[1].set_title('Temperature', fontsize=18)# at $t$='+'%.2f' % time, fontsize=18)
          plt.text(0.5,0.5,'$t$='+'%.2f' % time)
          savename = savename_func(file['scales/write_number'][index])
          savepath = output.joinpath(savename)
          #plt.tight_layout()
          fig.savefig(str(savepath), dpi=dpi)

    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)
