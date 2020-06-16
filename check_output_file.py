# Simple script to check output file

import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

outdir = '.\\output\\'
output_file = 'idealized_synthetic_lidar_scene_A.hdf5'
outfile = outdir + output_file

h5f = h5py.File(outfile, 'r')

z = np.array(h5f['bin_alt_array_km'])
x = np.array(h5f['distance_km'])
Bm = np.array(h5f['Bm'])
Em = np.array(h5f['sigma_m'])
Bp = np.array(h5f['Bp'])
Ep = np.array(h5f['sigma_p'])
h5f.close()
B = Bp + Bm

mx = 0.95 * Bm.max()
extent = [x.min(),x.max(),z.min(),z.max()]
plt.figure(figsize=(11,7))
im = plt.imshow(Bm.T, vmin=0, vmax=mx, aspect='auto', extent=extent, cmap='nipy_spectral')
plt.title('Molecular Backscatter')
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size="2%",pad=0.05)
plt.colorbar(im, cax=cax)
plt.savefig(outdir+'molecular_backscatter.png')


mx = 0.95 * Bp.max()
extent = [x.min(),x.max(),z.min(),z.max()]
plt.figure(figsize=(11,7))
im = plt.imshow(Bp.T, vmin=0, vmax=mx, aspect='auto', extent=extent, cmap='nipy_spectral')
plt.title('Particulate Backscatter')
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size="2%",pad=0.05)
plt.colorbar(im, cax=cax)
plt.savefig(outdir+'particulate_backscatter.png')


mx = 0.95 * Em.max()
extent = [x.min(),x.max(),z.min(),z.max()]
plt.figure(figsize=(11,7))
im = plt.imshow(Em.T, vmin=0, vmax=mx, aspect='auto', extent=extent, cmap='nipy_spectral')
plt.title('Molecular Extinction')
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size="2%",pad=0.05)
plt.colorbar(im, cax=cax)
plt.savefig(outdir+'molecular_extinction.png')


mx = 0.95 * Ep.max()
extent = [x.min(),x.max(),z.min(),z.max()]
plt.figure(figsize=(11,7))
im = plt.imshow(Ep.T, vmin=0, vmax=mx, aspect='auto', extent=extent, cmap='nipy_spectral')
plt.title('Particulate Extinction')
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size="2%",pad=0.05)
plt.colorbar(im, cax=cax)
plt.savefig(outdir+'particulate_extinction.png')


mx = 0.95 * B.max()
extent = [x.min(),x.max(),z.min(),z.max()]
plt.figure(figsize=(11,7))
im = plt.imshow(B.T, vmin=0, vmax=mx, aspect='auto', extent=extent, cmap='nipy_spectral')
plt.title('Total Backscatter')
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size="2%",pad=0.05)
plt.colorbar(im, cax=cax)
plt.savefig(outdir+'total_backscatter.png')
