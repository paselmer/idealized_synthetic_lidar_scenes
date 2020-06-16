""" 

This code creates an idealized synthetic lidar scene using input from a
CSV file.

This code outputs to an HDF5 file containing the following parameters...

  Molecular_Extinction
  Molecular_Backscatter
  Particulate_Extinction
  Particulate_Backscatter

"""

import numpy as np
import matplotlib.pyplot as plt
import csv # for Ed's function
from scipy.interpolate import griddata
import h5py
import pdb

# Soft-coded I/O stuff so it's easily changeable
particulate_def_file = 'idealized_synthetic_lidar_scenes_v0.csv'
canonical_file = 'canonical_atmosphere_zinterpolated_langleycross_15m_no_header.csv'
wavelength = '532'
outdir = '.\\output\\'
output_tag = 'idealized_synthetic_lidar_scene_A'
outfile = outdir + output_tag + '.hdf5'

# Read in user-defined particulate parameters from idealized_lidar_scenes_v*.csv
with open(particulate_def_file, 'r') as f_obj:
    id = []
    x0 = []
    x1 = []
    S = []
    y0 = []
    y1 = []
    OD0 = []
    OD1 = []
    line = f_obj.readline()
    split_line = line.split(',')
    xtot_km = float(split_line[8].split('=')[1])
    dx_m = float(split_line[9].split('=')[1])
    ytop_km = float(split_line[10].split('=')[1])
    ybot_km = float(split_line[11].split('=')[1])
    dy_m = float(split_line[12].split('=')[1])
    while line:
        line = f_obj.readline()
        split_line = line.split(',')
        if len(split_line) <= 1: break
        id.append( split_line[0].strip() )
        x0.append( float(split_line[1]) )
        x1.append( float(split_line[2]) )
        S.append( float(split_line[3]) )
        y0.append( float(split_line[4]) )
        y1.append( float(split_line[5]) )
        OD0.append( float(split_line[6]) )
        OD1.append( float(split_line[7]) ) 
        
# Define the grid using inputs from CSV file
dx_km = dx_m / 1e3
dy_km = dy_m / 1e3
xord = np.arange(0, xtot_km + dx_km/2, dx_km) 
yord = np.arange(ytop_km, ybot_km - dy_km/2, -1*dy_km)
#xv, yv = np.meshgrid(xord,yord)
nx = xord.shape[0]
ny = yord.shape[0]
# Convert 'lists' to arrays
x0 = np.asarray(x0)
x1 = np.asarray(x1)
S = np.asarray(S)
y0 = np.asarray(y0)
y1 = np.asarray(y1)
OD0 = np.asarray(OD0)
OD1 = np.asarray(OD1)
# Converting normalized distance coordinates to km
x0 = x0 * xtot_km
x1 = x1 * xtot_km


# Function taken from Ed Nowottnick's CPL-based lidar simulator code
def readCanonical(canonical_file, wavelength):
    with open(canonical_file,'rt') as f:
        data=csv.reader(f)
        ct = 0
        for row in data:
            if (ct == 0):
                header = row
            ct = ct+1
        
    # Initialize Arrays
    # -----------------
    beta_m = np.zeros(ct-1)
    sigma_m = np.zeros(ct-1)
    alt_can = np.zeros(ct-1)
    o3_mol_dens = np.zeros(ct-1)

    zloc = 0
    oloc = 4
    if (wavelength == '532'):
        bloc = 10
        sloc = 11
    if (wavelength == '1064'):
        bloc = 12
        sloc = 13

    with open(canonical_file,'rt') as f:
        data=csv.reader(f)
        ct = 0
        for row in data:
            if (ct > 0):
                beta_m[ct-1] = float(row[bloc])
                sigma_m[ct-1] = float(row[sloc])
                o3_mol_dens[ct-1] = float(row[oloc])
                alt_can[ct-1] = float(row[zloc])
            ct = ct+1
    return (beta_m, sigma_m, alt_can, o3_mol_dens)


# Pull in the Rayleigh information from a canonical file
beta_m, sigma_m, alt_can, o3_mol_dens = readCanonical(canonical_file, wavelength)

# Put all the input data onto the output grid

# Regridding the canonical data
beta_m_regrid = griddata(alt_can,beta_m,(yord),method='cubic')
sigma_m_regrid = griddata(alt_can,sigma_m,(yord),method='cubic') # I believe this is extinction
o3_mol_dens_regrid = griddata(alt_can,o3_mol_dens,(yord),method='cubic')
a = np.where(np.isnan(beta_m_regrid))
beta_m_regrid[a] = 0.
sigma_m_regrid[a] = 0.
o3_mol_dens_regrid[a] = 0.

# Define gridded output variables
beta_m_2D = np.tile(beta_m_regrid, (nx, 1))
sigma_m_2D = np.tile(sigma_m_regrid, (nx, 1))
Bp = np.zeros((nx,ny))
Extp = np.zeros((nx,ny))

# This loops through each of the input layers, which are rows in the CSV file
for x0_,x1_,S_,y0_,y1_,OD0_,OD1_ in zip(x0, x1, S, y0, y1, OD0, OD1):
    
    i0 = np.argmin(np.abs(xord - x0_))
    i1 = np.argmin(np.abs(xord - x1_))
    j1 = np.argmin(np.abs(yord - y0_))
    j0 = np.argmin(np.abs(yord - y1_))
    nx_ = i1-i0+1
    ny_ = j1-j0+1
    ODs = np.linspace(OD0_, OD1_, nx_)
    Ext = ODs / (y1_ - y0_)
    B = Ext / S_
    Bp[i0:i1+1, j0:j1+1] = np.tile(B[:],(ny_,1)).transpose()
    Extp[i0:i1+1, j0:j1+1] = np.tile(Ext[:],(ny_,1)).transpose()
    
    
# Output data to an HDF5 file
h5_obj = h5py.File(outfile, 'w')
nbins_dset = h5_obj.create_dataset('nbins', (1,), maxshape=(None,), dtype=np.int32)
nprofs_dset = h5_obj.create_dataset('nprofs', (1,), maxshape=(None,), dtype=np.int32)
distance_km_dset = h5_obj.create_dataset('distance_km', (nx,), maxshape=(nx,), dtype=np.float64)
bin_alt_array_km_dset = h5_obj.create_dataset('bin_alt_array_km', (ny,), maxshape=(ny,), dtype=np.float64)
Bm_dset = h5_obj.create_dataset('Bm', (nx,ny), maxshape=(nx,ny), dtype=np.float64)
sigma_m_dset = h5_obj.create_dataset('sigma_m', (nx,ny), maxshape=(nx,ny), dtype=np.float64)
Bp_dset = h5_obj.create_dataset('Bp', (nx,ny), maxshape=(nx,ny), dtype=np.float64)
sigma_p_dset = h5_obj.create_dataset('sigma_p', (nx,ny), maxshape=(nx,ny), dtype=np.float64)
#
nbins_dset[:] = ny
nprofs_dset[:] = nx
distance_km_dset[:] = xord[:]
bin_alt_array_km_dset[:] = yord[:]
Bm_dset[:,:] = beta_m_2D[:,:]
sigma_m_dset[:,:] = sigma_m_2D[:,:]
Bp_dset[:,:] = Bp[:,:]
sigma_p_dset[:,:] = Extp[:,:]
h5_obj.close()

