# griddata test
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.interpolate import griddata

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
points = np.random.rand(1000, 2)
values = func(points[:,0], points[:,1])

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')

plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
plt.show()
pdb.set_trace()
