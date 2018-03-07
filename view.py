'''
Created on Mar 5, 2018

@author: micou
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import utils
import setting

class view(utils):
    def visual_voxel(self, voxel_ndarray, blocking=True):
        fig = plt.figure()
        colors=np.empty(voxel_ndarray.shape, dtype=object)
        ax=fig.gca(projection='3d')
        ax.voxels(voxel_ndarray, facecolors=colors, edgecolor='k')
        if blocking:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.001)

if __name__ == "__main__":
    x, y, z = np.indices((8, 8, 8))
    
    a = np.array(((1, 0, 1),(0, 1, 1),(0, 0, 1)))
    cube_t = np.zeros((8, 8, 8), dtype=bool)
    print(cube_t)
    for i in range(3):
        for j in range(3):
            if a[i][j] == 1:
                cube_t = cube_t | ((x==i) & (y ==j) & (z==0))
    print(cube_t)
    cube1 = (x<3) & (y<3) & (z<3)
    cube2 = (x>=5) & (y>=5) & (z >=5)
    link = abs(x-y) + abs(y-z) + abs(z-x) <= 2
    
    voxels = cube1 | cube2 | link
    colors = np.empty(voxels.shape, dtype=object)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(cube_t, facecolors=colors, edgecolor='k')
    
    plt.show()
