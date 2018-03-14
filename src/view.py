'''
Created on Mar 5, 2018

@author: micou
'''
import os
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D

from utils import Utils
#from dataIO import DataIO
from setting import VOXEL_MODEL_ROOTPATH

class View(Utils):
    def visual_dataGraph(self, listOfRecords, showWindow = True, saveImage=False, figure=None, savefile_dest="./temp.png"):
        """
        listOfRecords should be a list of lists
        list should have this structure [[x axis], [[y axis], [y axis], ......]]
        x axis can be empty list. Then will use y axis to determine how long is x
        will try to use a figure to display all graphs
        """
        row_graphs = math.ceil(math.sqrt(len(listOfRecords)))
        col_graphs = math.ceil(len(listOfRecords)/row_graphs)
        
        if saveImage and not showWindow:
            if os.path.isfile(savefile_dest):
                self.warn("Destination file ", savefile_dest, " exist, will skip it.")
                return
            elif os.path.isdir(savefile_dest):
                self.warn("Destination is a directory, no filename given")
                return
            
        if figure == None:
            require_close = True
            fg = plt.figure(figsize=(row_graphs*3, col_graphs*3*0.618))
        else:
            require_close = False
            fg = figure
            fg.clf()
            fg.set_size_inches((row_graphs*3, col_graphs*3*0.618))
        
        for n, r in enumerate(listOfRecords, 1):
            ax = fg.add_subplot(col_graphs, row_graphs, n)
            if len(r[0]) == 0:
                for y in r[1]:
                    ax.plot(y)
            else:
                for y in r[1]:
                    ax.plot(r[0], y[:len(r[0])])
        
        if showWindow:
            plt.ion()
            fg.show()
            plt.pause(1)
        if saveImage:
            fg.savefig(savefile_dest, dpi=100)
            
        if require_close:
            plt.close(fg)
        
    
    def save_voxelImage(self, input_args):
        """
        if this function called, pyplot will use backend mode to run
        input_args should have (model, dest_filepath)
        """
        plt.switch_backend('agg')
        self.visual_voxel(input_args[0], showWindow=False, saveImage=True, savefile_dest=input_args[1])
    
    def visual_voxel(self, voxel_ndarray, showWindow=True, saveImage=False, figure=None, savefile_dest="./temp.png"):
        """
        savefile_dest only required when saveImage is True
        if showWindow, process will pause for 1 seconds
        can only visualize 3D object
        """
        if saveImage and not showWindow:
            if os.path.isfile(savefile_dest):
                self.warn("Destination file ", savefile_dest, " exist, will skip it.")
                return
            elif os.path.isdir(savefile_dest):
                self.warn("Destination is a directory, no filename given")
                return
        
        if figure==None:
            require_close = True
            fig = plt.figure(figsize=(6, 6*0.8))
        else:
            require_close = False
            fig=figure
            fig.clf()
            fig.set_size_inches((6, 6*0.8))
        
        # Reference: https://matplotlib.org/gallery/mplot3d/voxels_rgb.html
        # Prepare color for it
        def midpoints(x):
            sl = ()
            for _ in range(x.ndim):
                x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
                sl += np.index_exp[:]
            return x
        r, g, b = np.indices(tuple((voxel_ndarray.shape+np.ones(voxel_ndarray.ndim)).astype(np.int))) / voxel_ndarray.shape[0] 
        rc = midpoints(r)
        gc = midpoints(g)
        bc = midpoints(b)
        colors = np.zeros(voxel_ndarray.shape + (3,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc
        colors = np.clip(colors, 0, 1)
        
        ax=fig.gca(projection='3d')
        # Use z axis as bottom plane
        ax.voxels(r, g, b, np.transpose(voxel_ndarray, (0, 2, 1)),
                  facecolors=0.5*colors + 0.1,
                  edgecolors=np.clip(2*colors - 0.5, 0, 1),
                  linewidth=0.5)
        
        if showWindow:
            plt.ion()
            fig.show()
            plt.pause(1)
        if saveImage:
            fig.savefig(savefile_dest, dpi=100)
        # If figure is not given, must clear figure before exit to avoid memory leakage
        if require_close:
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataIO module to tranform mesh model to voxel model")
    parser.add_argument("-p", "--path", default=VOXEL_MODEL_ROOTPATH, dest="rootpath", help="Root path to unzipped shapeNet folder")
    parser.add_argument("-n", "--number", default=8, type=int, dest="processors_number", help="Define number of processors to do transform")
    args = parser.parse_args()
    
    test = DataIO(args.rootpath)
    model_generator = test.get_voxmodel_generator("", True)
    multi_pool = Pool(args.processors_number)
    temp_list = []
    view_obj = View()
    done = 0
    # Start to read data from model_generator. Each processor will process about 20 images per round
    for model in model_generator:
        if len(temp_list) < args.processors_number * 20:
            model_dir, model_file = os.path.split(model[1]["filepath"])
            model_file_prefix, _ = os.path.splitext(model_file)
            temp_list.append((model[0], os.path.join(model_dir, model_file_prefix+".png")))
        else:
            Utils().info("Processing one list, already done: ", done)
            multi_pool.map(view_obj.save_voxelImage, temp_list) #View().visual_voxel(x[0],False, x[1])
            done += len(temp_list)
            temp_list = []
            multi_pool.close()
            multi_pool.terminate()
            multi_pool = Pool(args.processors_number)
    Utils().info("Exist main loop, left: ",str(len(temp_list)))
    if len(temp_list) > 0:
        multi_pool.map(view_obj.save_voxelImage, temp_list)
    
