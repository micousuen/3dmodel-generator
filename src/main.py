'''
Created on Mar 12, 2018

@author: micou
'''
import os
import argparse
from multiprocessing import Pool

import setting
from training import Train
from view import View
from dataIO import DataIO
from utils import Utils

if __name__ == "__main__":
    ut = Utils()
    
    args = argparse.ArgumentParser(description="3d model generator")
    
    # Choose to run dataIO to pre-process models
    args.add_argument("-D", "--dataio", dest="D", action="store_true", help="Run dataIO module to pre-process models")
    # Choose to run view to visualize exist models
    args.add_argument("-V", "--view", dest="V", action="store_true", help="Run view module to visualize exist mat files")
    # Choose to run trianing to visualize exist model. 
    # This is default action if none of these three is chosen
    args.add_argument("-T", "--training", dest="T", action="store_true", help="Run training module to train network, default choice")
    
    # some parameters for modules, some common parameters for dataIO and view
    args.add_argument("-j", dest="j", default=1, help="Use multicores to run module, only valid for dataIO and view and default is 1")
    args.add_argument("-p", "--path", dest="p", default=setting.VOXEL_MODEL_ROOTPATH, help="path that stores data. default value is variable VOXEL_MODEL_ROOTPATH in setting.py. ")
    args.add_argument("-m", "--modeltree", dest="m", action="store_true", help="Option that allow dataIO to build model tree")
    args.add_argument("-o", "--outputPath", dest="o", default="./voxelModels", help="output path to store dataIO result")
    # below are arguments for training, if choose D or V mode, these parameters won't take effect
    args.add_argument("-i", "--bias_flag2false", dest="i", action="store_false", help="flag to set bias, default is True")
    args.add_argument("-n", "--normalLatent", dest="n", action="store_true", help="Use normal sampled latent vector, default is to use uniform sampled vector")
    args.add_argument("-s", "--softLabel_False", dest="s", action="store_false", help="Stop use soft label. Default is True")
    args.add_argument("-r", "--resume", dest="r", action="store_false", help="resume from last checkpoint, default is true")
    args.add_argument("-v", "--voxel_len", dest="v", type=int, default=64, help="voxel space size, default is 64")
    args.add_argument("-l", "--latent_vector_size", dest="l", type=int, default=200, help="latent vector size, default is 200")
    args.add_argument("-b", "--batchSize", dest="b", default=60, type=int, help="set batch size, default is 60")
    args.add_argument("-c", "--categories", dest="c", default=[], nargs="*", help="category to train, default is to train all")
    args.add_argument("-k", "--leakyrelu_value", dest="k", default=0.2, type=float, help="leakyrelu value used in discriminator")
    args.add_argument("-g", "--generator_learningRate", dest="g", default=0.0025, type=float, help="Generator Learning Rate, default is 0.0025")
    args.add_argument("-d", "--discriminator_learningRate", dest="d", default=0.00001, type=float, help="Discriminator Learning Rate, default is 0.00001")
    args.add_argument("-t", "--discriminator_training_threshold", dest="t", default=0.8, type=float, help="Discriminator stop training threshold, default is 0.8")
    args.add_argument("-e", "--epoch_limit", dest="e", default=-1, type=int, help="set epoch limit, set to -1 to run forever. Default -1")
    args.add_argument("-z", "--checkpoint_filename", dest="z", default="./checkpoint.pth.tar", help="path to store checkpoint")
    args.add_argument("-x", "--g_d_training_rate", dest="x", default=1, help="generator discriminator training rate, default 1, don't recommend to change it")
    args.add_argument("-y", "--save_model_interval", dest="y", default=50, type=int, help="save model every this number of intervals.")
    args.add_argument("-u", "--visual_status", dest="u", action="store_true", help="use window to visual status, cannot run in backend mode")    
    
    ag = args.parse_args()
    if [ag.D, ag.V, ag.T].count(True) > 1:
        ut.error("Don't accept more than one running mode select, only choose one between support mode")
    if not (ag.D or ag.V or ag.T):
        # if no one chosen, just run training module
        ag.T = True
    
    if ag.D:
        model_tree="./model_tree.json" if ag.m else "" 
        runDataIO=DataIO(ag.p, ag.c, model_tree)
        runDataIO.transform_saveVoxelFiles("", dim=ag.v, multiprocess=ag.j, dest_samedir=False, dest_dir=ag.o)
    elif ag.V:
        test = DataIO(ag.p)
        model_generator = test.get_voxmodel_generator("", True)
        multi_pool = Pool(ag.j)
        temp_list = []
        view_obj = View()
        done = 0
        # Start to read data from model_generator. Each processor will process about 20 images per round
        for model in model_generator:
            if len(temp_list) < ag.j * 20:
                model_dir, model_file = os.path.split(model[1]["filepath"])
                model_file_prefix, _ = os.path.splitext(model_file)
                temp_list.append((model[0], os.path.join(model_dir, model_file_prefix+".png")))
            else:
                Utils().info("Processing one list, already done: ", done)
                multi_pool.map(view_obj.save_voxelImage, temp_list) #View().visual_voxel(x[0],False, x[1])
                done += len(temp_list)
                temp_list = []
        Utils().info("Exist main loop, left: ",str(len(temp_list)))
        if len(temp_list) > 0:
            multi_pool.map(view_obj.save_voxelImage, temp_list)
    elif ag.T:
        arg_dict = {
                "cube_len": ag.v, 
                "latent_vector_size": ag.l, 
                "latent_vector_type": "uniform" if ag.n else "normal", 
                "bias_flag": ag.i, 
                "batch_size": ag.b, 
                "generator_learningRate": ag.g, 
                "discriminator_learningRate": ag.d,
                "discriminator_training_threshold": ag.t,
                "adam_beta": (0.5, 0.5), 
                "resume": ag.r, 
                "epoch_limit": ag.e,
                "checkpoint_filename": ag.z,
                "training_categories": " ".join(ag.c),
                "data_rootpath": ag.p,
                "leakyrelu_value": ag.k,
                "soft_label": ag.s, 
                "g_d_training_rate": ag.x,
                "save_model_interval":  ag.y,
                "visual_status": ag.u
            }
        runTraining = Train(arg_dict)
        runTraining.train()