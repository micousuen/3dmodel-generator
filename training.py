'''
Created on Mar 9, 2018

@author: micou
'''
import os
import numpy as np
from itertools import count
from multiprocessing import Process, Pipe

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

import dataIO
from utils import Utils
from dataIO import DataIO
from model import Generator, Discriminator

# if gpu is to be used
use_cuda = False # torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Train(Utils):
    data_send, data_recv = Pipe()
    
    def __init__(self, args={}):
        """
        args require:
            cube_len: int, default 64
            latent_vector_size: int, default 200
            bias_flag: bool, default True
            batch_size: int, default 100
            training_categories: str, default '' which means all categories. use space to split different categories
            data_rootpath: str, default VOXEL_MODEL_ROOTPATH in setting.py
            leakyrelu_value: float, default 0.2
            generator_learningRate: float, default 0.0025
            discriminator_learningRate: float, default 0.00001
            resume: bool, default True
            epoch_limit: int, default -1. If -1 run forever and save currently best model
            checkpoint_filename: string, default './checkpoint.pth.tar'. Current best checkpoint file will have epoch number in front of this checkpoint file
        """
        
        self.cube_len = args["cube_len"] if "cube_len" in args else 64
        self.latent_vector_size = args["latent_vector_size"] if "latent_vector_size" in args else 200
        self.bias_flag = args["bias_flag"] if "bias_flag" in args else True
        self.batch_size = args["batch_size"] if "batch_size" in args else 100
        self.generator_learningRate = args["generator_learningRate"] if "generator_learningRate" in args else 0.0025
        self.discriminator_learningRate = args["discriminator_learningRate"] if "discriminator_learningRate" in args else 0.00001
        self.adam_beta = args["adam_beta"] if "adam_beta" in args else 0.5
        self.resume = args["resume"] if "resume" in args else True
        self.epoch_limit = args["epoch_limit"] if "epoch_limit" in args else -1
        self.checkpoint_filename = args["checkpoint_filename"] if "checkpoint_filename" in args else "./checkpoint.pth.tar"
        self.training_categories = args["training_categories"].split() if "training_categories" in args else []
        self.data_rootpath = args["data_rootpath"] if "data_rootpath" in args else dataIO.VOXEL_MODEL_ROOTPATH
        self.leakyrelu_value = args["leakyrelu_value"] if "leakyrelu_value" in args else 0.2
        self.args = {
                "cube_len":self.cube_len, 
                "latent_vector_size":self.latent_vector_size, 
                "bias_flag":self.bias_flag, 
                "batch_size":self.batch_size, 
                "generator_learningRate": self.generator_learningRate, 
                "discriminator_learningRate": self.discriminator_learningRate,
                "adam_beta": self.adam_beta, 
                "resume": self.resume, 
                "epoch_limit": self.epoch_limit,
                "checkpoint_filename": self.checkpoint_filename,
                "training_categories": self.training_categories,
                "data_rootpath": self.data_rootpath,
                "leakyrelu_value":self.leakyrelu_value
            }
        
        # Display settings 
        for key in self.args:
            self.info("<Train> Set {0} to {1}".format(key, self.args[key]))
            
        # load data from dataIO
        self.data = DataIO()
            
        self.generator = Generator(self.args)
        self.discriminator = Discriminator(self.args)
        self.generator_optim = optim.Adam(self.generator.parameters(), lr=self.args["generator_learningRate"], betas=self.args["adam_beta"] )
        self.discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=self.args["discriminator_learningRate"], betas=self.args["adam_beta"])
        self.loss_function = nn.BCELoss()
        
        # Resume from checkpoint
        if self.resume:
            if self._load_status(self.checkpoint_filename):
                # Succesfully load from checkpoint
                self.info("load checkpoint")
            else:
                self.warn("load error, start from very beginning")
        
        for epoch in count():
            # If meet the epoch limit, exit training
            if self.epoch_limit > 0 and epoch >= self.epoch_limit:
                break
            break
        
    
    def _get_tensorInputData(self, data_pipeline):    
        """
        Use pipeline to receive data. This should start as another process.
        Once it finished data reading and send it out, it will get blocked and wait for True from pipe
        """
        self.info("Start to prepare tensor data")
        self.dataPort = DataIO(self.data_rootpath, self.training_categories)
        self.dataGenerator = self.dataPort.get_batchmodels("", self.batch_size, self.epoch_limit, True, "voxel")
        for raw_data in self.dataGenerator:
            prepared_data = FloatTensor(np.asarray(raw_data, dtype=np.float))
            while not data_pipeline.recv():
                pass
            data_pipeline.send(prepared_data)
        # Finished data transfer, wait for main process to get data and send out bye signal
        data_pipeline.recv()
        data_pipeline.send(False)
        
    def get_tensorData(self):
        """
        Start another process to read data and make it read. 
        This process will get data ready when you are doing other things
        """
        p = Process(target=Train._get_tensorInputData, args=(self, self.data_send))
        p.start()
        while True:
            self.data_recv.send(True)
            d = self.data_recv.recv()
            if isinstance(d, FloatTensor):
                yield d
            else:
                self.info("Meet the end of data")
                break
        
    def _save_status(self, save_filename):
        try:
            self.checkpoint = {
                    "generator_model":self.generator.state_dict(), 
                    "generator_optim":self.generator_optim.state_dict(), 
                    "discriminator_model":self.discriminator.state_dict(), 
                    "discriminator_optim":self.discriminator_optim.state_dict(),
                }
        except:
            self.warn("cannot get checkpoint from models")
            return False
        try:
            torch.save(self.checkpoint, save_filename)
        except:
            self.warn("in _save_status, save status error, check save_filename")
            return False
        return True
        
    def _load_status(self, save_filename):
        try:
            self.checkpoint = torch.load(save_filename)
        except:
            self.warn("cannot load checkpoint, check checkpoint file path and file status")
            return False
        try:
            self.generator.load_state_dict(self.checkpoint["generator_model"])
            self.generator_optim.load_state_dict(self.checkpoint["generator_optim"])
            self.discriminator.load_state_dict(self.checkpoint["discriminator_model"])
            self.discriminator_optim.load_state_dict(self.checkpoint["discriminator_optim"])
        except:
            self.warn("cannot load models, check models")
            return False
        return True
    
if __name__ == "__main__":
    a = Train({"epoch_limit":1})
    gene = a.get_tensorData()
    for d in gene:
        print(d.size())
            