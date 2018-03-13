'''
Created on Mar 9, 2018

@author: micou
'''
import os
#import time
import numpy as np
#from itertools import count
from multiprocessing import Process, Pipe

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import dataIO
from utils import Utils
from dataIO import DataIO
from model import Generator, Discriminator

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Train(Utils):
    # pipeline for multiprocessing data preparation
    data_send, data_recv = Pipe()
    data_process = None
    epoch = 0
    iteration = 0
    
    def __init__(self, args={}):
        """
        args require:
            cube_len:                         int,              default 64
            latent_vector_size:               int,              default 200
            latent_vector_type:               str,              default "uniform",              in {"uniform","normal"}
            bias_flag:                        bool,             default True
            batch_size:                       int,              default 100
            training_categories:              str,              default ''                      which means all categories. use space to split different categories
            data_rootpath:                    str,              default VOXEL_MODEL_ROOTPATH in setting.py
            leakyrelu_value:                  float,            default 0.2
            soft_label:                       bool,             default False.                  Enable soft_label value, real -> [0.7, 1.2], fake -> [0, 0.3] 
            generator_learningRate:           float,            default 0.0025
            adam_beta:                        tuple of floats,  defualt (0.5, 0.5)
            discriminator_learningRate:       float,            default 0.00001
            discriminator_training_threshold: float,            default 0.8
            g_d_training_rate:                float,            default 1,                      recommend > 1
            resume:                           bool,             default True
            epoch_limit:                      int,              default -1.                     If -1 run forever and save currently best model
            checkpoint_filename:              str,              default './checkpoint.pth.tar'. Current best checkpoint file will have epoch number in front of this checkpoint file
            save_model_interval:              int,              default 50.                      Save model every interval iterations
        """
        
        self.args = {
                "cube_len": args["cube_len"] if "cube_len" in args else 64,
                "latent_vector_size": args["latent_vector_size"] if "latent_vector_size" in args else 200, 
                "latent_vector_type": args["latent_vector_type"] if "latent_vector_type" in args else "uniform", 
                "bias_flag": args["bias_flag"] if "bias_flag" in args else True, 
                "batch_size": args["batch_size"] if "batch_size" in args else 100, 
                "generator_learningRate": args["generator_learningRate"] if "generator_learningRate" in args else 0.0025, 
                "discriminator_learningRate": args["discriminator_learningRate"] if "discriminator_learningRate" in args else 0.00001,
                "discriminator_training_threshold": args["discriminator_training_threshold"] if "discriminator_training_threshold" in args else 0.80,
                "adam_beta": args["adam_beta"] if "adam_beta" in args else (0.5, 0.5), 
                "resume": args["resume"] if "resume" in args else True, 
                "epoch_limit": args["epoch_limit"] if "epoch_limit" in args else -1,
                "checkpoint_filename": args["checkpoint_filename"] if "checkpoint_filename" in args else "./checkpoint.pth.tar",
                "training_categories": args["training_categories"].split() if "training_categories" in args else [],
                "data_rootpath": args["data_rootpath"] if "data_rootpath" in args else dataIO.VOXEL_MODEL_ROOTPATH,
                "leakyrelu_value": args["leakyrelu_value"] if "leakyrelu_value" in args else 0.2,
                "soft_label": args["soft_label"] if "soft_label" in args else False, 
                "g_d_training_rate": args["g_d_training_rate"] if "g_d_training_rate" in args else 1,
                "save_model_interval":  args["save_model_interval"] if "save_model_interval" in args else 50
            }
        
        # Display settings 
        for key in self.args:
            self.info("<Train> Set <{0:>35}> to <{1}>".format(key, self.args[key]))
            
        # Prepare data reader. It will start to prepare for data immediately. data_generator will yield nothing when meet the end of epoch
        # Use dataIO to shuffle data
        self.data_generator = self._get_tensorData() # a tensor data generator
            
        self.generator = Generator(self.args)
        self.discriminator = Discriminator(self.args)
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
        self.generator_optim = optim.Adam(self.generator.parameters(), lr=self.args["generator_learningRate"], betas=self.args["adam_beta"] )
        self.discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=self.args["discriminator_learningRate"], betas=self.args["adam_beta"])
        self.loss_function = nn.BCELoss()
        
        # Try to resume from checkpoint, could fail and if failed to load checkpoint, will start from very beginning
        if self.args["resume"]:
            if self._load_status(self.args["checkpoint_filename"]):
                # Succesfully load from checkpoint
                self.info("load checkpoint")
            else:
                self.warn("load error, start from very beginning")
        
        
    
    def train(self):
        saved_iteration = self.iteration
        saved_epoch = 0
        last_epoch = self.epoch
        # If meet the epoch limit, our data_generator will automatically exit
        for batch in self.data_generator:
            self.epoch += batch[1] # if finished one epoch, flag at this position will be 1 (or greater if more than 1 epoch in one batch), otherwise this flag will be 0
            self.iteration += 1
            realModels = batch[0] # A tensor on cpu or gpu, based one data type
            
            # Drop incompatible batch
            if realModels.size()[0] != self.args["batch_size"]:
                self.info("Drop incompatible batch with batch size <{0}>".format(realModels.size()[0] ))
                continue
            
            # Put realModels to gpu, and calculate at there
            if torch.cuda.is_available():
                if not realModels.is_cuda:
                    realModels = realModels.cuda()
            
            # Prepare label factors, used for soft_label
            if self.args["soft_label"]:
                realLabelFactor = Tensor(self.args["batch_size"]).uniform_(0.7, 1.2)
                fakeLabelFactor = Tensor(self.args["batch_size"]).uniform_(0, 0.3)
            else:
                realLabelFactor = torch.ones(self.args["batch_size"]).cuda() if torch.cuda.is_available() else torch.ones(self.args["batch_size"])
                fakeLabelFactor = torch.zeros(self.args["batch_size"]).cuda() if torch.cuda.is_available() else torch.zeros(self.args["batch_size"]).cuda()
            
            def train_discriminator_vanilla():
                #************* Train Discriminator ************
                latent_vectors = self._get_latentVectors(self.args)
                realDataLabel = torch.ones(self.args["batch_size"]).cuda() if torch.cuda.is_available() else torch.ones(self.args["batch_size"])
                fakeDataLabel = torch.ones(self.args["batch_size"]).cuda() if torch.cuda.is_available()  else torch.ones(self.args["batch_size"])
                
                d_eval_real = self.discriminator(self.transform_var(realModels)).squeeze()
#                 print(tuple(d_eval_real.data))
#                 print(tuple(realDataLabel * realLabelFactor))
                d_real_loss = self.loss_function(d_eval_real, self.transform_var(realDataLabel * realLabelFactor))
                
                fakeModels = self.generator(latent_vectors)
                d_eval_fake = self.discriminator(self.transform_var(fakeModels)).squeeze()
#                 print(tuple(d_eval_fake.data))
#                 print(tuple(fakeDataLabel * fakeLabelFactor))
                d_fake_loss = self.loss_function(d_eval_fake, self.transform_var(fakeDataLabel * fakeLabelFactor))
                
                d_loss = d_real_loss + d_fake_loss
                d_real_accuracy = d_eval_real.data.ge(0.5).float()
                d_fake_accuracy = d_eval_fake.data.lt(0.5).float()
                d_accuracy = torch.mean(torch.cat((d_real_accuracy , d_fake_accuracy)), 0)
                
                if torch.is_tensor(d_accuracy):
                    d_accuracy = d_accuracy[0]
                
                if d_accuracy <= self.args["discriminator_training_threshold"]:
                    self.generator.zero_grad()
                    self.discriminator.zero_grad()
                    d_loss.backward()
                    self.discriminator_optim.step()
                
                return (d_accuracy, d_real_loss.data[0], d_fake_loss.data[0])
            
            def train_discriminator_mixup():
                #************* Train Discriminator use mixed up models ***********
                latent_vectors = self._get_latentVectors(self.args)
                # Generate random 1d data mask for real & fake data
                realDataLabel = torch.rand(self.args["batch_size"]).cuda() if torch.cuda.is_available() else torch.rand(self.args["batch_size"])
                realDataMask = self._get_fullMask(realDataLabel.ge(0.5).type(Tensor), realModels)
                fakeDataMask = (1 - realDataMask).cuda() if torch.cuda.is_available() else (1 - realDataMask)
                # Wrap label with variable so it can cal gradient
                mixedLabel = self.transform_var(realDataLabel * realLabelFactor + (1-realDataLabel) * fakeLabelFactor)
                
                # Generate fake models from generators
                fakeModels = self.generator(latent_vectors).squeeze()
                # Mix fake models with real models
                mixedModels = fakeModels.data * fakeDataMask + realModels * realDataMask
                modelEvaluation = self.discriminator(self.transform_var(mixedModels)).squeeze()
                # loss which can be used to train discriminator
                d_loss = self.loss_function(modelEvaluation, mixedLabel)
                d_real_accuracy = torch.mean(modelEvaluation.data.ge(0.5).float() * realDataLabel + modelEvaluation.data.lt(0.5).float() * (1-realDataLabel))
                
                # According to paper, only update discriminator when accuracy is below threshold
                if d_real_accuracy <= self.args["discriminator_training_threshold"]:
                    self.discriminator.zero_grad()
                    d_loss.backward()
                    self.discriminator_optim.step()
                    
                d_real_loss = self.loss_function(self.transform_var(modelEvaluation.data*realDataLabel), self.transform_var(mixedLabel.data*realDataLabel)).data[0]
                d_fake_loss = self.loss_function(self.transform_var(modelEvaluation.data*(1-realDataLabel)), self.transform_var(mixedLabel.data*(1-realDataLabel))).data[0]
                return (d_real_accuracy, d_real_loss, d_fake_loss)
                    
            def train_generator():
                #************* Train Generator ************
                latent_vectors = self._get_latentVectors(self.args)
                
                # Generate fake models from geneator
                fakeModels = self.generator(latent_vectors)
                fakeModelEvaluation = self.discriminator(fakeModels).squeeze()
                # Wrap label with variable so it can cal gradient
                realLabel = self.transform_var(torch.ones(self.args["batch_size"]).type(Tensor) * realLabelFactor)
                generator_loss = self.loss_function(fakeModelEvaluation, realLabel)
                
                g_confuse_average = torch.mean(fakeModelEvaluation.data)
                g_confuse_rate = torch.mean(fakeModelEvaluation.data.ge(0.5).float())
                
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                generator_loss.backward()
                self.generator_optim.step()
                return (g_confuse_average, generator_loss.data[0], g_confuse_rate)
            
            # Based on g_d_training_rate, train our model
            if self.args["g_d_training_rate"] >= 1:
                d_info = train_discriminator_vanilla()
                for _ in range(int(self.args["g_d_training_rate"])):
                    g_info = train_generator()
            else:
                g_info = train_generator()
                for _ in range(int(1/self.args["g_d_training_rate"])):
                    d_info = train_discriminator_vanilla()
            
            self.info("E {0:<3} I {1:<3} --> ".format(self.epoch, self.iteration), \
                      "D: accu <{0:>3.2f}%>, ge_loss <{3:.4f}>, rl_l <{1:.4f}>, fk_l <{2:.4f}> | ".format(d_info[0]*100, d_info[1], d_info[2], d_info[1]+d_info[2]),\
                      "G: pass rate <{2:>3.2f}%>, loss <{1:.4f}>, avg score <{0:.4f}>".format(g_info[0], g_info[1], g_info[2]*100))
            
            if self.epoch - last_epoch >= 1:
                if (self.epoch)%int(self.args["save_model_interval"]) == 0:
                    cp_dir, cp_file = os.path.split(self.args["checkpoint_filename"])
                    self._save_status(os.path.join(cp_dir, str(self.epoch)+"_"+cp_file))
                    self.info("Epoch Checkpoint filesaved at", os.path.join(cp_dir, str(self.epoch)+"_"+cp_file))
                    saved_epoch = self.epoch
                last_epoch = self.epoch
                self.iteration = 0 
                saved_iteration = 0
            if self.iteration - saved_iteration >= self.args["save_model_interval"]:
                self._save_status(self.args["checkpoint_filename"])
                self.info("Checkpoint filesaved at", self.args["checkpoint_filename"])
                saved_iteration = self.iteration
            
            
    def _get_fullMask(self, oneDimMask, tensorToMask):
        """
        oneDimMask should be a tensor or numpy or list, 
        tensorToMask_size should be tensor's size
        return mask will mask data in row, data type is FloatTensor in cpu or gpu, based on where is oneDimMask
        """
        if not torch.is_tensor(tensorToMask):
            self.error("in get_fullMask, tensorToMask should be a Tensor")
        if not torch.is_tensor(oneDimMask):
            if isinstance(oneDimMask, np.ndarray):
                oneDimMask = torch.from_numpy(oneDimMask)
            elif isinstance(oneDimMask, tuple) or isinstance(oneDimMask, list):
                oneDimMask = Tensor(oneDimMask)
            else:
                self.error("oneDimMask should be a 1D list, tuple or Tensor")
        if oneDimMask.dim() != 1 or oneDimMask.size()[0] != tensorToMask.size()[0]:
            self.error("oneDimMask should only have one dimension and its size should equal to tensorToMask row size, but we get {0} and {1}".format(oneDimMask.size(), tensorToMask.size()))
            
        result = oneDimMask.clone()
        # Change result dimensions
        result = result.view(-1, *[1 for _ in range(tensorToMask.dim()-1)])
        # Then expand value to all position
        result = result.expand(-1, *tuple(tensorToMask.size()[1:]))
        # Type of result is same as oneDimMask
        return result.type(oneDimMask.type())
        
    
    def _get_latentVectors(self, args):
        """
        args should have latent_vector_size, latent_vector_type, batch_size key, otherwise program will raise exception
        return Variable wrapped latent vectors
        """
        if not "latent_vector_size" in args or not "latent_vector_type" in args or not "batch_size" in args:
            self.error("arguments missing in get_latentVectors")
            return
        if not args["latent_vector_type"].lower() in ("uniform", "normal"):
            self.error("unknown latent_vector_type")
            return
        
        if args["latent_vector_type"].lower() == "normal":
            # Use normal distributed latent vectors between [0, 1], mean = 0.5, std = 0.33
            latent_vectors = self.transform_var(torch.clamp(FloatTensor(args["batch_size"], args["latent_vector_size"]).normal_(mean=0.5, std=0.33), 0, 1))
        elif args["latent_vector_type"].lower() == "uniform":
            # Use uniform distributed latent vectors between [0, 1]
            latent_vectors = self.transform_var(FloatTensor(args["batch_size"], args["latent_vector_size"]).uniform_(0, 1))
        else:
            # If latent_vector_type is unknown, use all zeros
            latent_vectors = self.transform_var(torch.zeros(args["batch_size"], args["latent_vector_size"]))
        return latent_vectors
    
    def transform_var(self, obj):
        """
        obj must be a tensor. If not program will raise exception
        Wrap tensor with Variable, based on obj tensor type
        """
        if isinstance(obj, Variable):
            # If Variable given, directly return it in case this function is called twice
            return obj
        if not torch.is_tensor(obj):
            self.error("obj given to transform_var is not a tensor")
        if torch.cuda.is_available() and not obj.is_cuda:
            obj = obj.cuda()
        return Variable(obj)
    
    def _get_tensorInputData(self, data_pipeline):    
        """
        Use pipeline to receive data. This should start as another process.
        Once it finished data reading and send it out, it will get blocked and wait for True from pipe
        """
        self.info("Start to prepare tensor data")
        self.dataIOPort = DataIO(self.args["data_rootpath"], self.args["training_categories"] )
        self.dataGenerator = self.dataIOPort.get_batchmodels("", self.args["batch_size"], self.args["epoch_limit"], True, "voxel")
        for raw_data in self.dataGenerator:
            prepared_data = np.asarray(raw_data[0], dtype=np.float)
            while not data_pipeline.recv():
                pass
            data_pipeline.send(prepared_data)
            data_pipeline.send(raw_data[1])
        # Finished data transfer, wait for main process to get data and send out bye signal
        data_pipeline.recv()
        data_pipeline.send(False)
        data_pipeline.send(0)
        data_pipeline.close()
        
    def _get_tensorData(self):
        """
        Start another process to read data and make it read. 
        This process will get data ready when you are doing other things
        """
        self.data_process = Process(target=Train._get_tensorInputData, args=(self, self.data_send))
        self.data_process.start()
        while True:
            self.data_recv.send(True)
            d = self.data_recv.recv()
            e = self.data_recv.recv()
            if isinstance(d, np.ndarray):
                yield (torch.FloatTensor(d), e)
            else:
                self.info("Meet the end of data")
                break
        self.data_process.terminate()
        self.data_process = None
        
    def _save_status(self, save_filename):
        try:
            self.checkpoint = {
                    "generator_model":self.generator.state_dict(), 
                    "generator_optim":self.generator_optim.state_dict(), 
                    "discriminator_model":self.discriminator.state_dict(), 
                    "discriminator_optim":self.discriminator_optim.state_dict(),
                    "epoch_count":self.epoch
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
            self.epoch = self.checkpoint["epoch_count"]
        except:
            self.warn("cannot load models, check models")
            return False
        return True
    
    def error(self, *args):
        if isinstance(self.data_process, Process):
            self.data_process.terminate()
        super(Train, self).error(*args)
    
if __name__ == "__main__":
        a = Train({"epoch_limit":-1, "batch_size":60, "cube_len":64, "data_rootpath":"./voxelModels", "soft_label":True, "training_categories":"03001627"})
        a.train()
            
