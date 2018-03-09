import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable


from utils import Utils

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Generator(nn.Module):
    utils = Utils()
    
    def __init__(self, args={}):
        """
        args should contains:
            cube_len: int, default 64
            latent_vector_size: int, default 200
            bias_flag: bool, default True
        """
        super(Generator, self).__init__()
        
        self.cube_len = args["cube_len"] if "cube_len" in args else 64
        self.latent_vector_size = args["latent_vector_size"] if "latent_vector_size" in args else 200
        self.bias_flag = args["bias_flag"] if "bias_flag" in args else True
        firstLayer_padding = (0, 0, 0) if self.cube_len == 64 else (1, 1, 1)
        
        if self.cube_len not in {32, 64}:
            self.utils.error("Invalid cube_len")
        
        self.layer1_deconv = nn.Sequential(
                nn.ConvTranspose3d(self.latent_vector_size, self.cube_len*8, kernel_size=4, stride=2, bias=self.bias_flag, padding=firstLayer_padding),
                nn.BatchNorm3d(self.cube_len*8),
                nn.ReLU()
            )
        
        self.layer2_deconv = nn.Sequential(
                nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.BatchNorm3d(self.cube_len*4), 
                nn.ReLU()
            )
        
        self.layer3_deconv = nn.Sequential(
                nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.BatchNorm3d(self.cube_len*2), 
                nn.ReLU()
            )
        
        self.layer4_deconv = nn.Sequential(
                nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.BatchNorm3d(self.cube_len), 
                nn.ReLU()
            )
        
        self.layer5 = nn.Sequential(
                nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.Sigmoid()
            )
        
    def forward(self, latent_vector):
        out = latent_vector.view(-1, self.latent_vector_size, 1, 1, 1)
        print(out.size())
        out = self.layer1_deconv(out)
        print(out.size())
        out = self.layer2_deconv(out)
        print(out.size())
        out = self.layer3_deconv(out)
        print(out.size())
        out = self.layer4_deconv(out)
        print(out.size())
        out = self.layer5(out)
        print(out.size())
        
        return out

class Discriminator(nn.Module):
    uilts = Utils()
    
    def __init__(self, args={}):
        """
        args should contains:
            cube_len: int, default 64
            latent_vector_size: int, default 200
            bias_flag: bool, default True
        """
        self.cube_len = args["cube_len"] if "cube_len" in args else 64
        self.latent_vector_size = args["latent_vector_size"] if "latent_vector_size" in args else 200
        self.bias_flag = args["bias_flag"] if "bias_flag" in args else True
        
        self.layer1_conv = nn.Sequential(
                nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.BatchNorm3d(self.cube_len), 
                nn.LeakyReLU()
            )
        
        self.layer2_conv = nn.Sequential(
                nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.BatchNorm3d(self.cube_len*2), 
                nn.LeakyReLU()
            )
        
        self.layer3_conv = nn.Sequential(
                nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.BatchNorm3d(self.cube_len*4), 
                nn.LeakyReLU()
            )
        
        self.layer4_conv = nn.Sequential(
                nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.BatchNorm3d(self.cube_len*8), 
                nn.LeakyReLU()
            )
        
        self.layer5 = nn.Sequential(
                nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=self.bias_flag, padding=(1, 1, 1)), 
                nn.Sigmoid()
            )
        
    def forward(self, input_model):
        out = input_model.view(-1, 1, self.cube_len, self.cube_len, self.cube_len)
        print(out.size())
        out = self.layer1_conv(out)
        print(out.size())
        out = self.layer2_conv(out)
        print(out.size())
        out = self.layer3_conv(out)
        print(out.size())
        out = self.layer4_conv(out)
        print(out.size())
        out = self.layer5(out)
        print(out.size())
        
        return out

if __name__ == "__main__":
    a = Generator()
    b = Discriminator()
    