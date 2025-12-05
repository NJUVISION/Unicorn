# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-07

import os, sys, time
import torch
import MinkowskiEngine as ME
import numpy as np
sys.path.append(os.path.split(__file__)[0])
from resnet import ResNetBlock
from transformer import TransformerBlock


######################## lossy attribute ########################
class Backbone(torch.nn.Module):
    """
    """
    def __init__(self, scale=1,
                in_channels=3, channels=128, out_channels=128, 
                block_type='resnet', block_layers=3, kernel_size=3,  kernel_size_scale=None, knn=16, 
                activation='relu', stride=[2,2,2]):
        super().__init__()
        if kernel_size_scale==None: kernel_size_scale = kernel_size
        self.linear_in = ME.MinkowskiLinear(
            in_channels, channels, bias=True)
        self.linear_out = ME.MinkowskiLinear(
            channels, out_channels, bias=True)

        self.scale = scale 
        self.block_type = block_type
        self.scaler_list = torch.nn.ModuleList()
        self.block_list = torch.nn.ModuleList()

        for i in range(abs(scale)):
            if scale>0: self.scaler_list.append(ME.MinkowskiConvolution(
                in_channels=channels, out_channels=channels, 
                kernel_size=kernel_size_scale, stride=stride, bias=True, dimension=3))
            if scale<0: self.scaler_list.append(ME.MinkowskiConvolutionTranspose(
                in_channels=channels, out_channels=channels, 
                kernel_size=kernel_size_scale, stride=stride, bias=True, dimension=3))

        for i in range(abs(scale)+1):
            if block_type in ['linear']:
                self.block_list.append(torch.nn.Sequential(
                    ME.MinkowskiLinear(channels, channels),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiLinear(channels, channels),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiLinear(channels, channels)))
            elif block_type in ['conv', 'resnet']:
                self.block_list.append(ResNetBlock(
                    channels=channels, 
                    kernel_size=kernel_size, 
                    block_layers=block_layers, 
                    block_type='resnet', 
                    global_residual=False, 
                    activation=activation))
            elif block_type in ['tf']:
                self.block_list.append(TransformerBlock(
                    channels=channels, 
                    knn=knn,
                    block_layers=block_layers))
            elif block_type in ['convtf']:
                self.block_list.append(torch.nn.Sequential(ResNetBlock(
                    channels=channels, 
                    kernel_size=kernel_size, 
                    block_layers=block_layers, 
                    block_type='resnet', 
                    global_residual=False, 
                    activation=activation), TransformerBlock(
                    channels=channels, 
                    knn=knn,
                    block_layers=block_layers)))
            elif block_type in ['convtf2']:
                self.block_list.append(ConvTransformerBlock(
                    channels=channels, 
                    kernel_size=kernel_size, 
                    knn=knn,
                    block_layers=block_layers, 
                    block_type='resnet', 
                    global_residual=False, 
                    activation=activation))

            elif block_type in ['convlptf2']:
                self.block_list.append(ConvTransformerBlock(
                    channels=channels, 
                    kernel_size=kernel_size, 
                    knn=knn,
                    block_layers=block_layers, 
                    block_type='resnet', 
                    block_type_tf='lptf', 
                    global_residual=False, 
                    activation=activation))

    def forward(self, x):

        out = self.linear_in(x)

        for i in range(len(self.scaler_list)):
            out = self.block_list[i](out)
            out = self.scaler_list[i](out)
        out = self.block_list[-1](out)

        out = self.linear_out(out)

        return out
    

class ConvTransformerBlock(torch.nn.Module):
    def __init__(self, channels=128, kernel_size=3, knn=16, block_layers=3, block_type='resnet', global_residual=False, activation='relu'):
        super().__init__()
        self.convnet = ResNetBlock(
            channels=channels, 
            kernel_size=kernel_size, 
            block_layers=block_layers, 
            block_type=block_type, 
            global_residual=global_residual, 
            activation=activation)
        self.transformer = TransformerBlock(
            channels=channels, 
            knn=knn,
            block_layers=block_layers)
        self.linear_out = ME.MinkowskiLinear(
            channels*2, channels, bias=True)


    def forward(self, x):
        """input: Sparse Tensor with batch size of one.
        """
        out0 = self.convnet(x)
        out1 = self.transformer(x)

        out = ME.cat(out0, out1)
        out = self.linear_out(out)

        return out

    
######################## lossless attribute ########################
def make_convNet(channels, in_channels, out_channels, kernel_size=3, block_layers=3, activation='relu', block_type='conv'):
    if block_type in ['conv', 'resnet']:
        return torch.nn.Sequential(
            ME.MinkowskiLinear(in_channels, channels),
            ResNetBlock(channels=channels, kernel_size=kernel_size, block_layers=block_layers, 
                        block_type='resnet', global_residual=False, activation=activation),
            ME.MinkowskiLinear(channels, out_channels))
    if block_type=='tf':
        return torch.nn.Sequential(
            ME.MinkowskiLinear(in_channels, channels),
            TransformerBlock(block_layers=block_layers, channels=channels, knn=16),
            ME.MinkowskiLinear(channels, out_channels))
    if block_type=='convtf':
        return torch.nn.Sequential(
            ME.MinkowskiLinear(in_channels, channels),
            ResNetBlock(channels=channels, kernel_size=kernel_size, block_layers=block_layers, 
                        block_type='resnet', global_residual=False, activation=activation),
            TransformerBlock(block_layers=block_layers, channels=channels, knn=16),
            ME.MinkowskiLinear(channels, out_channels))


def make_linearNet(channels, in_channels, out_channels, activation='relu'):
    if activation=='relu':
        return torch.nn.Sequential(
            ME.MinkowskiLinear(in_channels, channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channels, channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channels, out_channels))
        
    elif activation=='prelu':
        return torch.nn.Sequential(
            ME.MinkowskiLinear(in_channels, channels),
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(channels, channels), 
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(channels, out_channels))
