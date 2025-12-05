# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-07

import torch
import MinkowskiEngine as ME


class InceptionResNet(torch.nn.Module):
    """Inception Residual Network
    """
    def __init__(self, channels, kernel_size=3, dimension=3, activation='relu'):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=dimension)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x
                
        return out

######################### ResNet #########################
class ResNet(torch.nn.Module): 
    """Residual Network
    """  
    def __init__(self, channels, kernel_size=3, dimension=3, activation='relu'):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        if activation=='relu':
            self.relu = ME.MinkowskiReLU(inplace=True)
        elif activation=='prelu':
            self.relu = ME.MinkowskiPReLU()

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu(out)
        out = self.conv1(out)
        out += x

        return out


# def make_layer(block, block_layers, channels, dimension=3):
#     """make stacked layers.
#     """
#     layers = []
#     for i in range(block_layers):
#         layers.append(block(channels=channels, dimension=dimension))
        
#     return torch.nn.Sequential(*layers)


class ResNetBlock(torch.nn.Module):
    def __init__(self, channels=32, kernel_size=3, block_layers=3, dimension=3, 
                block_type='inception', global_residual=False, activation='relu'):
        super().__init__()
        if block_type=='resnet': Net = ResNet
        if block_type=='inception': Net = InceptionResNet
        self.global_residual = global_residual
        if block_type=='inception': self.global_residual = True 

        self.block_type = block_type
        self.layers = torch.nn.ModuleList()
        for i in range(block_layers):
            if block_type=='resnet':
                self.layers.append(Net(channels=channels, kernel_size=kernel_size, dimension=dimension, activation=activation))
            else:
                self.layers.append(Net(channels=channels, kernel_size=kernel_size, dimension=dimension, activation=activation))

    def forward(self, x):
        out = x
        for resnet in self.layers:
            out = resnet(out)
        if len(self.layers)>1 and self.global_residual:
            out += x
        
        return out