# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-9-21

import os, sys

sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import time
import torch
import MinkowskiEngine as ME


class CompensatorConvSimple(torch.nn.Module):
    """ using convolution on the target coordinates (failed)
    """

    def __init__(self, channels=32, kernel_size=9, block_layers=3, scale=0):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, stride=1, bias=True, dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x0, x1):
        out = self.conv(x0, x1.C)
        out = ME.SparseTensor(
            features=out.F, coordinates=out.C,
            tensor_stride=x0.tensor_stride, device=out.device)

        return out