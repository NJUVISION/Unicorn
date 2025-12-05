# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-08

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import time
import torch
import MinkowskiEngine as ME
from data_utils.geometry.quantize import quantize_sparse_tensor
from model_upsampler import UpSampler8Stage


from cfg.get_args import get_args 
cfg = get_args(component='geometry')


class PCCModel(torch.nn.Module):
    def __init__(self, channels=32, kernel_size=3, block_layers=3, stage=8, scale=1, block_type='conv'):
        super().__init__()
        self.scale = scale
        self.stage = stage
        self.downsampler = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.upsampler = UpSampler8Stage(in_channels=1, channels=channels, kernel_size=kernel_size, 
                                    block_layers=block_layers, stage=stage, block_type=block_type)

        self.DBG = cfg.DBG

    def forward(self, ground_truth, training=True):
        out_set_list = []
        for idx in range(self.scale):
            x = quantize_sparse_tensor(ground_truth, factor=1/(2**idx), quant_mode='floor')
            if self.DBG: 
                print('DBG!!! forward:\t', idx, len(x), x.tensor_stride[0],
                    x.C.max().cpu().numpy(), x.C.min().cpu().numpy(), 
                    x.C.max().cpu().numpy() - x.C.min().cpu().numpy())
            
            x_low = self.downsampler(x)
            out_set = self.upsampler(x_low, x_high=x)
            out_set_list.append(out_set)

        return out_set_list


if __name__ == '__main__':
    model = PCCModel(block_type='ptf')
    print(model)