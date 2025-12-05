# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-06

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import time
import torch
import MinkowskiEngine as ME
from basic_models.resnet import ResNetBlock
from basic_models.transformer import TransformerBlock

from lossless_geometry.model_upsampler import UpSampler8Stage
from dynamic_geometry.model_compensator import CompensatorConvSimple


######################################################################################## 
class DownSampler(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, out_channels=32, block_layers0=3, block_layers1=3, kernel_size=5, block_type='tf'):
        super().__init__()

        self.block0 = self.make_block(in_channels=in_channels, channels=channels, out_channels=channels, 
                                    kernel_size=kernel_size, block_layers=block_layers0, block_type=block_type)                
        self.conv = ME.MinkowskiConvolution(in_channels=channels, out_channels=channels,
                                    kernel_size=2, stride=2, bias=True, dimension=3)
        self.block1 = self.make_block(in_channels=channels, channels=channels, out_channels=channels, 
                                    kernel_size=kernel_size, block_layers=block_layers1, block_type=block_type) 
        self.relu = ME.MinkowskiReLU(inplace=True)

    def make_block(self, in_channels=32, channels=32, out_channels=32, kernel_size=3, block_layers=3, knn=16, block_type='conv'):
        if block_type=='conv':
            return torch.nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=in_channels, out_channels=channels,
                    kernel_size=kernel_size, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ResNetBlock(block_layers=block_layers, channels=channels, kernel_size=kernel_size),
                ME.MinkowskiConvolution(
                    in_channels=channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=1, bias=True, dimension=3))

        if block_type=='tf': 
            return torch.nn.Sequential(
                ME.MinkowskiConvolution(in_channels=in_channels, out_channels=channels,
                    kernel_size=kernel_size, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                TransformerBlock(block_layers=block_layers, channels=channels, knn=knn),
                ME.MinkowskiConvolution(in_channels=channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=1, bias=True, dimension=3))

    def forward(self, x):
        out = self.block0(x)
        out = self.relu(self.conv(out))
        out = self.block1(out)

        return out


######################################################################################## 
class PCCModel(torch.nn.Module):
    """Dynamic PCC Model
    """
    def __init__(self, inter_mode=1, channels=32, scale=4, stage=8, kernel_size=5, block_type='tf'):
        """extract feature form pc0 & reconstruct pc1
           transport pc0's feature to pc1
        """
        super().__init__()
        self.inter_mode = inter_mode
        self.scale = scale
        self.channels = channels
        # 
        self.downsampler = DownSampler(in_channels=1, channels=channels, out_channels=channels, 
                                    kernel_size=5, block_layers0=3, block_layers1=3, block_type=block_type)
        self.upsampler = UpSampler8Stage(in_channels=channels, channels=channels, 
                                    kernel_size=5, block_layers=3, stage=8, block_type=block_type)
        self.compensator = CompensatorConvSimple(channels=channels, kernel_size=9)

        self.pruning = ME.MinkowskiPruning()
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)


    def compensate(self, latent0, latent1_one):
        # TODO: normalization
        coords1 = latent1_one.C[:,1:].float()/2
        feats1 = latent1_one.F
        coords0 = latent0.C[:,1:].float()/2
        feats0 = latent0.F

        tensor0 = ME.SparseTensor(
            features=feats0, 
            coordinates=torch.hstack([torch.zeros([len(coords0), 1]).to(coords0.device), coords0]),
            tensor_stride=1, device=feats0.device)
        tensor1 = ME.SparseTensor(
            features=feats1, 
            coordinates=torch.hstack([torch.zeros([len(coords1), 1]).to(coords1.device), coords1]),
            tensor_stride=1, device=feats1.device)
        latent1 = self.compensator(tensor0, tensor1)

        assert (latent1.C[:,1:]*2==latent1_one.C[:,1:]).all()
        latent1 = ME.SparseTensor(latent1.F,
            coordinate_map_key=latent1_one.coordinate_map_key, 
            coordinate_manager=latent1_one.coordinate_manager, 
            device=latent1_one.device)

        return latent1

    def forward(self, pc0, pc1, training=True):
        """x0: reference frame (known). x1: current frame (unknown).
        """
        out_set_list = []

        x0, x1 = pc0, pc1
        for idx in range(self.scale):
            latent1_one = self.pooling(x1)
            latent1_one = ME.SparseTensor(torch.ones([latent1_one.shape[0], self.channels]).float(), 
                                    coordinate_map_key=latent1_one.coordinate_map_key, 
                                    coordinate_manager=latent1_one.coordinate_manager, 
                                    device=latent1_one.device)
            if not self.inter_mode:
                latent1 = latent1_one
            elif self.inter_mode:
                latent0 = self.downsampler(x0)
                latent1 = self.compensate(latent0, latent1_one)
                assert (latent1.C==latent1_one.C).all()
                
            # decode
            out_set = self.upsampler(latent1, x_high=x1)
            out_set_list.append(out_set)

            # normalize
            pc0 = self.pooling(pc0)
            x0 = ME.SparseTensor(features=pc0.F, coordinates=torch.div(pc0.C,pc0.tensor_stride[0],rounding_mode='floor'),
                                tensor_stride=1, device=pc0.device)
            pc1 = self.pooling(pc1)
            x1 = ME.SparseTensor(features=pc1.F, coordinates=torch.div(pc1.C,pc1.tensor_stride[0],rounding_mode='floor'),
                                tensor_stride=1, device=pc1.device)
        
        all_out_set_list = []
        for idx, out_set in enumerate(out_set_list):
            all_out_set_list.append(out_set)

        return all_out_set_list



if __name__ == '__main__':
    model = PCCModel()
    print(model)
    