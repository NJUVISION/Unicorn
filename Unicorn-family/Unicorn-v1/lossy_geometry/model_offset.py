# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-03

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
from basic_models.resnet import ResNetBlock
from data_utils.geometry.quantize import quantize_sparse_tensor


class OffsetDecoder(torch.nn.Module):
    """
    """
    def __init__(self, in_channels=1, channels=128, out_channels=3, kernel_size=3, block_layers=3):
        super().__init__()
        self.block = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ResNetBlock(
                block_layers=block_layers, channels=channels, 
                kernel_size=kernel_size),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3))
        #
        self.fc = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=channels,
                kernel_size=1, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=channels,
                kernel_size=1, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=out_channels,
                kernel_size=1, stride=1, bias=True, dimension=3))
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out = self.fc(out)

        return out


class OffsetModel(torch.nn.Module):
    def __init__(self, channels=64, kernel_size=3, block_layers=3, posQuantscaleList=None):
        super().__init__()
        self.posQuantscaleList = posQuantscaleList
        self.offset_decoder = OffsetDecoder(
            in_channels=1, channels=channels, out_channels=3, kernel_size=kernel_size,
            block_layers=block_layers)

    def forward(self, x, posQuantscaleList=None):
        out_set_list = []
        if posQuantscaleList is not None: 
            self.posQuantscaleList = posQuantscaleList
        for idx, posQuantscale in enumerate(self.posQuantscaleList):
            x_offset = quantize_sparse_tensor(x, factor=1/posQuantscale, 
                                            return_offset=True, quant_mode='round')
            x_one = ME.SparseTensor(
                features=torch.ones([x_offset.C.shape[0], 1]), 
                coordinate_map_key=x_offset.coordinate_map_key, 
                coordinate_manager=x_offset.coordinate_manager, 
                device=x_offset.device)
            out = self.offset_decoder(x_one)
            # print('DBG!!!', idx, '\t', posQuantscale, len(x_offset), x_offset.C.max().cpu().numpy())
            out_set_list.append({'ground_truth':x_offset, 'out':out})

        return out_set_list

    @torch.no_grad()
    def downscale(self, ground_truth, posQuantscale=None):
        out = quantize_sparse_tensor(ground_truth, factor=1/posQuantscale, 
                                    return_offset=True, quant_mode='round')
        out = ME.SparseTensor(
            features=torch.ones([out.C.shape[0], 1]), 
            coordinate_map_key=out.coordinate_map_key, 
            coordinate_manager=out.coordinate_manager, 
            device=out.device)

        return out

    @torch.no_grad()
    def upscale(self, x, posQuantscale=1):
        if posQuantscale==1:
            coords = x.C[:,1:].float().cpu()
            return coords.numpy()

        out = self.offset_decoder(x)
        offset = out.F.float().cpu()
        coords = out.C[:,1:].float().cpu()
        coords = coords + offset
        coords = coords * posQuantscale

        return coords.numpy()