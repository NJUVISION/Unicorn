# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-9-21

import os, sys

sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import time
import torch
import MinkowskiEngine as ME

from data_utils.sparse_tensor import sort_sparse_tensor
from lossy_geometry.model_autoencoder import DownSampler
from model_compensator import CompensatorConvSimple


class PCCModel(torch.nn.Module):
    """Dynamic PCC Model
    """

    def __init__(self, mode='inter', channels=32, scale=4, stage=8, kernel_size=3):
        """extract feature form pc0 & reconstruct pc1 
           transport pc0's feature to pc1
        """
        super().__init__()
        self.mode = mode
        self.scale = scale
        self.channels = channels
        # 
        self.downsampler = DownSampler(in_channels=1, channels=channels, out_channels=channels,
                                       kernel_size=3, block_layers0=3, block_layers1=3)
        if stage == 1:
            from lossless_geometry.model_upsampler import UpSampler1Stage as UpSampler
        else:
            from lossless_geometry.model_upsampler import UpSampler8Stage as UpSampler
        self.upsampler = UpSampler(in_channels=channels, channels=channels, kernel_size=3, block_layers=3, stage=stage)
        self.conv_compensator = CompensatorConvSimple(channels=channels, kernel_size=9)
        self.pruning = ME.MinkowskiPruning()
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)

    def forward(self, pc0, pc1, scale=None, training=True):
        """x0: reference frame (known). x1: current frame (unknown).
        """
        if scale is not None:
            self.scale = scale
        out_set_list = []
        x0, x1 = pc0, pc1
        for idx in range(self.scale):
            latent1_one = self.pooling(x1)
            latent1_one = ME.SparseTensor(torch.ones([latent1_one.shape[0], self.channels]).float(),
                                          coordinate_map_key=latent1_one.coordinate_map_key,
                                          coordinate_manager=latent1_one.coordinate_manager,
                                          device=latent1_one.device)
            if self.mode == 'intra':
                latent1 = latent1_one
            elif self.mode == 'inter':
                latent0 = self.downsampler(x0)
                latent1 = self.conv_compensator(latent0, latent1_one)
                assert (latent1.C == latent1_one.C).all()
            # decode
            out_set = self.upsampler(latent1, x_high=x1)
            out_set_list.append(out_set)
            # normalize
            pc0 = self.pooling(pc0)
            x0 = ME.SparseTensor(features=pc0.F,
                                 coordinates=torch.div(pc0.C, pc0.tensor_stride[0], rounding_mode='floor'),
                                 tensor_stride=1, device=pc0.device)
            pc1 = self.pooling(pc1)
            x1 = ME.SparseTensor(features=pc1.F,
                                 coordinates=torch.div(pc1.C, pc1.tensor_stride[0], rounding_mode='floor'),
                                 tensor_stride=1, device=pc1.device)

        return out_set_list

    @torch.no_grad()
    def upsample(self, x0, latent1, num_points):
        """upsample latent1 with the help of x0.
        """
        assert x0.tensor_stride[0] == 1
        assert latent1.tensor_stride[0] == 2
        latent1_one = ME.SparseTensor(
            torch.ones([latent1.shape[0], self.channels]).float(),
            coordinate_map_key=latent1.coordinate_map_key,
            coordinate_manager=latent1.coordinate_manager,
            device=latent1.device)
        if self.mode == 'intra':
            latent1 = latent1_one
        elif self.mode == 'inter':
            latent0 = self.downsampler(x0)
            latent1 = self.conv_compensator(latent0, latent1_one)
            assert (latent1.C == latent1_one.C).all()
        # decode
        x1_dec = self.upsampler.upsample(latent1, num_points)

        return x1_dec

    @torch.no_grad()
    def encode(self, pc0, pc1, scale=4):
        bitstream_list = []
        x0, x1 = pc0, pc1
        for idx in range(scale):
            # downscale
            latent1_one = self.pooling(x1)
            latent1_one = ME.SparseTensor(torch.ones([latent1_one.shape[0], self.channels]).float(),
                                          coordinate_map_key=latent1_one.coordinate_map_key,
                                          coordinate_manager=latent1_one.coordinate_manager,
                                          device=latent1_one.device)
            if self.mode == 'intra':
                latent1 = latent1_one
            elif self.mode == 'inter':
                latent0 = self.downsampler(x0)
                latent1 = self.conv_compensator(latent0, latent1_one)
                assert (latent1.C == latent1_one.C).all()
            # upscale
            bitstream = self.upsampler.encode(x_low=latent1, x_high=x1)
            bitstream_list.append(bitstream)
            # normalize
            pc0 = self.pooling(pc0)
            x0 = ME.SparseTensor(features=pc0.F,
                                 coordinates=pc0.C // pc0.tensor_stride[0],
                                 tensor_stride=1, device=pc0.device)
            pc1 = self.pooling(pc1)
            x1 = ME.SparseTensor(features=pc1.F,
                                 coordinates=pc1.C // pc1.tensor_stride[0],
                                 tensor_stride=1, device=pc1.device)

        return bitstream_list, x1

    @torch.no_grad()
    def decode(self, pc0, x1, bitstream_list):
        scale = len(bitstream_list)
        # downscale pc0
        x0 = pc0
        x0_list = [x0]
        for i in range(scale - 1):
            x0 = self.pooling(x0)
            x0 = ME.SparseTensor(features=x0.F, coordinates=x0.C // x0.tensor_stride[0], tensor_stride=1,
                                 device=x0.device)
            x0_list.append(x0)
        #
        x0_list = x0_list[::-1]
        bitstream_list = bitstream_list[::-1]
        # upscale x1
        for idx in range(scale):
            bitstream = bitstream_list[idx]
            x0 = x0_list[idx]
            assert x1.tensor_stride[0] == 1
            latent1_one = ME.SparseTensor(features=torch.ones([x1.shape[0], self.channels]).float(),
                                          coordinates=x1.C * 2, tensor_stride=2, device=x1.device)
            if self.mode == 'intra':
                latent1 = latent1_one
            elif self.mode == 'inter':
                latent0 = self.downsampler(x0)
                latent1 = self.conv_compensator(latent0, latent1_one)
                assert (latent1.C == latent1_one.C).all()
            # upscale
            x1 = self.upsampler.decode(x_low=latent1, bitstream=bitstream)

        return x1

    def test(self, pc0, pc1, scale=4):
        start = time.time()
        bitstream_list, x1 = self.encode(pc0, pc1, scale=scale)
        print('enc time:\t', round(time.time() - start, 3))

        start = time.time()
        pc1_dec = self.decode(pc0, x1, bitstream_list)
        print('dec time:\t', round(time.time() - start, 3))
        assert (sort_sparse_tensor(pc1).C == sort_sparse_tensor(pc1_dec).C).all()

        bits = sum([len(bitstream) for bitstream in bitstream_list]) * 8
        bpp = round(bits / pc1.shape[0], 3)
        print('bpp:\t', bpp, pc1.shape[0])
        print("memoey:\t", round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2), 'GB')

        return


if __name__ == '__main__':
    model = PCCModel()
    print(model)
