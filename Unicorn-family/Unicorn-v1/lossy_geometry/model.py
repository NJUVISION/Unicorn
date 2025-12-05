# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-03

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import time
import torch
import MinkowskiEngine as ME
from data_utils.geometry.quantize import quantize_sparse_tensor
from data_utils.sparse_tensor import sort_sparse_tensor
from lossless_geometry.model_upsampler import UpSampler1Stage as UpSampler
from model_autoencoder import AutoEncoder


from cfg.get_args import get_args 
args = get_args(component='geometry')


class PCCModel(torch.nn.Module):
    def __init__(self, channels=32, kernel_size=3, block_layers=3, stage=8, scale=1, enc_type='pooling', block_type='conv'):
        super().__init__()
        self.scale = scale
        self.stage = stage
        self.enc_type = enc_type
        # print('DBG!!!PCCModel\n'*10, self.enc_type)
        if enc_type=='pooling':
            self.downsampler = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
            latent_channels=1
        elif enc_type=='ae':
            self.downsampler = AutoEncoder(
                in_channels=1, channels=channels, out_channels=channels, kernel_size=kernel_size, 
                block_layers=block_layers, block_type=block_type)
            latent_channels=channels
        self.upsampler = UpSampler(
            in_channels=latent_channels, channels=channels, kernel_size=kernel_size, 
            block_layers=block_layers, stage=stage, block_type=block_type)
    
    def forward(self, ground_truth, training=True):
        out_set_list = []
        for idx in range(self.scale):
            x = quantize_sparse_tensor(ground_truth, factor=1/(2**idx), quant_mode='floor')
            if self.enc_type=='pooling':
                x_low = self.downsampler(x)
            else:
                enc_set = self.downsampler(x, training=training)
                x_low = enc_set['x_low']
            out_set = self.upsampler(x_low, x_high=x)
            if self.enc_type!='pooling': out_set.update(enc_set)
            out_set_list.append(out_set)
        
        return out_set_list

    @torch.no_grad()
    def encode(self, x, scale=4):
        """lossless encode the input data
        """
        if x.C.min() < 0:
            ref_point = x.C.min(axis=0)[0]
            x = ME.SparseTensor(features=x.F, coordinates=x.C - ref_point, 
                                tensor_stride=x.tensor_stride, device=x.device)
        else: ref_point = None
        #
        bitstream_AE_list = []
        bitstream_list = []
        for idx in range(scale):
            if self.enc_type=='pooling': 
                x_low = self.downsampler(x)
            elif self.enc_type=='ae':
                x_low, bitstream_AE = self.downsampler.encode(x, return_one=False)
                bitstream_AE_list.append(bitstream_AE)
            bitstream = self.upsampler.encode(x_low, x)
            bitstream_list.append(bitstream)
            x = ME.SparseTensor(features=torch.ones((len(x_low),1)).float(),
                                coordinates=torch.div(x_low.C,2,rounding_mode='floor'), 
                                device=x_low.device)
            if x.shape[0]<64: break

        return {'ref_point':ref_point, 'bitstream_AE_list':bitstream_AE_list, 
                'bitstream_list':bitstream_list, 'x':x}

    @torch.no_grad()
    def decode(self, input_set):
        ref_point = input_set['ref_point']
        bitstream_AE_list = input_set['bitstream_AE_list'][::-1]
        bitstream_list = input_set['bitstream_list'][::-1]
        x = input_set['x']
        for idx, bitstream in enumerate(bitstream_list):
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            if self.enc_type=='ae':
                x = self.downsampler.decode(x, bitstream_AE_list[idx])
            x = self.upsampler.decode(x, bitstream)
        if ref_point is not None:
            x = ME.SparseTensor(features=x.F, coordinates=x.C + ref_point, 
                                tensor_stride=x.tensor_stride, device=x.device)

        return x

    @torch.no_grad()
    def test(self, x, scale=4):
        start = time.time()
        enc_set = self.encode(x, scale=scale)
        print('enc time:\t', round(time.time() - start, 3))
        
        start = time.time()
        x_dec = self.decode(enc_set)
        print('dec time:\t', round(time.time() - start, 3))
        assert (sort_sparse_tensor(x).C==sort_sparse_tensor(x_dec).C).all()
        bits = sum([len(bitstream) for bitstream in enc_set['bitstream_list']])*8
        if self.enc_type=='ae':
            bits_AE = sum([len(bitstream) for bitstream in enc_set['bitstream_AE_list']])*8
            # print('DBG!!!', bits_AE/x.shape[0])
            bits += bits_AE
        bpp = round(bits / x.shape[0], 3)
        print('bpp:\t', bpp, x.shape[0])
        print("memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')

        return
