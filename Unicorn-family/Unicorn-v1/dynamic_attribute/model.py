# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-09

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import time
import torch
import MinkowskiEngine as ME
import numpy as np

from basic_models.backbone import Backbone
from lossy_attribute.model import MultiscaleVAE

from cfg.get_args import get_args 
cfg = get_args(component='attribute')


######################################## InterFrameModel ########################################
class InterModel(torch.nn.Module):
    """
    """
    def __init__(self, stage=None):
        super().__init__()

        if stage is not None: self.stage = stage
        else: self.stage = cfg.stage

        self.stride_list = [[2,2,2]]
        self.stride_list = self.stride_list * cfg.scale

        if cfg.inter_mode:
            self.feature_extractor = MultiscaleExtractor(
                stride_list=self.stride_list)
            
            self.inter_predictor = MultiscaleInterPredictor(
                in_channels=cfg.channels+cfg.in_channels, out_channels=cfg.channels, 
                kernel_size=9, stride_list=self.stride_list)

        self.intra_model = MultiscaleVAE(ref_channels=cfg.channels if cfg.inter_mode else 0, stage=self.stage)


    def forward(self, x, ref=None, training=True, lmb=None, encode=False):
        """
        """
        start_time = time.time()
        if cfg.inter_mode:
            assert ref is not None
            _, ref_set, _ = self.feature_extractor(ref)
            pred_set = self.inter_predictor(x, ref_set)
        else:
            pred_set = None
        
        if encode:
            enc_set_list, x_low, gpcc_bits = self.intra_model(x, training=False, ref_set=pred_set, lmb=lmb, encode=True)

            return enc_set_list, x_low, gpcc_bits
        
        else:
            dectime = time.time() - start_time
            out_set = self.intra_model(x, training=training, ref_set=pred_set, lmb=lmb)
            out_set['dectime'].append(dectime)
            
            return out_set

    @torch.no_grad()
    def decode(self, x0, x_low, enc_set_list, ref=None, lmb=None):
        if cfg.inter_mode:
            assert ref is not None
            _, ref_set, _ = self.feature_extractor(ref)
            pred_set = self.inter_predictor(x0, ref_set)
        else:
            pred_set = None

        x_rec = self.intra_model.decode(x0, x_low, enc_set_list, ref_set=pred_set, lmb=lmb)

        return x_rec
    
    @torch.no_grad()
    def test(self, x, ref=None, lmb=None):
        
        start_time = time.time()
        enc_set_list, x_low, gpcc_bits = self.forward( x, ref, training=False, lmb=lmb, encode=True)
        enctime = round(time.time() - start_time, 3)

        x0 = ME.SparseTensor(features=torch.zeros(x.F.shape),
            coordinate_map_key=x.coordinate_map_key, 
            coordinate_manager=x.coordinate_manager, device=x.device)

        start_time = time.time()
        x_rec = self.decode(x0=x0, x_low=x_low, enc_set_list=enc_set_list, ref=ref, lmb=lmb)
        dectime = round(time.time() - start_time, 3)

        # results
        bits = [len(enc_set['strings'])*8 for enc_set in enc_set_list][::-1] + [gpcc_bits]
        bpps = np.array(bits)/len(x)
        bpp = round(sum(bpps), 3)
        gpcc_bpp = round(gpcc_bits/len(x),3)

        print('DBG!!! bpp:\t', bpp, '=', bpps.round(3))

        results = {}
        results['enctime'] = enctime
        results['dectime'] = dectime
        results['bpp'] = bpp
        results['gpcc_bpp'] = gpcc_bpp

        return x_rec, results


######################################## Inter Predictor ########################################
class MultiscaleInterPredictor(torch.nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride_list=None):
        super().__init__()
        self.stride_list = stride_list

        self.inter_predictor = ME.MinkowskiConvolution(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=1, bias=True, dimension=3)
        #                                       
        self.pooling_list = torch.nn.ModuleList()
        for stride in self.stride_list:         
            self.pooling_list.append(ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=3))
        self.pooling_base = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.unpooling_base = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)

    def forward(self, x, ref_set):

        x_list = [x]
        x_low = x
        for i, pooling in enumerate(self.pooling_list):
            x_low = pooling(x_low)
            x_list.append(x_low)
        
        assert len(x_list)==len(self.stride_list)+1

        pred_set = dict()
        for idx, curr_x in enumerate(x_list[:-1]):
            stride_size = curr_x.tensor_stride
            ref = ref_set[str(stride_size)]
            assert curr_x.tensor_stride==ref.tensor_stride
            pred = self.inter_predictor(ref, curr_x.C)
            pred = ME.SparseTensor(features=pred.F, 
                                coordinate_map_key=curr_x.coordinate_map_key, 
                                coordinate_manager=curr_x.coordinate_manager, device=curr_x.device)
            pred_set[str(stride_size)] = pred

        return pred_set


######################################## feature extraction ########################################
class MultiscaleExtractor(torch.nn.Module):
    """
    """
    def __init__(self, stride_list=None):
        super().__init__()
        self.stride_list = stride_list

        self.in_block = Backbone(scale=0, block_type='linear',
            in_channels=cfg.in_channels, channels=cfg.channels, out_channels=cfg.channels)
        self.enc_block = Backbone(scale=1,
            in_channels=cfg.channels+cfg.in_channels, 
            out_channels=cfg.channels, channels=cfg.channels, 
            block_type=cfg.block_type, block_layers=1, 
            kernel_size=cfg.kernel_size,
            stride=self.stride_list[0])
        #
        self.pooling_list = torch.nn.ModuleList()
        for stride in self.stride_list:
            self.pooling_list.append(
                ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=3))
        self.pooling_base = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.unpooling_base = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)
    
    def forward(self, x):
        x_set = dict()
        # 1. average pooling
        x_low = x
        x_set[str(x_low.tensor_stride)] = x_low
        for i, pooling in enumerate(self.pooling_list):
            x_low = pooling(x_low)
            x_set[str(x_low.tensor_stride)] = x_low
        
        x_low = ME.SparseTensor(
            features=torch.round(x_low.F*255.)/255,
            coordinate_map_key=x_low.coordinate_map_key, 
            coordinate_manager=x_low.coordinate_manager, device=x_low.device)

        # 2. extract feature
        f_set = dict()
        f = x
        for i in range(len(self.stride_list)):
            if i == 0: block = self.in_block
            else: block = self.enc_block
            f = block(f)
            stride_size = f.tensor_stride
            assert (f.C==x_set[str(stride_size)].C).all()
            f = ME.cat(f, x_set[str(stride_size)])
            f_set[str(stride_size)] = f
  
        return x_set, f_set, x_low