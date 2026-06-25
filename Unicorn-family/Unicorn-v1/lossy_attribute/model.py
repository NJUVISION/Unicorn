# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-08         

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch 
import MinkowskiEngine as ME
import numpy as np
import math
import time
from basic_models.backbone import Backbone
from lossy_attribute.model_resvae import ResidualVAE

from data_utils.attribute.inout import write_ply_ascii
from third_party.gpcc_attr import gpcc_encode

from cfg.get_args import get_args 
args = get_args(component='attribute')


class MultiscaleVAE(torch.nn.Module):
    """
    """
    def __init__(self, stage=None, ref_channels=0):
        super().__init__()
        # 1. set stage
        if stage is not None: self.stage = stage
        else: self.stage = args.stage
        if self.stage in [1]: self.stride_list = [[2,2,2]]
        elif self.stage in [3]: self.stride_list = [[2,1,1],[1,2,1],[1,1,2]]
        self.stride_list = self.stride_list * args.scale

        ############# pooling/unpooling
        # pooling
        self.pooling_list = torch.nn.ModuleList()
        for stride in self.stride_list:
            self.pooling_list.append(ME.MinkowskiAvgPooling(
                kernel_size=stride, stride=stride, dimension=3))
        # unpooling
        self.unpooling_list = torch.nn.ModuleList()
        for stride in self.stride_list:
            self.unpooling_list.append(ME.MinkowskiPoolingTranspose(
                kernel_size=stride, stride=stride, dimension=3))
        self.unpooling_list = self.unpooling_list[::-1]

        ############# VAE
        self.linear_in = Backbone(scale=0, block_type='linear',
            in_channels=args.in_channels, channels=args.channels, out_channels=args.channels)

        if self.stage==1:
            self.upscaler = Backbone(scale=-1,
                in_channels=args.channels+args.in_channels,
                channels=args.channels, out_channels=args.channels, 
                block_type=args.block_type, block_layers=args.block_layers, kernel_size=args.kernel_size,
                stride=stride)
            self.VAE = ResidualVAE(stride=stride)
        elif self.stage==3:
            self.upscaler_list = torch.nn.ModuleList()
            self.VAE_list = torch.nn.ModuleList()
            for i, stride in enumerate(self.stride_list[::-1][:3]):
                self.upscaler_list.append(Backbone(scale=-1,
                    in_channels=args.channels+args.in_channels, channels=args.channels, out_channels=args.channels, 
                    block_type=args.block_type, block_layers=args.block_layers, kernel_size=args.kernel_size, 
                    stride=stride))
                self.VAE_list.append(ResidualVAE(stride=stride))

        ############ variable-rate coding
        if args.Vmode!=0:
            self.embedder = Embedder(256)
        
        ############ inter prediction
        if ref_channels!=0:
            self.inter_compensator = Backbone(scale=0, 
                in_channels=args.channels+ref_channels, channels=args.channels, out_channels=args.channels,
                block_type=args.block_type, block_layers=args.block_layers, kernel_size=args.kernel_size)

    def forward(self, x, training=True, ref_set=None, lmb=None, encode=False):
        """
        """
        if args.Vmode: emb = self.embedder(lmb, device=x.device)
        else: emb = None

        # downscaling
        x_set = dict()
        x_low = x
        x_set[str(x_low.tensor_stride)] = x_low
        for i, pooling in enumerate(self.pooling_list):
            x_low = pooling(x_low)
            x_set[str(x_low.tensor_stride)] = x_low
        # losslessly encode x_low
        x_low = ME.SparseTensor(features=torch.round(x_low.F*255.)/255,
            coordinate_map_key=x_low.coordinate_map_key, 
            coordinate_manager=x_low.coordinate_manager, device=x_low.device)
        if not training or encode:
            gpcc_bits = gpcc_lossless_encode(x_low)
            gpcc_bpp = round(gpcc_bits/len(x), 3)
        else: gpcc_bpp = 0

        # upscaling
        if encode:
            enc_set_list = []
        else:
            likelihood_list = []
            Qlatent_list = []
            gt_list = []
            out_list = []
            dectime_list = []
            real_bits_list = []

        for idx_scale, unpooling in enumerate(self.unpooling_list):
            if idx_scale==0:
                curr_x = x_low
                curr_f = self.linear_in(curr_x)
                curr_dec = curr_f - curr_f
            
            assert (curr_x.C==curr_f.C).all()
            if self.stage==1: upscaler = self.upscaler
            if self.stage==3: upscaler = self.upscaler_list[idx_scale%self.stage]

            curr_f = upscaler(ME.cat([curr_f, curr_x]))
            curr_x = unpooling(curr_x)
            curr_dec = unpooling(curr_dec)
        
            assert (curr_f.C==curr_x.C).all()

            stride_size = curr_x.tensor_stride

            x_gt = x_set[str(stride_size)]
            
            ############### inter predictor
            if ref_set is not None and str(stride_size) in ref_set:
                curr_ref = ref_set[str(stride_size)]
                assert (curr_ref.C==curr_x.C).all()
                curr_f = self.inter_compensator(ME.cat([curr_f, curr_ref])) + curr_f

            if self.stage==1: VAE = self.VAE
            if self.stage==3: VAE = self.VAE_list[idx_scale%self.stage]

            if encode:
                # VAE.test(x_in=curr_x, x_gt=x_gt, f_in=curr_f, prior_dec=curr_dec, emb=emb)
                out_set = VAE.encode(x_in=curr_x, x_gt=x_gt, f_in=curr_f, prior_dec=curr_dec, emb=emb)
                enc_set_list.append({'strings':out_set['strings'], 'min_v':out_set['min_v'], 'max_v':out_set['max_v'],
                                    'x_out':out_set['x_out'], 'f_out':out_set['f_out'], 'dec':out_set['dec']})
                
            else:
                out_set = VAE(x_in=curr_x, x_gt=x_gt, f_in=curr_f, prior_dec=curr_dec, training=training, emb=emb)
                
                likelihood_list.append(out_set['likelihood'])
                Qlatent_list.append(out_set['Qlatent'])
                gt_list.append(out_set['x_gt'])
                out_list.append(out_set['x_out'])
                dectime_list.append(out_set['dectime'])
                real_bits_list.append(out_set['real_bits'])

            curr_x = out_set['x_out']
            curr_f = out_set['f_out']
            curr_dec = out_set['dec']
        
        if encode:

            return enc_set_list, x_low, gpcc_bits
        else:

            return {'gt_list':gt_list[::-1], 'out_list':out_list[::-1], 'x_low':x_low,
                    'likelihood_list': likelihood_list[::-1],
                    'Qvalue_list':[Qlatent.F.round() for Qlatent in Qlatent_list][::-1], 'gpcc_bpp':gpcc_bpp,
                    'dectime':dectime_list,
                    'real_bits_list':real_bits_list, 
                    'curr_f': curr_f}
    
    @torch.no_grad()
    def decode(self, x0, x_low, enc_set_list, ref_set=None, lmb=None):
        # downscaling
        x0_low = x0
        for i, pooling in enumerate(self.pooling_list):
            x0_low = pooling(x0_low)
        x_low = ME.SparseTensor(features=x_low.F,
            coordinate_map_key=x0_low.coordinate_map_key, 
            coordinate_manager=x0_low.coordinate_manager, device=x0_low.device)

        if args.Vmode: emb = self.embedder(lmb, device=x0_low.device)
        else: emb = None

        # downscaling
        for idx_scale, unpooling in enumerate(self.unpooling_list):
            if idx_scale==0:
                curr_x = x_low
                curr_f = self.linear_in(curr_x)
                curr_dec = curr_f - curr_f
            
            assert (curr_x.C==curr_f.C).all()
            if self.stage==1: upscaler = self.upscaler
            if self.stage==3: upscaler = self.upscaler_list[idx_scale%self.stage]

            curr_f = upscaler(ME.cat([curr_f, curr_x]))
            curr_x = unpooling(curr_x)
            curr_dec = unpooling(curr_dec)
        
            assert (curr_f.C==curr_x.C).all()

            stride_size = curr_x.tensor_stride

            ############### inter predictor
            if ref_set is not None and str(stride_size) in ref_set:
                curr_ref = ref_set[str(stride_size)]
                assert (curr_ref.C==curr_x.C).all()
                curr_f = self.inter_compensator(ME.cat([curr_f, curr_ref])) + curr_f

            if self.stage==1: VAE = self.VAE
            if self.stage==3: VAE = self.VAE_list[idx_scale%self.stage]

            enc_set = enc_set_list[idx_scale]
            strings, min_v, max_v = enc_set['strings'], enc_set['min_v'], enc_set['max_v']

            dec_set = VAE.decode(strings=strings, min_v=min_v, max_v=max_v, x_in=curr_x, f_in=curr_f, prior_dec=curr_dec, emb=emb)

            curr_x = dec_set['x_out']
            curr_f = dec_set['f_out']
            curr_dec = dec_set['dec']
            
            # assert (curr_x.C==enc_set['x_out'].C).all()
            # assert (curr_x.F==enc_set['x_out'].F).all()
            # assert (curr_f.F==enc_set['f_out'].F).all()
            # assert (curr_dec.F==enc_set['dec'].F).all()
        
        return curr_x
    
    def test(self, x, ref_set=None, lmb=None):
        
        start_time = time.time()
        enc_set_list, x_low, gpcc_bits = self.forward(x=x, training=False, ref_set=ref_set, lmb=lmb, encode=True)
        enctime = round(time.time() - start_time, 3)

        x0 = ME.SparseTensor(features=torch.zeros(x.F.shape),
            coordinate_map_key=x.coordinate_map_key, 
            coordinate_manager=x.coordinate_manager, device=x.device)

        start_time = time.time()
        x_rec = self.decode(x0=x0, x_low=x_low, enc_set_list=enc_set_list, ref_set=ref_set, lmb=lmb)
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


############################################### lossless encoding by G-PCC ###############################################
@torch.no_grad()
def gpcc_lossless_encode(x):
    coords = torch.div(x.C[:,1:], x.tensor_stride[0], rounding_mode='floor')
    coords = coords.detach().cpu().numpy()
    if args.color_format == 'reflectance':
        feats = x.F.detach().cpu() * 100
        feats = np.clip(feats.round().int().numpy(), 0, 100)
    else:
        feats = x.F.detach().cpu()*255
        feats = np.clip(feats.round().int().numpy(), 0,255)
    assert feats.max()>1
    os.makedirs('output/gpcc', exist_ok=True)
    import random
    prefix = str(random.randint(1,32))+str(len(x))
    filedir = os.path.join('output/gpcc', prefix+'tp.ply')
    write_ply_ascii(filedir=filedir, coords=coords, feats=feats)
    bin_dir = os.path.join('output/gpcc', prefix+'tp.bin')
    if x.F.shape[-1]==1:
        results_enc = gpcc_encode(filedir, bin_dir, transformType=1, qp=4, attribute='reflectance', show=False)
    else:
        results_enc = gpcc_encode(filedir, bin_dir, transformType=1, qp=4, show=False)
    if 'colors bitstream size' in results_enc:
        bits = results_enc['colors bitstream size']*8
    elif 'reflectances bitstream size' in results_enc:
        bits = results_enc['reflectances bitstream size']*8

    return bits
    

############################### embedding lambda for variable-rate coding (see https://github.com/duanzhiihao/lossy-vae) ###############################
class Embedder(torch.nn.Module):
    """
    """                                 
    def __init__(self, embed_channels): 
        super().__init__()              
        self.embed_channels = embed_channels
        self.MAX_LMB = 8192             
        self._sin_period = 64           
        self.lmb_embedding = torch.nn.Sequential(
            torch.nn.Linear(embed_channels, embed_channels),
            torch.nn.GELU(),
            torch.nn.Linear(embed_channels, embed_channels))
    
    def sinusoidal_embedding(self, values, dim=256, max_period=64):
        assert values.dim() == 1 and (dim % 2) == 0
        exponents = torch.linspace(0, 1, steps=(dim // 2))
        freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
        args = values.view(-1, 1) * freqs.view(1, dim//2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return embedding

    def forward(self, lmb, device='cuda'):
        """lmb: weight_distortion
            n: # batch size
        """
        lmb = torch.tensor([lmb]).to(device)
        lmb_input = torch.log(lmb) * self._sin_period / math.log(self.MAX_LMB)
        embedding = self.sinusoidal_embedding(lmb_input, dim=self.embed_channels,
                                            max_period=self._sin_period)
        embedding = self.lmb_embedding(embedding)

        return embedding
