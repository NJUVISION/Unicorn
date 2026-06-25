# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-06

import os, sys, time
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
from data_utils.sparse_tensor import sort_sparse_tensor, array2vector
from basic_models.backbone import Backbone
from data_utils.LoD_utils import concat_voxel
from basic_models.conditional_entropy_model import SymmetricConditional


from cfg.get_args import get_args 
args = get_args(component='attribute')


class ResidualVAE(torch.nn.Module):
    def __init__(self, stride=[2,2,2]):
        super().__init__()
        # 1. encoder/decoder
        ##################################################################################
        self.encoder = Backbone(scale=1,
            in_channels=args.in_channels, channels=args.channels, out_channels=args.latent_channels, 
            block_type=args.block_type, block_layers=args.block_layers, kernel_size=args.kernel_size, 
            stride=stride)

        self.decoder = Backbone(scale=-1,
            in_channels=args.latent_channels, channels=args.channels, out_channels=args.channels, 
            block_type=args.block_type, block_layers=args.block_layers, kernel_size=args.kernel_size, 
            stride=stride)
                
        if args.Vmode!=0:
            self.EQlayer = AdaptiveQuantizer(embed_channels=256, channels=args.latent_channels, mode=args.Vmode)
            self.DQlayer = AdaptiveQuantizer(embed_channels=256, channels=args.latent_channels, mode=args.Vmode)
        
        # 2. entropy model
        ##################################################################################
        self.entropy_fn = SymmetricConditional()
        
        self.block_prior = Backbone(scale=1,
            in_channels=args.in_channels+args.channels*2, 
            channels=args.channels, out_channels=args.channels, 
            block_type=args.block_type, block_layers=args.block_layers, kernel_size=args.kernel_size, 
            stride=stride)

        self.loc_net = Backbone(scale=0, block_type='linear',
            in_channels=args.channels, channels=args.channels, out_channels=args.latent_channels)
        self.scale_net = Backbone(scale=0, block_type='linear',
            in_channels=args.channels, channels=args.channels, out_channels=args.latent_channels)
        
        # 3. output
        ##################################################################################
        self.fuseNet = Backbone(scale=0, block_type='linear', 
            in_channels=args.channels, channels=args.channels, out_channels=args.channels)
        self.outNet = Backbone(scale=0, block_type='linear', 
            in_channels=args.channels, channels=args.channels, out_channels=args.in_channels)
        
        self.sumpooling = ME.MinkowskiSumPooling(kernel_size=stride, stride=stride, dimension=3)
        self.unpooling = ME.MinkowskiPoolingTranspose(kernel_size=stride, stride=stride, dimension=3)
        self.pruning = ME.MinkowskiPruning()
    

    def forward(self, x_in=None, x_gt=None, f_in=None, prior_dec=None, training=True, emb=None):
        """             
        skip_mode==1:   skip single-voxel at encoder side (lidar PCs)
        """
        # skip mode: for sparse lidar PCC
        if args.skip_mode==1:
            x_in0, x_in1, x_gt0, x_gt1, f_in0, f_in_single, prior_dec0 = self.split_input(x_in, x_gt, f_in, prior_dec)
        else:
            x_in0, x_gt0, f_in0, prior_dec0 = x_in, x_gt, f_in, prior_dec
        if x_in0 is None and x_gt0 is None:
            latent = self.encoder(x_in1)
            QlatentF = self.entropy_fn._quantize(latent.F, mode='noise' if training else "symbols")
            Qlatent = ME.SparseTensor(features=QlatentF, coordinate_map_key=latent.coordinate_map_key,
                                      coordinate_manager=latent.coordinate_manager, device=latent.device)
            return {'x_gt': x_gt1, 'x_out': x_in1,
                    'f_out': f_in_single, 'dec': prior_dec, 'Qlatent':Qlatent,
                    'likelihood': torch.ones([x_gt.shape[0], x_gt.shape[1]]).cuda(),
                    'dectime': 0,
                    'real_bits': 0}
        # input
        enc = x_gt0 - x_in0
        prior = ME.cat([x_in0, f_in0, prior_dec0])

        # encoder transform
        latent = self.encoder(enc)
        if args.Vmode!=0: latent = self.EQlayer(latent, emb)

        # entropy modeling
        start_dec_time = time.time()
        Qlatent, likelihood, real_bits = self.get_entropy(latent=latent, prior=prior, training=training)

        # decoder transform
        if args.Vmode!=0:
            Qlatent = self.DQlayer(Qlatent, emb)

        dec = self.decoder(Qlatent)

        if args.skip_mode==1:
            dec = self.concat_input(dec, f_in_single, target=f_in)

        # fuse
        f_out = self.fuseNet(f_in + dec)
        x_out = self.outNet(f_out)
        x_out = x_out + x_in

        # self.test(x_in, x_gt, f_in, prior_dec, emb)

        return {'x_gt':x_gt, 'x_out':x_out, 
                'f_out':f_out, 'dec': dec,
                'likelihood':likelihood, 'Qlatent':Qlatent,
                'dectime':time.time() - start_dec_time, 
                'real_bits':real_bits}

    def get_entropy(self, latent, prior, training):
        
        QlatentF = self.entropy_fn._quantize(latent.F, mode='noise' if training else "symbols")
        Qlatent = ME.SparseTensor(features=QlatentF, coordinate_map_key=latent.coordinate_map_key,
                                coordinate_manager=latent.coordinate_manager, device=latent.device)
        # prior
        prior = self.block_prior(prior)
        # entropy modeling
        loc = self.loc_net(prior)
        scale = self.scale_net(prior)
        assert (Qlatent.C==loc.C).all()
        loc = loc.F
        scale = scale.F.abs()
        scale = torch.clamp(scale, min=1e-8)# lower_bound
        _, likelihood = self.entropy_fn(QlatentF, loc, scale, quantize_mode=None)
        #======================== arithmetic encoding & decoding ========================
        if not training:
            strings, min_v, max_v = self.entropy_fn.compress(QlatentF, loc, scale)
            real_bits = len(strings)*8
        else:
            real_bits = 0

        return Qlatent, likelihood, real_bits
    
    @torch.no_grad()
    def encode(self, x_in, x_gt, f_in, prior_dec=None, emb=None):
        # skip mode: for sparse lidar PCC
        if args.skip_mode==1:
            x_in0, _, x_gt0, _, f_in0, f_in_single, prior_dec0 = self.split_input(x_in, x_gt, f_in, prior_dec)
        else:
            x_in0, x_gt0, f_in0, prior_dec0 = x_in, x_gt, f_in, prior_dec

        enc = x_gt0 - x_in0

        prior = ME.cat([x_in0, f_in0, prior_dec0])

        # encoding transform
        latent = self.encoder(enc)
        if args.Vmode!=0:
            latent = self.EQlayer(latent, emb)
        
        # entropy modeling
        QlatentF = self.entropy_fn._quantize(latent.F, mode="symbols")
        Qlatent = ME.SparseTensor(features=QlatentF, coordinate_map_key=latent.coordinate_map_key,
                                coordinate_manager=latent.coordinate_manager, device=latent.device)
        prior = self.block_prior(prior)
        loc = self.loc_net(prior)
        scale = self.scale_net(prior)

        Qlatent = sort_sparse_tensor(Qlatent)
        loc = sort_sparse_tensor(loc)
        scale = sort_sparse_tensor(scale)

        assert (Qlatent.C==loc.C).all()

        loc = loc.F
        scale = scale.F.abs()
        scale = torch.clamp(scale, min=1e-8) # lower_bound

        # arithmetic encoding
        strings, min_v, max_v = self.entropy_fn.compress(Qlatent.F, loc, scale)
        Qlatent = sort_sparse_tensor(Qlatent, target=prior)

        # decoding transform
        if args.Vmode!=0:
            Qlatent = self.DQlayer(Qlatent, emb)

        dec = self.decoder(Qlatent)

        if args.skip_mode==1:
            dec = self.concat_input(dec, f_in_single, target=f_in)

        # fuse
        dec = sort_sparse_tensor(dec, target=f_in)
        f_out = self.fuseNet(f_in + dec)
        x_out = self.outNet(f_out)
        x_out = x_out + x_in

        return {'x_out':x_out, 'f_out':f_out, 'dec': dec,
                'strings':strings, 'min_v':min_v, 'max_v':max_v,
                 'real_bits':len(strings)*8}

    @torch.no_grad()
    def decode(self, strings, min_v, max_v, x_in, f_in, prior_dec=None, emb=None):

        if args.skip_mode==1:
            x_in0, _, _, _, f_in0, f_in_single, prior_dec0 = self.split_input(x_in, None, f_in, prior_dec)
        else:
            x_in0, f_in0, prior_dec0 = x_in, f_in, prior_dec

        prior = ME.cat([x_in0, f_in0, prior_dec0])

        prior = self.block_prior(prior)
        loc = self.loc_net(prior)
        scale = self.scale_net(prior)
        loc = sort_sparse_tensor(loc)
        scale = sort_sparse_tensor(scale)

        loc = loc.F
        scale = scale.F.abs()
        scale = torch.clamp(scale, min=1e-8) # lower_bound

        # arithmetic decoding
        QlatentF = self.entropy_fn.decompress(strings, loc, scale, min_v, max_v, channels=args.latent_channels)
        
        index = array2vector(prior.C, step=prior.C.max()+1).argsort().argsort()
        assert (sort_sparse_tensor(prior).C[index]==prior.C).all()
        index = index.to(QlatentF.device)
        Qlatent = ME.SparseTensor(features=QlatentF[index], 
                            coordinate_map_key=prior.coordinate_map_key, 
                            coordinate_manager=prior.coordinate_manager, 
                            device=prior.device)

        # decoding transform
        if args.Vmode!=0:
            Qlatent = self.DQlayer(Qlatent, emb)

        dec = self.decoder(Qlatent)

        if args.skip_mode==1:
            dec = self.concat_input(dec, f_in_single, target=f_in)
        
        # fuse
        dec = sort_sparse_tensor(dec, target=f_in)
        f_out = self.fuseNet(f_in + dec)
        x_out = self.outNet(f_out)
        x_out = x_out + x_in

        return {'x_out':x_out, 'f_out':f_out, 'dec': dec}

    @torch.no_grad()
    def test(self, x_in=None, x_gt=None, f_in=None, prior_dec=None, emb=None):
        enc_set = self.encode(x_in, x_gt, f_in, prior_dec, emb)

        strings = enc_set['strings']
        min_v = enc_set['min_v']
        max_v = enc_set['max_v']

        dec_set = self.decode(strings, min_v, max_v, x_in, f_in, prior_dec, emb)

        assert (dec_set['x_out'].C==enc_set['x_out'].C).all()
        assert (dec_set['x_out'].F==enc_set['x_out'].F).all()

        return 
    
    def split_input(self, x_in, x_gt, f_in, prior):
        """x_gt0: 
            x_gt1, x_in1, f_in1: single voxels
        """
        # single voxel
        mask_single, _ = self.get_single_voxel(x_in)
        if mask_single.all():

            return None, x_in, None, x_gt, None, f_in, None
        
        # split x_in
        x_in0 = self.pruning(x_in, ~mask_single)
        x_in1 = self.pruning(x_in, mask_single)
        x_in0 = ME.SparseTensor(features=x_in0.F, coordinates=x_in0.C, 
                                tensor_stride=x_in0.tensor_stride, device=x_in0.device)
        x_in1 = ME.SparseTensor(features=x_in1.F, coordinates=x_in1.C, 
                                tensor_stride=x_in1.tensor_stride, device=x_in1.device)
        # split x_gt
        if x_gt is not None:
            assert (x_gt.C==x_in.C).all()
            x_gt0 = self.pruning(x_gt, ~mask_single)
            x_gt1 = self.pruning(x_gt, mask_single)
            x_gt0 = sort_sparse_tensor(x_gt0, target=x_in0)
            x_gt1 = sort_sparse_tensor(x_gt1, target=x_in1)
        else:
            x_gt0, x_gt1 = None, None

        # split f_in
        assert (f_in.C==x_in.C).all()
        f_in0 = self.pruning(f_in, ~mask_single)
        f_in1 = self.pruning(f_in, mask_single)
        f_in0 = sort_sparse_tensor(f_in0, target=x_in0)
        f_in1 = sort_sparse_tensor(f_in1, target=x_in1)

        # split prior
        assert (prior.C==x_in.C).all()
        prior = self.pruning(prior, ~mask_single)
        prior = sort_sparse_tensor(prior, target=x_in0)

        return x_in0, x_in1, x_gt0, x_gt1, f_in0, f_in1, prior

    def concat_input(self, f_dec0, f_in1, target):
        """f_dec1 is just placehoder.
        """
        f_dec1 = ME.SparseTensor(
            torch.zeros(f_in1.shape).float(), 
            coordinate_map_key=f_in1.coordinate_map_key, 
            coordinate_manager=f_in1.coordinate_manager, 
            device=f_in1.device)
        f_dec = concat_voxel([f_dec0, f_dec1])
        f_dec = sort_sparse_tensor(f_dec, target=target)

        return f_dec

    def get_single_voxel(self, x):
        x1 = ME.SparseTensor(torch.ones([len(x), 1]).float(), 
                                coordinate_map_key=x.coordinate_map_key, 
                                coordinate_manager=x.coordinate_manager, 
                                device=x.device)
        if args.DBG: print('DBG!!!x_num', x1.shape, x1.tensor_stride)
        x2 = self.sumpooling(x1)

        x3 = self.unpooling(x2)
        assert (x3.C==x.C).all()

        mask_single = x3.F.squeeze()==1
        index_single = torch.where(mask_single)[0]
        assert (x3.F[index_single]==1).all()

        if args.DBG:
            print('DBG!!! get_single_voxel:\t', x3.tensor_stride, len(x3), x3.F.max(), x3.F.mean(), x3.F.min())
            for i in range(1,9):
                tp_index = torch.where(x3.F.squeeze()==i)[0]
                print('DBG!!!', i, round(len(tp_index)/len(x3),3), len(tp_index), len(x3))
        
        return mask_single, index_single
    

######################################### affine transform for adaptive quantization ##############################################
class AdaptiveQuantizer(torch.nn.Module):   
    def __init__(self, embed_channels, channels, mode=1):
        super().__init__()
        self.scale_net = torch.nn.Sequential(
            torch.nn.Linear(embed_channels, embed_channels),
            torch.nn.GELU(),
            torch.nn.Linear(embed_channels, embed_channels),
            torch.nn.GELU(),
            torch.nn.Linear(embed_channels, channels))
        self.bias_net = torch.nn.Sequential(
            torch.nn.Linear(embed_channels, embed_channels),
            torch.nn.GELU(),
            torch.nn.Linear(embed_channels, embed_channels),
            torch.nn.GELU(),
            torch.nn.Linear(embed_channels, channels))

        self.mode = mode
        if self.mode==2:
            self.MLP = torch.nn.Sequential(
                torch.nn.Linear(channels, channels),
                torch.nn.GELU(),
                torch.nn.Linear(channels, channels),
                torch.nn.GELU(),
                torch.nn.Linear(channels, channels))
        
    def forward(self, x, emb):
        scale = self.scale_net(emb)
        bias = self.bias_net(emb)

        x_F = x.F
        # x_F = x_F * (1 + scale) + bias
        x_F = x_F * scale + bias

        if self.mode==2:
            x_F = self.MLP(x_F)

        x_out = ME.SparseTensor(
            features=x_F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)

        return x_out
