# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

import os, sys, time
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
from data_utils.sparse_tensor import isin, sort_sparse_tensor
from basic_models.resnet import ResNetBlock
from basic_models.backbone import make_convNet, make_linearNet
from basic_models.conditional_entropy_model import SymmetricConditional

from data_utils.LoD_utils import split_voxel, concat_voxel, concat_channels, get_single_voxel, get_one_voxel
import numpy as np

from cfg.get_args import get_args 
args = get_args(component='attribute')


######################################## basic ########################################
class ModelStage(torch.nn.Module):
    def __init__(self, channels=128, in_channels=3, out_channels=3, kernel_size=3, block_layers=3, 
                    ref_channels=0, stride=[2,2,2], split_group=True, block_type='conv', inter_mode=0,
                    in_channels_refT=3, out_channels_refT=32):
        super().__init__()
        in_channels_refT = in_channels# Attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.inter_mode = inter_mode
        if not self.inter_mode: out_channels_refT = 0
        if stride==[2,2,2]: stage, offset_list = 8, [[0],[1],[2],[3],[4],[5],[6],[7]]
        # if stride==[2,2,2]: stage, offset_list = 8, [[0],[7],[1],[6],[5],[2],[3],[4]]
        if stride==[2,1,1]: stage, offset_list = 8, [[0],[4],[1],[5],[2],[6],[3],[7]]
        if stride==[1,2,1]: stage, offset_list = 4, [[0,4],[2,6],[1,5],[3,7]]
        if stride==[1,1,2]: stage, offset_list = 2, [[0,2,4,6],[1,3,5,7]]
        if not split_group: stride, stage, offset_list = [2,2,2], 1, [[0,1,2,3,4,5,6,7]]
        
        # pooling/unpooling for hand-crafted prediction
        self.stride = stride
        self.avgpooling = ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=3)
        self.sumpooling = ME.MinkowskiSumPooling(kernel_size=stride, stride=stride, dimension=3)
        self.unpooling = ME.MinkowskiPoolingTranspose(kernel_size=stride, stride=stride, dimension=3)
        self.pruning = ME.MinkowskiPruning()
        # neural network for prediction from neighboring voxels
        self.stage = stage
        self.offset_list = offset_list
        self.ctxNet_list = torch.nn.ModuleList()
        self.locNet_list = torch.nn.ModuleList()
        self.scaleNet_list = torch.nn.ModuleList()
        self.conditional_entropy_model = SymmetricConditional(distribution='laplace')  # laplace
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

        for i in range(self.stage):
            self.ctxNet_list.append(make_convNet(
                channels=channels, in_channels=in_channels+ref_channels+out_channels_refT, out_channels=channels, 
                kernel_size=kernel_size, block_layers=block_layers, block_type=block_type))
            self.locNet_list.append(make_linearNet(
                channels=channels, in_channels=channels, out_channels=out_channels))
            self.scaleNet_list.append(make_linearNet(
                channels=channels, in_channels=channels, out_channels=out_channels))

        if self.inter_mode:
            self.inter_encoder = torch.nn.Sequential(
                ME.MinkowskiLinear(in_channels_refT, channels),
                ResNetBlock(channels=channels, kernel_size=kernel_size, block_layers=block_layers, 
                            block_type='resnet', global_residual=False),
                ME.MinkowskiConvolution(in_channels=channels, out_channels=channels, 
                            kernel_size=kernel_size, stride=2, bias=True, dimension=3),
                ResNetBlock(channels=channels, kernel_size=kernel_size, block_layers=block_layers, 
                            block_type='resnet', global_residual=False),
                ME.MinkowskiLinear(channels, channels))
            
            self.inter_conv = ME.MinkowskiConvolution(
                in_channels=channels, out_channels=channels, 
                kernel_size=9, stride=1, bias=True, dimension=3)
            
            self.inter_decoder = torch.nn.Sequential(
                ME.MinkowskiLinear(channels, channels),
                ResNetBlock(channels=channels, kernel_size=kernel_size, block_layers=block_layers, 
                            block_type='resnet', global_residual=False),
                ME.MinkowskiConvolutionTranspose(in_channels=channels, out_channels=channels, 
                            kernel_size=kernel_size, stride=2, bias=True, dimension=3),
                ResNetBlock(channels=channels, kernel_size=kernel_size, block_layers=block_layers, 
                            block_type='resnet', global_residual=False),
                ME.MinkowskiLinear(channels, out_channels_refT))

    def inter_predict(self, x, x_refT):
        # encode
        y_refT = self.inter_encoder(x_refT)
        # inter predict
        y = self.avgpooling(x)
        assert y_refT.tensor_stride[0]==y.tensor_stride[0]
        y_pred = self.inter_conv(y_refT, y.C)
        assert (y_pred.C==y.C).all()
        y_pred = ME.SparseTensor(features=y_pred.F, 
            coordinate_map_key=y.coordinate_map_key, 
            coordinate_manager=y.coordinate_manager, device=y.device)
        # decode
        x_pred = self.inter_decoder(y_pred)

        assert (x_pred.C==x.C).all()

        return x_pred

    def forward(self, x, y, x_ref=None, x_refT=None, training=True):
        if self.inter_mode:
            assert x_refT is not None
            x_refT = self.inter_predict(x, x_refT)
        else:
            x_refT = None
        
        # split to multiple groups
        slice_list = split_voxel(x, pruning=self.pruning, offset_list=self.offset_list)
        # TODO: remove empty slices & merge slices
        single_voxel = get_single_voxel(x, sumpooling=self.sumpooling, unpooling=self.unpooling, pruning=self.pruning)
        
        target_list, init_pred_list, pred_list = [], [], []
        for idx in range(len(slice_list)):
            ctxNet, locNet, scaleNet = self.ctxNet_list[idx], self.locNet_list[idx], self.scaleNet_list[idx]
            # undecoded;    decodecd;   target;
            dec_list = slice_list[:idx]
            undec_list = slice_list[idx:]
            target = slice_list[idx]
            if len(target)==0:
                continue
            if sum([len(tp) for tp in undec_list])==0:
                continue
            # 0. remove target have no parents
            mask_single = ~isin(target.C, single_voxel.C)
            target = self.pruning(target, mask_single)
            # print('DBG!!!self.pruning(target, mask_single)')
            if len(target)==0:
                continue
            # 0.1 remove target with only one voxels
            one_voxel = get_one_voxel(dec_list, undec_list, self.sumpooling, self.unpooling, self.pruning)
            mask_one = ~isin(target.C, one_voxel.C) 
            if mask_one.sum()==0:
                # print('bypass', '!'*64)
                continue
            target = self.pruning(target, mask_one)

            # 1. update parent
            parent = update_parent(parent=y, dec_list=dec_list, undec_list=undec_list, sumpooling=self.sumpooling)
            # 2. undate undecoded voxels by unpooling and interpolation from parent
            undec = unpooling_predict(source=parent, target=concat_voxel(undec_list), unpooling=self.unpooling, pruning=self.pruning)
            init_pred = unpooling_predict(source=parent, target=target, unpooling=self.unpooling, pruning=self.pruning)
            res = init_pred.F - target.F
            if res.abs().max()<1e-2:  
                print('bypass zero', '!'*100)
                continue
            # TODO residual prediction
            # 3. predict by network from neighbors
            pred = predict(ctxNet=ctxNet, locNet=locNet, scaleNet=scaleNet, 
                source=concat_voxel(dec_list+[undec]), target=target, pruning=self.pruning, source_ref=x_ref, 
                time_ref=x_refT)
            # collect results
            target_list.append(target)
            init_pred_list.append(init_pred)
            pred_list.append(pred)
            
        return {'target_list':target_list, 'pred_list':pred_list, 'init_pred_list':init_pred_list}

    @torch.no_grad()
    def encode(self, x, y, filename, x_ref=None, x_refT=None):
        # split to multiple groups
        if self.inter_mode:
            assert x_refT is not None
            x_refT = self.inter_predict(x, x_refT)
        else:
            x_refT = None
        if x_ref is not None:
            x_ref = sort_sparse_tensor(x_ref)
        slice_list = split_voxel(x, pruning=self.pruning, offset_list=self.offset_list)  # 8 stage
        single_voxel = get_single_voxel(x, sumpooling=self.sumpooling, unpooling=self.unpooling,
                                        pruning=self.pruning)
        # mask_singel = isin(up_y.C, single_voxel.C)
        target_list, init_pred_list, pred_list = [], [], []
        Bytes = 0
        for idx in range(len(slice_list)):
            ctxNet, locNet, scaleNet = self.ctxNet_list[idx], self.locNet_list[idx], self.scaleNet_list[idx]
            # undecoded;    decodecd;   target;
            dec_list = slice_list[:idx]
            undec_list = slice_list[idx:]
            target = slice_list[idx]

            if len(target) == 0:
                continue
            if sum([len(tp) for tp in undec_list]) == 0:
                continue
            # 0. remove target have no parents
            mask_single = ~isin(target.C, single_voxel.C)
            singel_dec_stage = self.pruning(target, ~mask_single)  # keep have brothers
            target = self.pruning(target, mask_single)  # keep have brothers
            if len(target) == 0:
                continue
            one_voxel = get_one_voxel(dec_list, undec_list, self.sumpooling, self.unpooling, self.pruning)
            mask_one = ~isin(target.C, one_voxel.C)
            last_voxel_target = self.pruning(target, ~mask_one)
            target = self.pruning(target, mask_one)
            if mask_one.sum() == 0:
                # print('bypass', '!'*64)
                continue

            # 1. update parent
            parent = update_parent(parent=y, dec_list=dec_list, undec_list=undec_list, sumpooling=self.sumpooling)
            # 2. undate undecoded voxels by unpooling and interpolation from parent
            undec = unpooling_predict(source=parent, target=concat_voxel(undec_list), unpooling=self.unpooling,
                                      pruning=self.pruning)
            # init_pred = unpooling_predict(source=parent, target=target, unpooling=self.unpooling, pruning=self.pruning)
            with torch.no_grad():
                pred = predict(ctxNet=ctxNet, locNet=locNet, scaleNet=scaleNet,
                               source=concat_voxel(dec_list + [undec]), target=target, pruning=self.pruning,
                               source_ref=x_ref,
                               time_ref=x_refT)

            loc = pred.F[:, :pred.F.shape[-1] // 2]
            scale = pred.F[:, pred.F.shape[-1] // 2:]

            strings, min_v, max_v = self.conditional_entropy_model.compress(target.F, loc, scale)
            with open(filename + '_' + str(idx) + '_A.bin', 'wb') as fout:
                fout.write(strings)
            with open(filename + '_' + str(idx) + '_H.bin', 'wb') as fout:
                fout.write(np.array(min_v, dtype=np.float32).tobytes())
                fout.write(np.array(max_v, dtype=np.float32).tobytes())
            Bytes += os.path.getsize(filename + '_' + str(idx) + '_A.bin') + os.path.getsize(
                filename + '_' + str(idx) + '_H.bin')
            torch.cuda.empty_cache()
        return {'true_bytes': Bytes}

    @torch.no_grad()
    def decode(self, y, y_dec, filename, true_y_sum, x_ref=None, x_refT=None):
        x = self.unpooling(y)  # parent
        if self.inter_mode:
            assert x_refT is not None
            x_refT = self.inter_predict(x, x_refT)
        else:
            x_refT = None
        if x_ref is not None:
            x_ref = sort_sparse_tensor(x_ref)
        slice_list = split_voxel(x, pruning=self.pruning, offset_list=self.offset_list)  # 8 stage
        single_voxel = get_single_voxel(x, sumpooling=self.sumpooling, unpooling=self.unpooling,
                                        pruning=self.pruning)
        # mask_singel = isin(x.C, single_voxel.C)
        true_dec_list = []
        for idx in range(len(slice_list)):
            ctxNet, locNet, scaleNet = self.ctxNet_list[idx], self.locNet_list[idx], self.scaleNet_list[idx]
            undec_list = slice_list[idx:]
            target = slice_list[idx]
            if len(target) == 0:
                continue
            if sum([len(tp) for tp in undec_list]) == 0:
                continue
            # 0. remove target have no parents
            mask_single = ~isin(target.C, single_voxel.C)
            singel_dec_stage = self.pruning(target, ~mask_single)

            target = self.pruning(target, mask_single)  # keep have brothers

            # print('DBG!!!self.pruning(target, mask_single)')
            if len(target) == 0:
                true_dec_list.append(singel_dec_stage)
                continue
            # print('DBG!!! remove empty:\t', slice_list[idx].shape[0], ' --> ', target.shape[0])
            # 0.1 remove target with only one voxels
            one_voxel = get_one_voxel(true_dec_list, undec_list, self.sumpooling, self.unpooling, self.pruning)
            mask_one = ~isin(target.C, one_voxel.C)
            undec_target = self.pruning(target, ~mask_one)
            target = self.pruning(target, mask_one)
            if len(target) > 0:
                # 1. update parent
                parent = update_parent(parent=y, dec_list=true_dec_list, undec_list=undec_list,
                                       sumpooling=self.sumpooling)
                # 2. undate undecoded voxels by unpooling and interpolation from parent
                undec = unpooling_predict(source=parent, target=concat_voxel(undec_list), unpooling=self.unpooling,
                                          pruning=self.pruning)
                with torch.no_grad():
                    pred = predict(ctxNet=ctxNet, locNet=locNet, scaleNet=scaleNet,
                                   source=concat_voxel(true_dec_list + [undec]), target=target, pruning=self.pruning,
                                   source_ref=x_ref,
                                   time_ref=x_refT)

                loc = pred.F[:, :pred.F.shape[-1] // 2]
                scale = pred.F[:, pred.F.shape[-1] // 2:]

                with open(filename + '_' + str(idx) + '_H.bin', 'rb') as fin:
                    min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                with open(filename + '_' + str(idx) + '_A.bin', 'rb') as fin:
                    strings = fin.read()
                target_dec = self.conditional_entropy_model.decompress(strings, loc, scale, min_v, max_v)
                # print(idx)
                # print(target_dec[:10])
                # print('!' * 150)
                target = ME.SparseTensor(features=target_dec, coordinate_manager=target.coordinate_manager,
                                         coordinate_map_key=target.coordinate_map_key, device=target.device)
                # print(target.F[:10])
                # print('!'*100)
                true_dec_list.append(concat_voxel([singel_dec_stage, target]))
                # torch.cuda.empty_cache()
            if len(undec_target) > 0:
                if len(target) <= 0:
                    if len(singel_dec_stage) > 0:
                        true_dec_list.append(singel_dec_stage)

                undec_voxel = concat_voxel(undec_list[1:])
                if undec_voxel is not None:
                    parent_new = update_parent(parent=y, dec_list=true_dec_list, undec_list=[undec_target, undec_voxel],
                                               sumpooling=self.sumpooling)
                else:
                    parent_new = update_parent(parent=y, dec_list=true_dec_list, undec_list=[undec_target],
                                               sumpooling=self.sumpooling)

                last_voxel = unpooling_predict(source=parent_new, target=undec_target, unpooling=self.unpooling,
                                               pruning=self.pruning)
                last_voxel = ME.SparseTensor(features=torch.round(last_voxel.F), coordinate_manager=last_voxel.coordinate_manager,
                                             coordinate_map_key=last_voxel.coordinate_map_key, device=last_voxel.device)

                true_dec_list.append(last_voxel)
                # torch.cuda.empty_cache()
                continue
        dec_pc = concat_voxel(true_dec_list)
        del true_dec_list
        # dec_pc = sort_sparse_tensor(dec_pc)
        torch.cuda.empty_cache()
        return dec_pc



################################################ utils ################################################

######################## update ########################
def update_parent(parent, dec_list, undec_list, sumpooling):
    """use average (sum) voxels (parent) and decoded voxels to update the estimated value of parent voxels.
    """
    if len(dec_list)==0 or sum([len(tp) for tp in dec_list])==0: 
        return parent
    # 
    dec = concat_voxel(dec_list)
    undec = concat_voxel(undec_list)
    # 
    dec_one = ME.SparseTensor(torch.ones([dec.F.shape[0], 1]).float(), 
        coordinate_map_key=dec.coordinate_map_key, coordinate_manager=dec.coordinate_manager, device=dec.device)
    dec_zero = ME.SparseTensor(torch.zeros([dec.F.shape[0], 1]).float(), 
        coordinate_map_key=dec.coordinate_map_key, coordinate_manager=dec.coordinate_manager, device=dec.device)
    undec_one = ME.SparseTensor(torch.ones([undec.F.shape[0], 1]).float(), 
        coordinate_map_key=undec.coordinate_map_key, coordinate_manager=undec.coordinate_manager, device=undec.device)
    undec_zero = ME.SparseTensor(torch.zeros([undec.F.shape[0], dec.F.shape[-1]]).float(), 
        coordinate_map_key=undec.coordinate_map_key, coordinate_manager=undec.coordinate_manager, device=undec.device)
    # numbers 
    num = sumpooling(concat_voxel([dec_one, undec_one]))
    num_undec = sumpooling(concat_voxel([dec_zero, undec_one]))
    sum_dec = sumpooling(concat_voxel([dec, undec_zero]))
    # sort 
    out = sort_sparse_tensor(parent)
    num = sort_sparse_tensor(num)
    num_undec = sort_sparse_tensor(num_undec)
    sum_dec = sort_sparse_tensor(sum_dec)
    # print('DBG!!! update_parent:\t', out.C.shape, num.C.shape, sumpooling)
    # print('DBG!!! update_parent:\t', '\n', out.C, '\n', num.C)
    assert (out.C==num.C).all()
    assert (out.C==num_undec.C).all()
    assert (out.C==sum_dec.C).all()
    # update 
    out_feats = out.F*num.F - sum_dec.F
    mask_undec = torch.where(num_undec.F>0)[0]
    out_feats[mask_undec] /= num_undec.F[mask_undec]
    # out_feats[out_feats<0] = 0
    assert torch.isnan(out_feats).any()==False and torch.isinf(out_feats.abs()).any()==False
    # resort
    out = ME.SparseTensor(features=out_feats, 
        coordinate_map_key=out.coordinate_map_key, 
        coordinate_manager=out.coordinate_manager, device=out.device)
    out = sort_sparse_tensor(out, target=parent)

    return out


######################## predict ########################
def unpooling_predict(source, target, unpooling, pruning):
    """ predict from source (parent node) by unpooling and pruning.
    """
    out = unpooling(source)
    mask = isin(out.C, target.C)
    out = pruning(out, mask)
    out = sort_sparse_tensor(out, target=target)
    assert (out.C==target.C).all()

    return out


def predict(ctxNet, locNet, scaleNet, source, target, pruning, source_ref=None, time_ref=None, prior=None, weightNet=None, softmax=None):
    """ predict from neighbor by network and pruning
    """
    assert torch.isnan(source.F).any()==False and torch.isinf(source.F.abs()).any()==False
    # concat channels
    if source_ref is not None: 
        source_in = concat_channels([source, source_ref])
    else: source_in = source
    # concat temporal channels
    if time_ref is not None: 
        source_in = concat_channels([source_in, time_ref])
    else: source_in = source_in

    # context modeling
    ctx = ctxNet(source_in)
    assert (ctx.C==source.C).all()
    # concat hyper prior
    if prior is not None: ctx = concat_channels([ctx, prior])
    # loc & scale
    loc = locNet(ctx)
    if weightNet is None: loc = loc + source
    else: loc = loc + ME.cat([source]*3)
    scale = scaleNet(ctx)
    assert (loc.C==scale.C).all()
    scale = torch.clamp(scale.F.abs(), min=1e-8)# lower_bound
    # weight (optional)
    if weightNet is not None:
        weight = weightNet(ctx).F
        ch = weight.shape[-1]//3
        weight = torch.stack([weight[:,ch*0:ch*1], weight[:,ch*1:ch*2], weight[:,ch*2:ch*3]], dim=-1)
        weight = softmax(weight)
        weight = torch.cat([weight[:,:,0],weight[:,:,1],weight[:,:,2]],dim=-1)
        out = ME.SparseTensor(torch.cat([loc.F, scale, weight], dim=-1), 
            coordinate_map_key=loc.coordinate_map_key, 
            coordinate_manager=loc.coordinate_manager, device=loc.device)
    else:
        out = ME.SparseTensor(torch.cat([loc.F, scale], dim=-1), 
            coordinate_map_key=loc.coordinate_map_key, 
            coordinate_manager=loc.coordinate_manager, device=loc.device)
    # prune
    mask = isin(out.C, target.C)

    out = pruning(out, mask)
    assert (out.C==target.C).all()
    
    return out