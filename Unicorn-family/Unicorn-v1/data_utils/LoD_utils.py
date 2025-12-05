# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-07

import os, sys, time
import torch
import MinkowskiEngine as ME
import numpy as np
sys.path.append(os.path.split(__file__)[0])
from sparse_tensor import sort_sparse_tensor


######################## scale ########################
def quantize(x, Qstep=1):
    x_feat = x.F.clone()
    mask = torch.abs(torch.abs(torch.frac(x_feat))-0.5)<1e-3
    x_feat[torch.where(mask)] = x_feat[torch.where(mask)].ceil()
    x_feat[torch.where(~mask)] = x_feat[torch.where(~mask)].round()
    # x_feat = x_feat.round()
    out = ME.SparseTensor(x_feat, coordinate_map_key=x.coordinate_map_key, 
                        coordinate_manager=x.coordinate_manager,  device=x.device)

    return out


######################## spatial ########################
def split_voxel(x, pruning, offset_list=[[0,1,2,3,4,5,6,7]]):
    """checkboard-style split for multi-stage prediction.
        split according to the parity of coordinates
    """
    slice_list = []
    # octant = torch.sum((x.C[:,1:]%torch.tensor(2).to(x.device))*torch.tensor([4,2,1]).to(x.device), axis=1)
    octant = x.C[:,1:] / torch.tensor(x.tensor_stride).to(x.device)
    octant =octant % torch.tensor(2).to(x.device)
    # octant = octant * torch.tensor([1,2,4]).to(x.device)# Attention!!! in lossless geometry it's 1,2,4, in lossless attribute coding, it's 4,2,1
    octant = octant * torch.tensor([4,2,1]).to(x.device)# Attention!!! in lossless geometry it's 1,2,4, in lossless attribute coding, it's 4,2,1
    octant = torch.sum(octant, axis=1)

    for _, offsets  in enumerate(offset_list):
        mask = torch.sum(torch.stack([octant==a for a in offsets]), axis=0).bool()
        # print('DBG!!! split_voxel', offsets, mask.sum().item(), mask.shape[0])
        curr_slice = pruning(x, mask)
        slice_list.append(curr_slice)

    return slice_list

def concat_voxel(voxel_list):
    voxel_list = [tp for tp in voxel_list if tp is not None]
    voxel_list = [tp for tp in voxel_list if len(tp)>0]

    if len(voxel_list)==0: return None
    if len(voxel_list)==1: return voxel_list[0]

    feats = torch.cat([x.F for x in voxel_list], dim=0)
    coords = torch.cat([x.C for x in voxel_list], dim=0)
    out = ME.SparseTensor(features=feats, coordinates=coords, 
                        tensor_stride=voxel_list[-1].tensor_stride, 
                        device=voxel_list[-1].device)

    return out

def get_single_voxel(x, sumpooling, unpooling, pruning):
    """get single voxels across scales.
    """
    x_num = ME.SparseTensor(torch.ones([len(x), 1]).float(), coordinate_map_key=x.coordinate_map_key, 
                            coordinate_manager=x.coordinate_manager, device=x.device)
    y_num = sumpooling(x_num)
    x_num_dec = unpooling(y_num)
    mask = x_num_dec.F.squeeze()==1
    assert (x.C==x_num_dec.C).all()
    out = pruning(x, mask)
    
    return out

def get_one_voxel(dec_list, undec_list, sumpooling, unpooling, pruning):
    """get single voxels across scales.
    """
    dec_list = [tp for tp in dec_list if len(tp)>0]
    if len(dec_list)!=0:
        dec = concat_voxel(dec_list)
        dec_zero = ME.SparseTensor(torch.zeros([dec.F.shape[0], 1]).float(), 
            coordinate_map_key=dec.coordinate_map_key, 
            coordinate_manager=dec.coordinate_manager, device=dec.device)
    else: dec_zero = None
    undec_list = [tp for tp in undec_list if len(tp)>0]
    if len(undec_list)!=0:
        undec = concat_voxel(undec_list)
        undec_one = ME.SparseTensor(torch.ones([undec.F.shape[0], 1]).float(), 
            coordinate_map_key=undec.coordinate_map_key, 
            coordinate_manager=undec.coordinate_manager, device=undec.device)
    else: undec_one = None
    if dec_zero is None: x_num = undec_one
    elif undec_one is None: x_num = dec_zero
    else: x_num = concat_voxel([dec_zero, undec_one])
    y_num = sumpooling(x_num)
    x_num_dec = unpooling(y_num)
    mask = x_num_dec.F.squeeze()==1
    assert (x_num.C==x_num_dec.C).all()
    out = pruning(x_num_dec, mask)

    return out

######################## channels ########################
def split_channels(x, channels_list=[1,2]):
    assert x.shape[-1]==sum(channels_list)
    out_list = []
    for idx, _ in enumerate(channels_list):
        start = sum(channels_list[:idx])
        end = sum(channels_list[:idx+1])
        out_list.append(ME.SparseTensor(x.F[:, start:end], 
            coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager, device=x.device))

    return out_list

def concat_channels(x_list):
    if len(x_list)==0: return None
    if len(x_list)==1: return x_list[0]
    #
    out = x_list[0]
    # resort & assert
    for idx, curr_x in enumerate(x_list[1:]):
        if not (curr_x.C==out.C).all(): curr_x = sort_sparse_tensor(curr_x, target=out)
        assert (curr_x.C==out.C).all()
        x_list[idx+1] = curr_x
    #
    out = ME.SparseTensor(torch.cat([curr_x.F for curr_x in x_list], dim=-1),  
        coordinate_map_key=out.coordinate_map_key, 
        coordinate_manager=out.coordinate_manager, 
        device=out.device)

    return out


######################## value ########################

def split_value(x, x_num, pruning, value_list=[1,2,3,4,5,6,7,8]):
    """checkboard-style split for multi-stage prediction.
        split according to the parity of coordinates
    """
    assert (x.C==x_num.C).all()
    slice_list = []
    for idx, value in enumerate(value_list):
        mask = x_num.F.squeeze()==value
        curr_slice = pruning(x, mask)
        slice_list.append(curr_slice)

    return slice_list
