# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-06

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import glob, os
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME
from geometry.inout import read_h5, read_ply_o3d, read_coords
from geometry.quantize import quantize_precision, random_quantize
from geometry_dataloader import InfSampler


def collate_dynamic_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords0, feats0, coords1, feats1 = list(zip(*list_data))
    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    return coords_batch0, feats_batch0, coords_batch1, feats_batch1


class DynamicPCDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, voxel_size=1, quant_mode='round', augment=False):
        self.files = []
        self.cache = {}
        # self.last_cache_percent = 0
        self.files = files
        self.transforms = transforms
        self.voxel_size = voxel_size
        self.quant_mode = quant_mode
        self.augment = augment

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir0 = self.files[idx][0]
        filedir1 = self.files[idx][1]
        if idx in self.cache:
            coords0, feats0, coords1, feats1 = self.cache[idx]
        else:
            coords0 = read_coords(filedir0, dtype='float32')
            coords1 = read_coords(filedir1, dtype='float32')
            coords0 = quantize_precision(coords0, precision=self.voxel_size, quant_mode=self.quant_mode, return_offset=False)
            coords0 = np.unique(coords0.astype('int'), axis=0).astype('int')
            coords1 = quantize_precision(coords1, precision=self.voxel_size, quant_mode=self.quant_mode, return_offset=False)
            coords1 = np.unique(coords1.astype('int'), axis=0).astype('int')
            if self.augment: 
                factor = round(np.random.uniform(0.5, 1), 2)
                coords0 = random_quantize(coords0, factor)
                coords1 = random_quantize(coords1, factor)
            feats0 = np.ones([len(coords0), 1]).astype('bool')
            feats1 = np.ones([len(coords1), 1]).astype('bool')
            # cache
            self.cache[idx] = (coords0, feats0, coords1, feats1)
        feats0 = feats0.astype("float32")
        feats1 = feats1.astype("float32")

        return coords0, feats0, coords1, feats1

######################## data loader ############################
def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=collate_dynamic_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader


######################## get dataset filedirs ############################
def get_seqs_list(rootdir, forward=True, backward=False):
    seqs_list = []
    filedirs = sorted(glob.glob(os.path.join(rootdir, '**', f'*.ply'), recursive=True))
    filedirs += sorted(glob.glob(os.path.join(rootdir, '**', f'*.h5'), recursive=True))
    filedirs += sorted(glob.glob(os.path.join(rootdir, '**', f'*.bin'), recursive=True))
    filedirs += sorted(glob.glob(os.path.join(rootdir, f'*.ply'), recursive=True))
    filedirs += sorted(glob.glob(os.path.join(rootdir, f'*.bin'), recursive=True))

    for idx in range(len(filedirs)):
        if idx==0: continue
        curr_frame = filedirs[idx]
        ref_frame = filedirs[idx-1]

        Idx0 = os.path.split(curr_frame)[-1].split('.')[0].split('_')[-1]
        Idx1 = os.path.split(ref_frame)[-1].split('.')[0].split('_')[-1]
        
        try:
            Idx0 = int(Idx0)
            Idx1 = int(Idx1)
        except (ValueError) as e:
            Idx0 = int(Idx0.split('-')[-1])
            Idx1 = int(Idx1.split('-')[-1])

        if Idx0 - Idx1!=1: continue
        if forward: seqs_list.append([ref_frame, curr_frame])
        if backward: seqs_list.append([curr_frame, ref_frame])

    return seqs_list