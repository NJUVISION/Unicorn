import os, sys, glob, time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
import MinkowskiEngine as ME
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from attribute.inout import read_h5, read_ply_ascii, read_ply_o3d
from geometry_dataloader import InfSampler
from attribute.color_format import rgb2yuv, yuv2rgb, rgb2YCoCg, YCoCg2rgb


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

    def __init__(self, files, have_label=False, color_format='rgb', normalize=True, motion_scale=1):
        self.files = []
        self.cache = {}
        # self.last_cache_percent = 0
        self.files = files
        self.have_label = have_label
        self.normalize = normalize
        self.color_format = color_format
        self.motion_scale = motion_scale


    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir0 = self.files[idx][0]
        filedir1 = self.files[idx][1]

        if idx in self.cache:
            coords0, feats0, coords1, feats1 = self.cache[idx]
        else:
            if filedir0.endswith('.h5'): 
                coords0, feats0 = read_h5(filedir0)
                coords1, feats1 = read_h5(filedir1)
            if filedir0.endswith('.ply'): 
                if self.color_format=='reflectance': 
                    coords0, feats0 = read_ply_ascii(filedir0)
                    coords1, feats1 = read_ply_ascii(filedir1)
                else: 
                    coords0, feats0 = read_ply_o3d(filedir0)
                    coords1, feats1 = read_ply_o3d(filedir1)
            self.cache[idx] = (coords0, feats0, coords1, feats1)

        coords0 = coords0.astype("int32")
        feats0 = feats0.astype("float32")
        coords1 = coords1.astype("int32")
        feats1 = feats1.astype("float32")

        # normalize & convert color space
        if feats0.shape[-1]==3:
            if self.normalize: out_range=1
            else: out_range=255.
            if self.color_format=='rgb':
                feats0 = feats0 / 255. * out_range
                feats1 = feats1 / 255. * out_range
            elif self.color_format=='yuv':
                feats0 = rgb2yuv(feats0, out_range=out_range)
                feats1 = rgb2yuv(feats1, out_range=out_range)
            elif self.color_format=='y':
                feats0 = rgb2yuv(feats0, out_range=out_range)[:,0:1]
                feats1 = rgb2yuv(feats1, out_range=out_range)[:,0:1]
            elif self.color_format=='ycocg':
                feats0 = rgb2YCoCg(feats0)
                feats1 = rgb2YCoCg(feats1)
            elif self.color_format=='Y':
                feats0 = rgb2YCoCg(feats0)[:,0:1]
                feats1 = rgb2YCoCg(feats1)[:,0:1]
        if feats0.shape[-1]==1:
            if self.normalize: 
                feats0 = feats0 / 255.
                feats1 = feats1 / 255.

        return (coords0, feats0, coords1, feats1)


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
            try:
                Idx0 = int(Idx0.split('-')[-1])
                Idx1 = int(Idx1.split('-')[-1])
            except (ValueError, AttributeError) as e:
                continue
        
        if Idx0 - Idx1!=1: continue
        if forward: seqs_list.append([ref_frame, curr_frame])
        if backward: seqs_list.append([curr_frame, ref_frame])


    return seqs_list