import os, sys, time, glob
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
import MinkowskiEngine as ME
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from geometry_dataloader import InfSampler
from attribute.inout import read_h5, read_ply_ascii, read_ply_o3d
from attribute.color_format import rgb2yuv, yuv2rgb, rgb2YCoCg, YCoCg2rgb


def load_sparse_tensor(filedir, device='cuda', order='rgb', color_format='rgb', normalize=True):
    if filedir.endswith('h5'): coords, feats = read_h5(filedir)
    if filedir.endswith('ply'): 
        # 
        if color_format=='reflectance': 
            coords, feats = read_ply_ascii(filedir)
        else: 
            coords, feats = read_ply_ascii(filedir, order=order)

    coords = coords.astype("int32")
    feats = feats.astype("float32")
    if feats.shape[-1]==3:
        if normalize: out_range=1
        else: out_range=255.
        if color_format=='rgb':
            feats = feats / 255. * out_range
        elif color_format=='yuv':
            feats = rgb2yuv(feats, out_range=out_range)
        elif color_format=='ycocg':
            feats = rgb2YCoCg(feats)
    if feats.shape[-1]==1 and normalize:
        feats = feats / 255.
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)

    return x


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None: new_list_data.append(data)
        else: num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0: raise ValueError('No data in the batch')
    if len(list_data[0])==2:
        coords, feats = list(zip(*list_data))
        coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

        return coords_batch, feats_batch

    elif len(list_data[0])==3:
        coords, feats, label = list(zip(*list_data))
        coords_batch, feats_batch, label_batch = ME.utils.sparse_collate(coords, feats, label)

        return coords_batch, feats_batch, label_batch


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files, have_label=False, color_format='rgb', normalize=True):
        self.files = []
        self.cache = {}
        self.files = files
        self.have_label = have_label
        self.color_format = color_format
        self.normalize = normalize

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]
        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            if filedir.endswith('.h5'): coords, feats = read_h5(filedir)
            if filedir.endswith('.ply'): 
                if self.color_format=='reflectance':
                    coords, feats = read_ply_ascii(filedir)
                else: 
                    coords, feats = read_ply_o3d(filedir)

            self.cache[idx] = (coords, feats)
        coords = coords.astype("int32")
        feats = feats.astype("float32")
        
        # normalize & convert color space
        if feats.shape[-1]==3:
            if self.normalize: out_range=1
            else: out_range=255.
            if self.color_format=='rgb':
                feats = feats / 255. * out_range
            elif self.color_format=='yuv':
                feats = rgb2yuv(feats, out_range=out_range)
            elif self.color_format=='y':
                feats = rgb2yuv(feats, out_range=out_range)[:,0:1]
            elif self.color_format=='ycocg':
                feats = rgb2YCoCg(feats)
            elif self.color_format=='Y':
                feats = rgb2YCoCg(feats)[:,0:1]
            # print('DBG!!!feats', feats)
        if feats.shape[-1]==1:
            if self.normalize: 
                feats = feats / 255.

        return (coords, feats)


def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=collate_pointcloud_fn):
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