
import numpy as np
import torch
import MinkowskiEngine as ME
import time


######################## basic operation ########################
def normalize(points, offset='min'):
    if offset=='mean':
        ref_point = points.mean(axis=0)
    elif offset=='min':
        ref_point = points.min(axis=0)
    else:
        ref_point = np.array([0,0,0])
    points = points - ref_point

    return points, ref_point

def quantize_precision(points, precision=0.001, quant_mode='round', return_offset=False):
    points = points.astype('float')
    points_out = points / float(precision)
    if quant_mode=='round': points_quant = np.round(points_out)
    if quant_mode=='floor': points_quant = np.floor(points_out)
    points_quant = points_quant.astype('int')
    if not return_offset:
        return points_quant
    else:
        quant_error = points_out - points_quant
        return points_quant, quant_error


def dequantize_precision(points, quant_error=0, precision=0.001):
    # points = points.astype('float')
    points_out = points + quant_error
    points_out = points_out * precision

    return points_out

def quantize_resolution(points, resolution=65535, quant_mode='round', return_offset=False):
    # points = points.astype('float')
    min_bound = points.min(axis=0)
    points_out = points - min_bound
    max_bound = points_out.max()
    points_out = points_out / max_bound
    points_out = points_out * resolution
    if quant_mode=='round': points_quant = np.round(points_out)
    if quant_mode=='floor': points_quant = np.floor(points_out)
    points_quant = points_quant.astype('int')
    if not return_offset:
        return points_quant, max_bound, min_bound
    else:
        quant_error = points_out - points_quant
        return points_quant, max_bound, min_bound, quant_error

def dequantize_resolution(points, max_bound, min_bound, quant_error=0, resolution=65535):
    points = points.astype('float')
    points_out = points + quant_error
    points_out = points_out / resolution
    points_out = points_out * max_bound
    points_out = points_out + min_bound

    return points_out


def quantize_octree(points, qlevel=12, quant_mode='round', return_offset=False):
    """Quantization method of OctAttention & VoxelContextNet
    """
    points_out = points.copy()
    # normalize
    centroid = points_out.mean(axis=0)
    points_out = points_out - centroid
    max_bound = np.abs(points_out).max()
    points_out = points_out / max_bound
    # attention
    min_bound = points_out.min(axis=0)
    points_out = points_out - min_bound
    resolution = (2**qlevel-1)/2
    points_out = points_out * resolution
    if quant_mode=='round': points_quant = np.round(points_out)
    if quant_mode=='floor': points_quant = np.floor(points_out)
    points_quant = points_quant.astype('int')
    if not return_offset:
        return points_quant, min_bound, max_bound, centroid
    else:
        quant_error = points_out - points_quant
        return points_quant, min_bound, max_bound, centroid, quant_error

def dequantize_octree(points, min_bound, max_bound, centroid, quant_error=0, qlevel=12):
    points = points.astype('float')
    points_out = points + quant_error
    resolution = (2**qlevel-1)/2
    points_out = points_out / resolution
    points_out = points_out + min_bound
    points_out = points_out * max_bound
    points_out = points_out + centroid

    return points_out

def random_quantize(points, factor=None, min_factor=0.5, max_factor=1):
    if factor is None:
        factor = np.random.uniform(min_factor, max_factor)
    pointsQ = quantize_precision(points, precision=1/factor, quant_mode='round', return_offset=False)
    pointsQ = np.unique(pointsQ, axis=0).astype('int')
    
    return pointsQ


######################## get quantize error ########################
def merge_points(points, offset):
    """TODO
    """
    # quantize
    points = points.astype('int64')
    min_value = points.min(axis=0)
    points_in = points - min_value
    # collect duplicated points
    step = points_in.max() + 1
    points_1d = (sum([points_in[:,i]*(step**i) for i in range(3)]))
    offset_dict = {}
    for i, pt in enumerate(points_1d):
        if not pt in offset_dict: offset_dict[pt] = [offset[i]]
        else: offset_dict[pt].append(offset[i])
    # average duplicated points
    points_out = []
    offset_out = []
    for k, v in offset_dict.items():
        points_out.append(k)
        if len(v)==1: offset_out.append(v[0])
        else: offset_out.append(np.vstack(v).mean(axis=0))
    # collection
    points_out = np.array(points_out)
    points_out = np.vstack([(points_out//(step**i))%step for i in range(3)]).transpose(1,0)
    points_out = points_out + min_value
    offset_out = np.vstack(offset_out)
    assert (np.unique(points, axis=0)==np.unique(points_out, axis=0)).all()

    return points_out, offset_out


def quantize_sparse_tensor(x, factor, return_offset=False, quant_mode='round'):
    # assert factor <= 1
    if factor==1: return x
    coords_batch = x.decomposed_coordinates
    coordsQ_batch, offset_batch = [], []

    for _, coords in enumerate(coords_batch):
        coordsQ, offset = quantize_precision(points=coords.cpu().numpy(), 
                            precision=1/factor, quant_mode=quant_mode, return_offset=True)
        if return_offset: 
            coordsQ, offset = merge_points(coordsQ, offset)
        else:
            coordsQ = np.unique(coordsQ, axis=0)
            offset = np.ones((len(coordsQ),1))
        coordsQ_batch.append(coordsQ)
        offset_batch.append(offset)
    #
    coordsQ_batch, offset_batch = ME.utils.sparse_collate(coordsQ_batch, offset_batch)
    out = ME.SparseTensor(features=offset_batch.float(),
                        coordinates=coordsQ_batch.int(), 
                        tensor_stride=1, device=x.device)

    return out