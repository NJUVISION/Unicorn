import open3d as o3d
import os
import numpy as np
import h5py
import torch
import MinkowskiEngine as ME

def kdtree_partition(points, max_num, n_parts=None):
    parts = []
    if n_parts is not None: max_num = len(points) / n_parts + 2

    class KD_node:
        def __init__(self, point=None, LL=None, RR=None):
            self.point = point
            self.left = LL
            self.right = RR

    def createKDTree(root, data):
        if len(data) <= max_num:
            parts.append(data)
            return
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]

        point = data_sorted[int(len(data) / 2)]
        root = KD_node(point)
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):])
        return root

    init_root = KD_node(None)
    root = createKDTree(init_root, points)

    return parts

def list2set(out_set_list):
    out_set = out_set_list[0]
    for curr_out_set in out_set_list[1:]:
        for k in curr_out_set.keys():
            out_set[k] += curr_out_set[k]

    return out_set


def rgb2YCoCg(rgb):
    """input should be integer
    """
    rgb = rgb.astype('float32')
    assert rgb.max()>1
    R, G, B = rgb[:,0], rgb[:,1], rgb[:,2]
    Co = R - B
    t = B + np.floor(Co/2)
    Cg = G - t
    Y = t + np.floor(Cg/2)
    YCoCg = np.stack([Y,Co,Cg], axis=-1)

    return YCoCg


def YCoCg2rgb(YCoCg):
    Y, Co, Cg = YCoCg[:,0], YCoCg[:,1], YCoCg[:,2]

    t = Y - np.floor(Cg/2)
    G = Cg + t
    B = t - np.floor(Co/2)
    R = Co + B
    RGB = np.stack([R,G,B], axis=-1)

    return RGB


def read_ply_ascii(filedir, order='rgb', dtype_coords='int32', dtype_feats='int32'):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    # print('DBG!!!data read_ply_ascii', data.shape)
    # print('DBG!!!data read_ply_ascii', data[0])

    coords = data[:,0:3].astype(dtype_coords)
    if data.shape[-1]==6: feats = data[:,3:6].astype(dtype_feats)
    if data.shape[-1]>6: feats = data[:,6:9].astype(dtype_feats)
    if data.shape[-1] in [4,7]: feats = data[:,3:4].astype(dtype_feats)# for reflectance
    if data.shape[-1]>10: feats = data[:,3:6].astype(dtype_feats)

    # print('DBG!!! read_ply_ascii: feats\t', feats.max(), feats.min())
    # print('DBG!!!data read_ply_ascii', feats)

    if feats.shape[-1]==3: feats = np.clip(feats, a_min=0, a_max=255)

    if order=='gbr':
        feats = np.hstack([feats[:,2:3], feats[:,0:2]])
        # print('DBG!!! gbr  '*100)

    return coords, feats

def write_ply_ascii(filedir, coords, feats, dtype_coords='int32', dtype_feats='int32'):
    if os.path.exists(filedir): os.remove(filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    if feats.shape[-1]==3:
        f.writelines(['property uchar red\n','property uchar green\n','property uchar blue\n'])
    if feats.shape[-1]==1:
        f.writelines(['property uint16 reflectance\n'])
    # f.writelines(['property float x\n','property float y\n','property float z\n',
    #             'property uchar red\n','property uchar green\n','property uchar blue\n',])
    f.write('end_header\n')
    coords = coords.astype(dtype_coords)
    if feats.shape[-1]==3:
        feats = np.clip(feats, 0, 255).astype(dtype_feats)
        for xyz, rgb in zip(coords, feats):
            f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
                        str(rgb[0]), ' ', str(rgb[1]), ' ',str(rgb[2]), '\n'])
    if feats.shape[-1]==1:
        feats = feats.astype(dtype_feats)
        for xyz, r in zip(coords, feats):
            f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
                        str(r[0]), '\n'])
    f.close()

    return

def concat_voxel(voxel_list):
    if len(voxel_list) == 0:
        return None
    voxel_list = [tp for tp in voxel_list if len(tp)>0]
    # print('DBG!!!', [x.device for x in voxel_list])
    # print('DBG!!!', [x.shape for x in voxel_list])
    feats = torch.cat([x.F for x in voxel_list], dim=0)
    coords = torch.cat([x.C for x in voxel_list], dim=0)
    # assert torch.unique(coords.cpu(), dim=0).shape[0]==sum([len(x.C) for x in voxel_list])
    out = ME.SparseTensor(features=feats, coordinates=coords,
                        tensor_stride=voxel_list[-1].tensor_stride,
                        device=voxel_list[-1].device)

    return out

def sort_sparse_tensor(sparse_tensor, target=None):
    """ Sort points in sparse tensor according to their coordinates or the coords of target
    """
    if target is not None and (sparse_tensor.C==target.C).all():
        return ME.SparseTensor(features=sparse_tensor.F,
                            coordinate_map_key=target.coordinate_map_key,
                            coordinate_manager=target.coordinate_manager,
                            device=target.device)

    # positive value
    coords = sparse_tensor.C.clone()
    min_value =  coords.min()
    if min_value < 0: coords[:,1:] -= min_value
    # sort
    indices = torch.argsort(array2vector(coords, coords.max()+1)).cuda()
    out_coords = sparse_tensor.C[indices]
    out_feats = sparse_tensor.F[indices]
    out = create_new_sparse_tensor(coordinates=out_coords, features=out_feats,
                                tensor_stride=sparse_tensor.tensor_stride,
                                dimension=sparse_tensor.D, device=sparse_tensor.device)
    if target is not None:
        # positive value
        target_coords = target.C.clone()
        min_value =  target_coords.min()
        if min_value < 0: target_coords[:,1:] -= min_value
        # sort
        target_indices = torch.argsort(array2vector(target_coords, target_coords.max()+1))
        inverse_indices = target_indices.sort()[1]
        assert (out_coords[inverse_indices]==target.C).all()
        out = ME.SparseTensor(features=out_feats[inverse_indices],
                            coordinate_map_key=target.coordinate_map_key,
                            coordinate_manager=target.coordinate_manager,
                            device=target.device)

    return out


def create_new_sparse_tensor(coordinates, features, tensor_stride, dimension, device):
    sparse_tensor = ME.SparseTensor(features=features,
                                coordinates=coordinates,
                                tensor_stride=tensor_stride,
                                device=device)

    return sparse_tensor

def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    # array, step = array.long().clone(), step.long().clone()
    # if array.min()<0:
    #     min_value = array.min()
    #     array = array - min_value
    #     step = step - min_value
    assert array.min()>=0 and array.max()-array.min()<step
    array, step = array.long(), step.long()
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector


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
