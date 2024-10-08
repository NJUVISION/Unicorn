import os, sys
import open3d as o3d
import numpy as np
import h5py


def read_h5(filedir, dtype_coords='int16', dtype_feats='uint8'):
    # dtype_coords='int32', dtype_feats='int16' for LiDAR
    coords = h5py.File(filedir, 'r')['coords'][:].astype(dtype_coords)
    feats = h5py.File(filedir, 'r')['feats'][:].astype(dtype_feats)

    return coords, feats

def write_h5(filedir, coords, feats, dtype_coords='int16', dtype_feats='uint8'):
    coords = coords.astype(dtype_coords)
    feats = feats.astype(dtype_feats)
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('coords', data=coords, shape=coords.shape)
        h.create_dataset('feats', data=feats, shape=feats.shape)

    return

def read_bin(filedir, dtype="float32"):
    """kitti
    """
    data = np.fromfile(filedir, dtype=dtype).reshape(-1, 4)
    coords = data[:,:3]
    feats = data[:,3:]
    
    return coords, feats


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

    coords = data[:,0:3].astype(dtype_coords)
    if data.shape[-1]==6: feats = data[:,3:6].astype(dtype_feats)
    if data.shape[-1]>6: feats = data[:,6:9].astype(dtype_feats)
    if data.shape[-1] in [4,7]: feats = data[:,3:4].astype(dtype_feats)# for reflectance
    if feats.shape[-1]==3: feats = np.clip(feats, a_min=0, a_max=255)

    if order=='gbr': 
        feats = np.hstack([feats[:,2:3], feats[:,0:2]])

    return coords, feats

def write_ply_ascii(filedir, coords, feats, dtype_coords='int32', dtype_feats='int32'):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    if feats.shape[-1]==3:
        f.writelines(['property uchar red\n','property uchar green\n','property uchar blue\n'])
    if feats.shape[-1]==1:
        f.writelines(['property uint16 reflectance\n'])
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

def read_ply_o3d(filedir, dtype_coords='int32', dtype_feats='int32'):
    pcd = o3d.io.read_point_cloud(filedir)
    coords = np.asarray(pcd.points).astype(dtype_coords)
    feats = (np.asarray(pcd.colors).astype('float32')*255).round()
    feats = np.clip(feats, 0, 255)
    feats = feats.astype(dtype_feats)

    return coords, feats

def write_ply_o3d(filedir, coords, feats, dtype_coords='int32'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(dtype_coords))
    pcd.colors = o3d.utility.Vector3dVector(feats.astype('float32')/255.)
    o3d.io.write_point_cloud(filedir, pcd, write_ascii=True)
    f = open(filedir)
    lines = f.readlines()
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'
    lines[7] = 'property uchar red\n'
    lines[8] = 'property uchar green\n'
    lines[9] = 'property uchar blue\n'
    fo = open(filedir, "w")
    fo.writelines(lines)

    return


