import open3d as o3d
import numpy as np

def quantize_attribute(xyz, rgb, precision=0.05, quant_mode='floor', DBG=False):
    assert xyz.shape[0]==rgb.shape[0]
    xyz = np.array(xyz).astype('float32')
    rgb = np.array(rgb).astype('float32')
    if DBG: print('DBG!!!quantize in:\t', xyz.shape[0], '\trange:', xyz.min(axis=0), '~', xyz.max(axis=0))
    # quantize
    xyz = xyz/precision
    if quant_mode=='round': xyz = np.round(xyz)
    if quant_mode=='floor': xyz = np.floor(xyz)
    # voxel_down_sample by open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype('int32'))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype('float32'))
    pcd = pcd.voxel_down_sample(voxel_size=1)
    #
    xyz_out = np.asarray(pcd.points).astype('float32')
    rgb_out = np.asarray(pcd.colors).astype('float32')
    if DBG: print('DBG!!!quantize out:\t', xyz_out.shape[0], xyz_out.min(axis=0), '~', xyz_out.max(axis=0))
    
    return xyz_out, rgb_out