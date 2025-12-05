# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-07

import os, sys, glob, time
import open3d as o3d
import numpy as np
import argparse
from tqdm import tqdm
import random
rootdir = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(rootdir)
from attribute.inout import read_h5, read_ply_ascii, write_h5, write_ply_ascii, read_ply_o3d, write_ply_o3d, read_bin
from attribute.quantize import quantize_attribute
from attribute.partition import kdtree_partition



def main_partition(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, num_points, n_parts=None):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))
    # random.shuffle(input_filedirs)
    print("input length:\t", len(input_filedirs))
    count = 0
    for _, input_filedir in enumerate(tqdm(input_filedirs)):
        # load
        if input_filedir.endswith('h5'): coords, feats = read_h5(input_filedir)
        if input_filedir.endswith('ply'): 
            if os.path.split(input_filedir)[-1][:4]=='Ford' or args.attribute=='reflectance': 
                coords, feats = read_ply_ascii(input_filedir)
            else:
                # coords, feats = read_ply_o3d(input_filedir)
                coords, feats = read_ply_ascii(input_filedir)
        points = np.hstack([coords, feats])
        # partition
        kdtree_parts = kdtree_partition(points, max_num=num_points, n_parts=n_parts)
        
        for idx_part, points_part in enumerate(kdtree_parts):
            coords_part = points_part[:,:3]
            feats_part = points_part[:,3:]
            # save
            output_filedir = os.path.join(output_rootdir, 'P'+str(idx_part), input_filedir[len(input_rootdir):].split('.')[0])
            # output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0]+'_P'+str(idx_part))
            output_folder, _ = os.path.split(output_filedir)
            os.makedirs(output_folder, exist_ok=True)
            if output_format == 'ply': 
                if os.path.split(input_filedir)[-1][:4]=='Ford' or args.attribute=='reflectance': 
                    write_ply_ascii(output_filedir+'.ply', coords_part, feats_part)
                else: 
                    write_ply_ascii(output_filedir+'.ply', coords_part, feats_part)
            if output_format == 'h5': write_h5(output_filedir+'.h5', coords_part, feats_part)
            count += 1
        if count >= output_length: break
    
    return


def main_quantize(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, precision, quant_mode, max_precision, min_precision):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))[:input_length]
    # random.shuffle(input_filedirs)
    print("input length:\t", len(input_filedirs))
    count = 0
    for _, input_filedir in enumerate(tqdm(input_filedirs)):
        # load
        if input_filedir.endswith('h5'): coords, feats = read_h5(input_filedir)
        if input_filedir.endswith('ply'): coords, feats = read_ply_o3d(input_filedir, dtype_coords='float32')
        if input_filedir.endswith('bin'): 
            coords, feats = read_bin(input_filedir)
            feats = feats * 255.
            feats = np.round(feats)
            feats = np.clip(feats, 0, 255)
        
        # quantize
        
        if max_precision != min_precision:
            precision = random.uniform(min_precision, max_precision)
        
        print('main_quantize', quant_mode, precision)

        coords, feats = quantize_attribute(xyz=coords, rgb=feats, precision=precision, quant_mode=quant_mode, DBG=True)
        feats = feats.round()
        # save
        output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
        output_folder, _ = os.path.split(output_filedir)
        os.makedirs(output_folder, exist_ok=True)
        # if output_format == 'ply': write_ply_ascii(output_filedir+'.ply', coords, feats)
        if output_format == 'ply': write_ply_o3d(output_filedir+'.ply', coords, feats)
        if output_format == 'h5': write_h5(output_filedir+'.h5', coords, feats)
        count += 1
        if count >= output_length: break
    
    return

def main_quantize_kitti(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, precision, quant_mode):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'), recursive=True))[:input_length]
    # random.shuffle(input_filedirs)
    print("input length:\t", len(input_filedirs))
    count = 0
    for _, input_filedir in enumerate(tqdm(input_filedirs)):
        if len(os.path.split(input_filedir)[-1])<4: continue
        try: coords, feats = read_bin(input_filedir)
        except IsADirectoryError: continue

        feats = feats * 255.
        feats = np.round(feats)
        feats = np.clip(feats, 0, 255)

        print('DBG!!!in:\t', feats.shape, feats.min(), feats.mean().round(2), feats.max())
        feats = np.hstack([feats,feats,feats])

        # quantize
        coords, feats = quantize_attribute(xyz=coords, rgb=feats, precision=precision, quant_mode=quant_mode, DBG=True)
        feats = feats.round()
        feats = feats[:,0:1]
        print('DBG!!!out:\t', feats.shape, feats.min(), feats.mean().round(2), feats.max())

        # save
        output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
        output_folder, _ = os.path.split(output_filedir)
        os.makedirs(output_folder, exist_ok=True)
        if output_format == 'ply': write_ply_ascii(output_filedir+'.ply', coords, feats)
        # if output_format == 'ply': write_ply_o3d(output_filedir+'.ply', coords, feats)
        if output_format == 'h5': write_h5(output_filedir+'.h5', coords, feats)
        print('DBG!!! quantize_kitti', input_filedir, '-->', output_filedir)
        count += 1
        if count >= output_length: break
    
    return


def main_quantize_ford(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, precision, quant_mode):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))[:input_length]
    # random.shuffle(input_filedirs)
    print("input length:\t", len(input_filedirs))
    count = 0
    for _, input_filedir in enumerate(tqdm(input_filedirs)):
        # load
        try:
            if input_filedir.endswith('ply'): coords, feats = read_ply_ascii(input_filedir, dtype_coords='int32')
        except (IndexError) as e:
            print('ERROR', e)
            continue

        print('DBG!!!in:\t', feats.shape, feats.min(), feats.mean().round(2), feats.max())
        feats = np.hstack([feats,feats,feats])
        # quantize
        coords, feats = quantize_attribute(xyz=coords, rgb=feats, precision=precision, quant_mode=quant_mode, DBG=True)
        feats = feats.round()
        feats = feats[:,0:1]
        print('DBG!!!out:\t', feats.shape, feats.min(), feats.mean().round(2), feats.max())
        # save
        output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
        output_folder, _ = os.path.split(output_filedir)
        os.makedirs(output_folder, exist_ok=True)
        if output_format == 'ply': write_ply_ascii(output_filedir+'.ply', coords, feats)
        # if output_format == 'ply': write_ply_o3d(output_filedir+'.ply', coords, feats)
        if output_format == 'h5': write_h5(output_filedir+'.h5', coords, feats)
        count += 1
        if count >= output_length: break
    
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--process", default='partition')
    parser.add_argument("--input_rootdir", default='')
    parser.add_argument("--output_rootdir", default='')
    parser.add_argument("--image_rootdir", default='')
    parser.add_argument("--input_format", default='ply')
    parser.add_argument("--output_format", default='ply')
    parser.add_argument("--input_length", type=int, default=int(1e6))
    parser.add_argument("--output_length", type=int, default=int(1e6))
    parser.add_argument("--num_points", type=int, default=8e5)
    parser.add_argument("--precision", type=float, default=0.05)
    parser.add_argument("--n_parts", type=int, default=0)
    parser.add_argument("--max_rate", type=float, default=0.35)
    parser.add_argument("--min_rate", type=float, default=0.25)
    parser.add_argument("--quant_mode", default='floor')
    parser.add_argument("--sample_mode", default='global')
    parser.add_argument("--max_precision", type=float, default=0.05)
    parser.add_argument("--min_precision", type=float, default=0.05)
    parser.add_argument("--attribute", type=str, default='color')
    parser.add_argument("--txtfile", default='')
    args = parser.parse_args()
    if args.n_parts==0: args.n_parts = None

    # partition
    if args.process=='partition':
        main_partition(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                        input_format=args.input_format, output_format=args.output_format, 
                        input_length=args.input_length, output_length=args.output_length, 
                        num_points=args.num_points, n_parts=args.n_parts)

    # quantize
    if args.process=='quantize':
        main_quantize(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                    input_format=args.input_format, output_format=args.output_format, 
                    input_length=args.input_length, output_length=args.output_length,
                    precision=args.precision, quant_mode=args.quant_mode, 
                    max_precision=args.max_precision, min_precision=args.min_precision)

    if args.process=='quantize_ford':
        main_quantize_ford(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                    input_format=args.input_format, output_format=args.output_format,
                    input_length=args.input_length, output_length=args.output_length,
                    precision=args.precision, quant_mode=args.quant_mode)

    if args.process=='quantize_kitti':
        main_quantize_kitti(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                    input_format=args.input_format, output_format=args.output_format,
                    input_length=args.input_length, output_length=args.output_length,
                    precision=args.precision, quant_mode=args.quant_mode)