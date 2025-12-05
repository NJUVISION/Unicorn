# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-03

import os, sys, time
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
import pandas as pd
from data_utils.geometry.quantize import quantize_sparse_tensor
from data_utils.geometry.partition import kdtree_partition
from data_utils.geometry.inout import read_ply_o3d, write_ply_o3d
from third_party.pc_error_geo import pc_error

from cfg.get_args import get_args 
args = get_args(component='geometry')


################################### LossyCoder ###################################
class LossyCoder():
    """
    """
    def __init__(self, lossless_coder, model_AE_low, model_AE_high, model_SR_low, model_SR_high, model_offset=None, device='cuda'):
        #
        self.lossless_coder = lossless_coder

        self.model_AE_low = model_AE_low
        self.model_AE_high = model_AE_high
        #
        self.model_SR_low = model_SR_low
        self.model_SR_high = model_SR_high

        self.model_offset = model_offset
        
        self.threshold_lossy = args.threshold_lossy

        self.max_num = args.max_num
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)

        self.device = device

    @torch.no_grad()
    def encode(self, x_raw, scale_AE, scale_SR, posQuantscale=1):
        x = quantize_sparse_tensor(x_raw, factor=1/posQuantscale, quant_mode='round')

        x_tp = x
        num_points_list = [len(x_tp)]
        for _ in range(scale_AE+scale_SR):
            x_tp = self.pooling(x_tp)# pooling is fast       
            num_points_list.append(len(x_tp))

        gt_list = []
        idx = 0
        for _ in range(scale_SR):
            gt_list.append(x)
            assert x.shape[0]==num_points_list[idx]
            model_SR = self.model_SR_low
            
            #
            x = model_SR.downsampler(x)
            x = ME.SparseTensor(features=torch.ones((len(x),1)).float(),
                                coordinates=torch.div(x.C,2,rounding_mode='floor'), device=x.device)
            idx += 1
        
        if scale_AE==0:
            bitstream = None
        else:
            gt_list.append(x)
            assert x.shape[0]==num_points_list[idx]
            density =num_points_list[idx]/num_points_list[idx+1]
            if density>self.threshold_lossy: model_AE = self.model_AE_low
            else: model_AE = self.model_AE_high
            
            if args.DBG: print('DBG!!! threshold_lossy', idx, self.threshold_lossy, 
                    'high' if density<self.threshold_lossy > self.threshold_lossy else 'low')
            #
            x, bitstream = model_AE.downsampler.encode(x, return_one=True)
            x = ME.SparseTensor(features=torch.ones((len(x),1)).float(),
                                coordinates=torch.div(x.C,2,rounding_mode='floor'), device=x.device)
            
        return x, bitstream, gt_list, num_points_list

    @torch.no_grad()
    def decode(self, x, bitstream, num_points_list, gt_list=None, posQuantscale=1):
        num_points_list = num_points_list[::-1]
        gt_list = gt_list[::-1]
        
        # print('DBG!!! num_points_list', num_points_list)
        idx = 0
        if bitstream is not None:
            assert x.shape[0]==num_points_list[idx]
            density = num_points_list[idx+1]/num_points_list[idx]
            if density>self.threshold_lossy: model_AE = self.model_AE_low
            else: model_AE = self.model_AE_high

            if args.DBG: print('DBG!!! threshold_lossy', idx, self.threshold_lossy, 
                    'high' if density<self.threshold_lossy > self.threshold_lossy else 'low')

            if model_AE.scale==1:
                x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                    tensor_stride=2, device=x.device)
                x = model_AE.downsampler.decode(x, bitstream)
                num_points = num_points_list[idx+1]
                x = model_AE.upsampler.upsample(x, num_points)
            else:
                nums_list = num_points_list[:model_AE.scale][::-1]
                x = model_AE.decode(x, bitstream, nums_list)
            idx += model_AE.scale

        for _, num_points in enumerate(num_points_list[idx+1:]):
            gt = gt_list[idx]
            # upscaling
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            if args.DBG: print('DBG!!!!!! upsample:\t', idx, '\tdensity:', round(len(gt)/len(x),2))

            assert x.shape[0]==num_points_list[idx]
            if num_points_list[idx+1]/num_points_list[idx]>self.threshold_lossy: model_SR = self.model_SR_low
            else: model_SR = self.model_SR_high
            x = model_SR.upsampler.upsample(x, num_points)

            torch.cuda.empty_cache()
            idx += 1

        x_dec = quantize_sparse_tensor(x, factor=posQuantscale)
        coords_dec = x_dec.C[:,1:].cpu().numpy()

        if self.model_offset is not None:
            coords_offset = self.model_offset.upscale(x, posQuantscale=posQuantscale)
        else:
            coords_offset = coords_dec

        return coords_dec, coords_offset


    @torch.no_grad()
    def test_one(self, filedir, bin_dir, dec_dir, scale_AE=1, scale_SR=2, posQuantscale=1, psnr_resolution=1023, test_psnr=True):
        # load data
        x_raw = self.lossless_coder.read_data(filedir, voxel_size=1)
        num_points_input = x_raw.shape[0]

        start = time.time()
        # downscale
        x, bitstream_AE, gt_list, num_points_list = self.encode(
                            x_raw, scale_AE=scale_AE, scale_SR=scale_SR, posQuantscale=posQuantscale)
        # bits
        ae_bits = len(num_points_list)*32# int32
        if bitstream_AE is not None:      
            ae_bits += len(bitstream_AE)*8
        ##################### lossless coder

        bitstream_list = self.lossless_coder.encode(x)

        enc_time = round(time.time() - start, 3)
        lossless_bits = self.lossless_coder.write_bitstream(bitstream_list, bin_dir)
        enc_time_all = round(time.time() - start, 3)

        start = time.time()
        # decode
        bitstream_list_dec = self.lossless_coder.read_bitstream(bin_dir)
        start_dec = time.time()

        x_dec = self.lossless_coder.decode(bitstream_list_dec)

        points_dec, points_offset = self.decode(x_dec, bitstream_AE, num_points_list, gt_list=gt_list, posQuantscale=posQuantscale)
        dec_time = round(time.time() - start_dec, 3)
        dec_time_all = round(time.time() - start, 3)

        # bpp
        bits = ae_bits + lossless_bits
        lossless_bpp = round(lossless_bits / num_points_input, 3)
        ae_bpp = round(ae_bits / num_points_input, 3)
        bpp = round(bits / num_points_input, 3)

        # write file
        write_ply_o3d(dec_dir, points_dec, dtype='int32')

        if self.model_offset is not None:
            offset_dir = dec_dir[:-4]+'_offset.ply'
            write_ply_o3d(offset_dir, points_offset, dtype='float32')
        
        memory = round(torch.cuda.max_memory_allocated()/1024**3,2)

        results = {'filedir':filedir, 'num_points_input':num_points_input, 'num_points':points_dec.shape[0], 'psnr_resolution':psnr_resolution, 
                    'file_size':bits, 'bpp':bpp, 
                    'lossless_file_size':lossless_bits, 'lossless_bpp':lossless_bpp, 
                    'ae_file_size':ae_bits, 'ae_bpp':ae_bpp, 
                    'enc_time':enc_time, 'dec_time':dec_time, 
                    'enc_time_all':enc_time_all, 'dec_time_all':dec_time_all, 
                    'memory(GB)':memory}
        
        # psnr
        if test_psnr:
            ref_dir = dec_dir[:-4]+'_ref.ply'
            points_raw = x_raw.C[:,1:].detach().cpu().numpy()
            write_ply_o3d(ref_dir, points_raw, dtype='int32', normal=True, knn=16)
            psnr_results = self.get_psnr(dec_dir, ref_dir, resolution=psnr_resolution)
            #
            results.update(psnr_results)

        return results

    def get_psnr(self, dec_dir, ref_dir, resolution, test_d3=False):
        psnr_results = pc_error(ref_dir, dec_dir, resolution=resolution, normal=True, show=False)
        if test_d3:
            # d3 psnr
            from third_party.d3metric.d3_utils import load_geometry
            from third_party.d3metric.metrics import d3psnr
            pts_A = load_geometry(ref_dir, False)
            pts_B = load_geometry(dec_dir, False)
            print("Calculate d3-PSNR (A->A|B)")
            d3_mse1, d3_psnr1, nPtsA, nPtsB, radiusA_A, radiusB_A, blk_size = d3psnr(pts_A, pts_B,
                                                                                    bitdepth_psnr=16,
                                                                                    mapsfolder='', 
                                                                                    mapsprefix='density_map', 
                                                                                    diffradius=False, 
                                                                                    localaverage=False)
            d3_results = {'d3_psnr':round(d3_psnr1, 4), 'd3_mse':round(d3_mse1, 4)}
            psnr_results.update(d3_results)

        return psnr_results

    @torch.no_grad()
    def test(self, filedir, bin_dir, dec_dir, scale_AE=1, scale_SR=2, posQuantscale=1, psnr_resolution=1023, max_num=4e5, test_psnr=True):
        """
        """
        # print('DBG!!!test', posQuantscale)
        points_raw = read_ply_o3d(filedir)
        if points_raw.min()<0:
            min_point = points_raw.min()
            print('DBG!!! min_point <0:\t', min_point)
            points_raw = points_raw - min_point

        if len(points_raw)>self.max_num: points_list = kdtree_partition(points_raw, max_num=self.max_num)
        else: points_list = [points_raw]
        results_list = []
        dec_list = []
        for idx, points0 in enumerate(points_list):
            filedir0 = dec_dir[:-4]+'_part'+str(idx)+'.ply'
            dec_dir0 = dec_dir[:-4]+'_part'+str(idx)+'_dec.ply'
            bin_dir0 = bin_dir[:-4]+'_part'+str(idx)+'_dec.bin'
            write_ply_o3d(filedir0, points0, dtype='int32')
            # 
            results0 = self.test_one(filedir0, bin_dir0, dec_dir0, scale_AE=scale_AE, scale_SR=scale_SR, posQuantscale=posQuantscale,
                                    psnr_resolution=psnr_resolution, test_psnr=False)
            results_list.append(results0)
            dec_list.append(dec_dir0)
            torch.cuda.empty_cache()
        # summary
        # 1. test psnr
        points_dec = np.vstack([read_ply_o3d(dec_dir0) for dec_dir0 in dec_list])

        if points_raw.min()<0:
            points_dec = points_dec + min_point
            points_raw = points_raw + min_point

        write_ply_o3d(dec_dir, points_dec, dtype='int32')
        ref_dir = dec_dir[:-4]+'_ref.ply'
        write_ply_o3d(ref_dir, points_raw, dtype='int32', normal=True, knn=16)
        psnr_results = self.get_psnr(dec_dir, ref_dir, resolution=psnr_resolution)
        
        if self.model_offset is not None:
                
            points_dec_offset = np.vstack([read_ply_o3d(dec_dir0[:-4]+'_offset.ply') for dec_dir0 in dec_list])
            write_ply_o3d(dec_dir[:-4]+'_offset.ply', points_dec_offset, dtype='int32')
            psnr_results_offset = self.get_psnr(dec_dir[:-4]+'_offset.ply', ref_dir, resolution=psnr_resolution)

        # 2. results
        results = {'filedir':filedir, 'num_points_input':len(points_raw), 'num_points':points_dec.shape[0], 
                    'max_num':self.max_num, 'n_parts':len(points_list), 'scale_AE':scale_AE, 'scale_SR':scale_SR, 'posQuantscale':posQuantscale}
        for index in ['file_size', 'lossless_file_size', 'ae_file_size', 'enc_time', 'dec_time', 'enc_time_all', 'dec_time_all']:
            results[index] = round(sum([results0[index] for results0 in results_list]), 3)
        results['memory(GB)'] = max([results0['memory(GB)'] for results0 in results_list])
        results['bpp'] = round(results['file_size']/results['num_points_input'], 6)
        results['lossless_bpp'] = round(results['lossless_file_size']/results['num_points_input'], 6)
        results['ae_bpp'] = round(results['ae_file_size']/results['num_points_input'], 6)
        results['psnr_resolution'] = psnr_resolution
        results.update(psnr_results)
        
        if self.model_offset is not None:
            for k, v in psnr_results_offset.items(): results['offset_'+ k] = v

        return results



