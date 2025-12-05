# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-06

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import glob, os, time
import numpy as np
import torch
import MinkowskiEngine as ME
from data_utils.dataloaders.geometry_dataloader import load_sparse_tensor
from data_utils.geometry.quantize import quantize_sparse_tensor, \
    quantize_precision, dequantize_precision, quantize_resolution, dequantize_resolution
from data_utils.geometry.inout import read_ply_o3d, write_ply_o3d, read_coords
from data_utils.sparse_tensor import sort_sparse_tensor
from third_party.pc_error_geo import pc_error


class InterCoder():
    def __init__(self, model, device='cuda'):
        self.model = model
        assert self.inter_mode==1
        self.min_num = 32

    def load_data(self, filedir, voxel_size=1, posQuantscale=1):
        """load data & pre-quantize if posQuantscale>1
        """
        x_raw = load_sparse_tensor(filedir, voxel_size=voxel_size, device=self.device)
        x = quantize_sparse_tensor(x_raw, factor=1/posQuantscale, quant_mode='round')
        ref_point = None
        self.ref_point = ref_point# TODO
        
        return x

    @torch.no_grad()
    def encode(self, x1, x0):
        bitstream_list = []
        x0_list = []
        while len(x1) > self.min_num:
            latent1_one = self.pooling(x1)
            latent1_one = ME.SparseTensor(torch.ones([latent1_one.shape[0], self.channels]).float(), 
                                    coordinate_map_key=latent1_one.coordinate_map_key, 
                                    coordinate_manager=latent1_one.coordinate_manager, 
                                    device=latent1_one.device)
            x0_list.append(x0)
            latent0 = self.model.downsampler(x0)
            latent1 = self.model.compensate(latent0, latent1_one)
            assert (latent1.C==latent1_one.C).all()

            # decode
            bitstream = self.model.upsampler.encode(latent1, x_high=x1)
            bitstream_list.append(bitstream)
            x1 = ME.SparseTensor(features=torch.ones((len(latent1_one),1)).float(),
                                coordinates=torch.div(latent1_one.C,2,rounding_mode='floor'), 
                                device=latent1_one.device)

            x0 = ME.SparseTensor(features=torch.ones((len(latent0),1)).float(),
                                coordinates=torch.div(latent0.C,2,rounding_mode='floor'), 
                                device=latent0.device)

        
        coords = x1.C[:,1:].cpu().numpy().astype('int16')
        bitstream_coords = self.num2bits(coords)
        bitstream_list.append(bitstream_coords)

        return bitstream_list, x0_list

    @torch.no_grad()
    def decode(self, bitstream_list, x0_list):
        bitstream_list = bitstream_list[::-1]
        x0_list = x0_list[::-1]
        assert len(x0_list)==len(bitstream_list)-1

        bitstream_coords = bitstream_list[0]
        coords = self.bit2num(bitstream_coords).astype('int')
        coords = torch.tensor(coords).int()
        feats = torch.ones((len(coords),1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x1 = ME.SparseTensor(features=feats, coordinates=coords, 
                            tensor_stride=1, device=self.device)
        
        for bitstream, x0 in enumerate(bitstream_list[1:], x0_list):
            x1 = ME.SparseTensor(features=x1.F, coordinates=x1.C*2,
                                tensor_stride=2, device=x1.device)

            latent1_one = ME.SparseTensor(torch.ones([x1.shape[0], self.model.channels]).float(), 
                                    coordinate_map_key=x1.coordinate_map_key, 
                                    coordinate_manager=x1.coordinate_manager, 
                                    device=x1.device)

            latent0 = self.model.downsampler(x0)
            latent1 = self.model.compensate(latent0, latent1_one, motion=motion)
            assert (latent1.C==latent1_one.C).all()

            x1 = self.model.upsampler.decode(latent1, bitstream)

        return x1

    def num2bits(self, coords, dtype='int16'):
        min_v = coords.min(axis=0)
        coords = coords - min_v
        bitstream = np.array(min_v, dtype=dtype).tobytes()
        bitstream += coords.tobytes()

        return bitstream

    def bit2num(self, bitstream, dtype='int16'):
        s = 0
        min_v = np.frombuffer(bitstream[s:s+3*2], dtype=dtype).reshape(-1,3)
        s += 3*2
        coords = np.frombuffer(bitstream[s:], dtype=dtype).reshape(-1,3)
        coords = coords + min_v

        return coords

    def pack_bitstream(self, bitstream_list, bin_dir, dtype='uint32'):
        bitstream_all = np.array(len(bitstream_list), dtype=dtype).tobytes()
        bitstream_all += np.array([len(bitstream) for bitstream in bitstream_list], dtype=dtype).tobytes()
        for bitstream in bitstream_list:
            assert len(bitstream)<2**32-1
            bitstream_all += bitstream
        with open(bin_dir, 'wb') as f: f.write(bitstream_all)

        return os.path.getsize(bin_dir)*8
        
    def unpack_bitstream(self, bin_dir, dtype='uint32'):
        with open(bin_dir, 'rb') as fin: bitstream_all = fin.read()
        s = 0
        num = np.frombuffer(bitstream_all[s:s+1*4], dtype=dtype)[0]
        s += 1*4
        lengths = np.frombuffer(bitstream_all[s:s+num*4], dtype=dtype)
        s += num*4
        bitstream_list = []
        for l in lengths:
            bitstream = bitstream_all[s:s+l]
            bitstream_list.append(bitstream)
            s += l

        return bitstream_list

    @torch.no_grad()
    def test(self, filedir, filedir_ref, bin_dir, dec_dir, voxel_size=1, posQuantscale=1):
        start = time.time()

        x = self.load_data(filedir, voxel_size, posQuantscale)
        num_points_raw = x.shape[0]
        ref = self.load_data(filedir_ref, voxel_size, posQuantscale)
        out_set_list = self.model_low.forward(ref, x, training=False)

        start_enc = time.time()
        bitstream_list, x0_list = self.encode(x1=x, x0=ref)
        enc_time = round(time.time() - start_enc, 3)
        # print('DBG!!! bpp_list', [round(8*len(bitstream)/num_points_raw, 3) for bitstream in bitstream_list])
        bits = self.pack_bitstream(bitstream_list, bin_dir)
        all_enc_time = round(time.time() - start, 3)
        bpp = round(bits / num_points_raw, 3)

        start = time.time()
        bitstream_list_dec = self.unpack_bitstream(bin_dir)
        start_dec = time.time()

        x_dec = self.decode(bitstream_list_dec, x0_list)

        dec_time = round(time.time() - start_dec, 3)
        write_ply_o3d(dec_dir, x_dec.C[:,1:].cpu().numpy(), dtype='int32')
        all_dec_time = round(time.time() - start, 3)
        if not sort_sparse_tensor(x).C.shape[0]==sort_sparse_tensor(x_dec).C.shape[0]:
            print('MISMATCH'+'!'*20)
            print(x.C.shape, x_dec.C.shape)
        # print("DBG!!! GPU memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')

        results = {'filedir':filedir, 'num_points_raw':num_points_raw, 'file_size':bits, 'bpp':bpp, 
                'enc_time':enc_time, 'all_enc_time':all_enc_time, 'dec_time':dec_time, 'all_dec_time':all_dec_time}

        return results

################################################################################

class InterCoder2(InterCoder):
    def __init__(self, model_low, model_high, device='cuda'):
        self.model_low = model_low
        self.model_high = model_high
        self.min_num = 32
        self.threshold = 12000#!!!
        self.device = device

    @torch.no_grad()
    def encode(self, x1, x0):
        bitstream_list = []
        idx = 0

        x0_list = []
        while len(x1) > self.min_num:

            max_value = (x1.C.max(dim=0)[0] - x1.C.min(dim=0)[0]).max().cpu()
            if max_value > self.threshold:
                self.model = self.model_high
            else:
                self.model = self.model_low

            latent1_one = self.model.pooling(x1)
            latent1_one = ME.SparseTensor(torch.ones([latent1_one.shape[0], self.model.channels]).float(), 
                                    coordinate_map_key=latent1_one.coordinate_map_key, 
                                    coordinate_manager=latent1_one.coordinate_manager, 
                                    device=latent1_one.device)
            
            x0_list.append(x0)

            latent0 = self.model.downsampler(x0)

            latent1 = self.model.compensate(latent0, latent1_one)
            assert (latent1.C==latent1_one.C).all()

            # decode
            bitstream = self.model.upsampler.encode(latent1, x_high=x1)
            bitstream_list.append(bitstream)
            # print('DBG!!! encode', idx, len(x1), '-->', len(latent1_one), '\tmax_value:', max_value.cpu().numpy(),
            #     '\tbits/bpp:', len(bitstream), round(8*len(bitstream)/len(x1), 3),  'high' if max_value > self.threshold else 'low')
            idx = idx + 1

            x1 = ME.SparseTensor(features=torch.ones((len(latent1_one),1)).float(),
                                coordinates=torch.div(latent1_one.C,2,rounding_mode='floor'), 
                                device=latent1_one.device)

            x0 = ME.SparseTensor(features=torch.ones((len(latent0),1)).float(),
                                coordinates=torch.div(latent0.C,2,rounding_mode='floor'), 
                                device=latent0.device)

        coords = x1.C[:,1:].cpu().numpy().astype('int16')
        bitstream_coords = self.num2bits(coords)
        bitstream_list.append(bitstream_coords)

        return bitstream_list, x0_list

    @torch.no_grad()
    def decode(self, bitstream_list, x0_list):
        bitstream_list = bitstream_list[::-1]
        x0_list = x0_list[::-1]
        assert len(x0_list)==len(bitstream_list)-1

        bitstream_coords = bitstream_list[0]
        coords = self.bit2num(bitstream_coords).astype('int')
        coords = torch.tensor(coords).int()
        feats = torch.ones((len(coords),1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x1 = ME.SparseTensor(features=feats, coordinates=coords, 
                            tensor_stride=1, device=self.device)
        
        idx = 0
        for bitstream, x0 in zip(bitstream_list[1:], x0_list):
            x1 = ME.SparseTensor(features=x1.F, coordinates=x1.C*2,
                                tensor_stride=2, device=x1.device)

            len_low = len(x1)
            max_value = (x1.C.max(dim=0)[0] - x1.C.min(dim=0)[0]).max().cpu()
            if max_value > self.threshold: 
                self.model = self.model_high
            else:
                self.model = self.model_low

            latent1_one = ME.SparseTensor(torch.ones([x1.shape[0], self.model.channels]).float(), 
                                    coordinate_map_key=x1.coordinate_map_key, 
                                    coordinate_manager=x1.coordinate_manager, 
                                    device=x1.device)

            latent0 = self.model.downsampler(x0)
            latent1 = self.model.compensate(latent0, latent1_one)
            assert (latent1.C==latent1_one.C).all()

            x1 = self.model.upsampler.decode(latent1, bitstream)
            idx = idx + 1
            
        return x1

    def num2bits(self, coords, dtype='int16'):
        min_v = coords.min(axis=0)
        coords = coords - min_v
        bitstream = np.array(min_v, dtype=dtype).tobytes()
        bitstream += coords.tobytes()

        return bitstream

    def bit2num(self, bitstream, dtype='int16'):
        s = 0
        min_v = np.frombuffer(bitstream[s:s+3*2], dtype=dtype).reshape(-1,3)
        s += 3*2
        coords = np.frombuffer(bitstream[s:], dtype=dtype).reshape(-1,3)
        coords = coords + min_v

        return coords


################################################################################
class LossyCoder():
    """for lidar point clouds (kitti, ford)
    """
    def __init__(self, basic_coder, model_offset=None, device='cuda'):
        self.basic_coder = basic_coder
        self.model_offset = model_offset
        self.device = device


    @torch.no_grad()
    def downscale(self, x, posQuantscale=1):
        x_down = quantize_sparse_tensor(x, factor=1/posQuantscale, quant_mode='round')
        
        return x_down
    
    @torch.no_grad()
    def upscale(self, x, posQuantscale):
        # if self.model_offset is None:
        x_dec = quantize_sparse_tensor(x, factor=posQuantscale)
        coords_dec = x_dec.C[:,1:].cpu().numpy()

        if self.model_offset is not None:
            coords_offset = self.model_offset.upscale(x, posQuantscale=posQuantscale)
        else:
            coords_offset = coords_dec

        return coords_dec, coords_offset

    @torch.no_grad()
    def prequantize(self, points_raw, quant_mode=None, quant_factor=1):
        if quant_mode is None:
            points_q = points_raw
        elif quant_mode=='precision':
            points_q = quantize_precision(points_raw, precision=quant_factor, return_offset=False)
            meta_info = {}
        elif quant_mode=='resolution':
            points_q, max_bound, min_bound = quantize_resolution(points_raw, resolution=quant_factor, return_offset=False)
            meta_info = {'max_bound':max_bound, 'min_bound':min_bound}
        points_q = np.unique(points_q, axis=0).astype('int32')
        ref_point = None
        meta_info['ref_point'] = ref_point

        return points_q, meta_info

    @torch.no_grad()
    def postquantize(self, points_q, meta_info, quant_mode=None, quant_factor=1):
        ref_point = meta_info['ref_point']
        if ref_point is not None:
            points_q = points_q + ref_point
        if quant_mode is None:
            points = points_q
        elif quant_mode=='precision':
            points = dequantize_precision(points_q, precision=quant_factor)
        elif quant_mode=='resolution':
            max_bound, min_bound = meta_info['max_bound'], meta_info['min_bound']
            points = dequantize_resolution(points_q, max_bound=max_bound, min_bound=min_bound, resolution=quant_factor)        

        return points
    
    @torch.no_grad()
    def test(self, filedir, filedir_ref, bin_dir, dec_dir, posQuantscale=1, quant_mode=None, quant_factor=1, psnr_mode='gpcc', test_d2=False):
        start = time.time()

        points_raw = read_coords(filedir)
        num_points_raw = points_raw.shape[0]

        points_raw_ref = read_coords(filedir_ref)

        # preprocessing: quantize if input raw float points
        points_q, meta_info = self.prequantize(points_raw, quant_mode=quant_mode, quant_factor=quant_factor)
        points_q_ref, _ = self.prequantize(points_raw_ref, quant_mode=quant_mode, quant_factor=quant_factor)

        start_enc = time.time()
        # sparse tensor
        coords = torch.tensor(points_q).int()
        feats = torch.ones((len(points_q),1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=self.device)
        # downscale
        x_down = self.downscale(x, posQuantscale)


        coords_ref = torch.tensor(points_q_ref).int()
        feats_ref = torch.ones((len(points_q_ref),1)).float()
        coords_ref, feats_ref = ME.utils.sparse_collate([coords_ref], [feats_ref])
        x_ref = ME.SparseTensor(features=feats_ref, coordinates=coords_ref, tensor_stride=1, device=self.device)
        # downscale
        x_down_ref = self.downscale(x_ref, posQuantscale)

        # encode
        bitstream_list, x0_list = self.basic_coder.encode(x_down, x0=x_down_ref)
        enc_time = round(time.time() - start_enc, 3)
        bits = self.basic_coder.pack_bitstream(bitstream_list, bin_dir)
        all_enc_time = round(time.time() - start, 3)
        start = time.time()
        # decode
        bitstream_list_dec = self.basic_coder.unpack_bitstream(bin_dir)
        start_dec = time.time()
        x_dec = self.basic_coder.decode(bitstream_list_dec, x0_list)

        if len(x_down)==len(x_dec):
            if (sort_sparse_tensor(x_down).C==sort_sparse_tensor(x_dec).C).all():
                print('correct')
            else:
                print('!'*600, 'x_down.C!=x_dec.C')
                x_dec = x_down
        else:
            print('!'*600, 'len(x_down)!=len(x_dec)')
            x_dec = x_down

        assert (sort_sparse_tensor(x_down).C==sort_sparse_tensor(x_dec).C).all()

        # upscale
        coords_dec, coords_offset = self.upscale(x_dec, posQuantscale)
        points_dec = self.postquantize(coords_dec, meta_info, quant_mode=quant_mode, quant_factor=quant_factor)
        dec_time = round(time.time() - start_dec, 3)
        write_ply_o3d(dec_dir, points_dec, dtype='float32')
        all_dec_time = round(time.time() - start, 3)
        bpp = round(bits / num_points_raw, 3)
        print('bpp:\t', bpp, num_points_raw)
        print("memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')
        # metric psnr
        ref_dir = dec_dir[:-4]+'_ref.ply'
        if test_d2:
            write_ply_o3d(ref_dir, points_raw, normal=True, knn=16)# test d2
        else:
            write_ply_o3d(ref_dir, points_raw, dtype='float32')
        psnr_results = pc_error(ref_dir, dec_dir, resolution=30000, normal=test_d2, show=False)
        print('psnr:\t', psnr_results)

        results = {'filedir':filedir, 'num_points_raw':num_points_raw, 'file_size':bits, 'bpp':bpp, 'num_points':points_dec.shape[0],
                'enc_time':enc_time, 'all_enc_time':all_enc_time, 'dec_time':dec_time, 'all_dec_time':all_dec_time}
        results.update(psnr_results)

        if self.model_offset is not None:
            points_offset = self.postquantize(coords_offset, meta_info, quant_mode=quant_mode, quant_factor=quant_factor)
            offset_dir = dec_dir[:-4]+'_offset.ply'
            write_ply_o3d(offset_dir, points_offset, dtype='float32')
            
            assert psnr_mode=='gpcc'
            psnr_results_offset = pc_error(ref_dir, offset_dir, resolution=30000, normal=test_d2, show=False)
            for k, v in psnr_results_offset.items(): results['offset_'+ k] = v

        return results
