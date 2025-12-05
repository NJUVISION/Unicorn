# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-08

import os, sys, time
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
from data_utils.dataloaders.geometry_dataloader import load_sparse_tensor
from data_utils.geometry.quantize import quantize_sparse_tensor
from data_utils.geometry.inout import write_ply_o3d
from data_utils.sparse_tensor import sort_sparse_tensor
from third_party.gpcc_geo import gpcc_encode, gpcc_decode


from cfg.get_args import get_args 
args = get_args(component='geometry')


############################################################################################

class LosslessCoderBase():
    """basic lossless coder
    """
    def __init__(self, model, device='cuda'):  
        self.model = model
        # assert self.model.enc_type=='pooling'
        self.min_num = 32
        self.device = device
        self.CHECK = True
    
    def read_data(self, filedir, voxel_size=1):
        x = load_sparse_tensor(filedir, voxel_size=voxel_size, device=self.device)

        return x
    
    def write_data(self, x, filedir):
        write_ply_o3d(filedir, x.C[:,1:].cpu().numpy(), dtype='int32', normal=True, knn=16)

        return 

    def preprocess(self, x, posQuantscale=1):
        """scaling the coordinates
        """
        if posQuantscale==1: x_out = x
        else: x_out = quantize_sparse_tensor(x, factor=1/posQuantscale, quant_mode='round')

        if x_out.C.min() < 0:
            ref_point = x_out.C.min(axis=0)[0]
            print('DBG!!! LosslessCoderBase--preprocess--min_points', ref_point.cpu().numpy())
            x_out = ME.SparseTensor(features=x_out.F, coordinates=x_out.C - ref_point, 
                                tensor_stride=x_out.tensor_stride, device=x_out.device)
        else: ref_point = None
        self.ref_point = ref_point

        return x_out

    def postprocess(self, x, posQuantscale=1):
        if self.ref_point is not None:
            x = ME.SparseTensor(features=x.F, coordinates=x.C + self.ref_point, 
                                tensor_stride=x.tensor_stride, device=x.device)

        if posQuantscale==1: x_out = x 
        else: x_out = quantize_sparse_tensor(x, factor=posQuantscale)

        return x_out
    
    def encode(self, x):
        bitstream_list = []
        gt_list = [x]
        while len(x) > self.min_num:
            x_low = self.model.downsampler(x)
            bitstream = self.model.upsampler.encode(x_low, x)
            # print('DBG!!! bitstream:\t', len(bitstream)*8/len(x))
            bitstream_list.append(bitstream)
            x = ME.SparseTensor(features=torch.ones((len(x_low),1)).float(),
                                coordinates=torch.div(x_low.C,2,rounding_mode='floor'), 
                                device=x_low.device)
            gt_list.append(x)
        coords = x.C[:,1:].cpu().numpy().astype('int16')
        bitstream_coords = self.num2bits(coords)
        bitstream_list.append(bitstream_coords)

        if self.CHECK: self.gt_list = gt_list # DBG

        return bitstream_list

    def decode(self, bitstream_list):
        bitstream_list = bitstream_list[::-1]
        bitstream_coords = bitstream_list[0]
        coords = self.bit2num(bitstream_coords).astype('int')
        coords = torch.tensor(coords).int()
        feats = torch.ones((len(coords),1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x = ME.SparseTensor(features=feats, coordinates=coords, 
                            tensor_stride=1, device=self.device)
        for idx, bitstream in enumerate(bitstream_list[1:]):
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            x = self.model.upsampler.decode(x, bitstream)

        return x

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

    def write_bitstream(self, bitstream_list, bin_dir, dtype='uint32'):
        bitstream_all = np.array(len(bitstream_list), dtype=dtype).tobytes()
        bitstream_all += np.array([len(bitstream) for bitstream in bitstream_list], dtype=dtype).tobytes()
        for bitstream in bitstream_list:
            assert len(bitstream)<2**32-1
            bitstream_all += bitstream
        with open(bin_dir, 'wb') as f: f.write(bitstream_all)

        return os.path.getsize(bin_dir)*8
        
    def read_bitstream(self, bin_dir, dtype='uint32'):
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
    
    def encode_all(self, filedir, bin_dir, voxel_size=1, posQuantscale=1):
        start_enc_all = time.time()
        x = self.read_data(filedir, voxel_size=voxel_size)
        num_points_input = x.shape[0]
        if self.CHECK: self.x_gt = x # DBG
        x = self.preprocess(x, posQuantscale=posQuantscale)
        start_enc = time.time()
        bitstream_list = self.encode(x)
        enc_time = round(time.time() - start_enc, 3)
        if args.DBG: print('DBG!!! LosslessCoderBase -- encode_all -- bpp_list', [round(8*len(bitstream)/num_points_input, 3) for bitstream in bitstream_list])
        bits = self.write_bitstream(bitstream_list, bin_dir)
        bpp = round(bits / num_points_input, 3)
        enc_time_all = round(time.time() - start_enc_all, 3)
        results = {'filedir':filedir, 'bin_dir':bin_dir, 'posQuantscale':posQuantscale,
                   'num_points_input':num_points_input, 'file_size':bits, 'bpp':bpp, 
                    'enc_time':enc_time, 'enc_time_all':enc_time_all}

        return results
    
    def decode_all(self, bin_dir, dec_dir, posQuantscale=1):
        start_dec_all = time.time()
        bitstream_list_dec = self.read_bitstream(bin_dir)
        start_dec = time.time()
        x_dec = self.decode(bitstream_list_dec)
        dec_time = round(time.time() - start_dec, 3)
        x_dec = self.postprocess(x_dec, posQuantscale=posQuantscale)
        if self.CHECK: self.x_dec = x_dec 
        self.write_data(x_dec, dec_dir)
        dec_time_all = round(time.time() - start_dec_all, 3)
        results = {'dec_time':dec_time, 'dec_time_all':dec_time_all}
        
        return results

    
    def test(self, filedir, bin_dir, dec_dir, voxel_size=1, posQuantscale=1):
        """TODO: MISMATCH DBG
        """
        enc_results = self.encode_all(filedir, bin_dir, voxel_size=voxel_size, posQuantscale=posQuantscale)
        dec_results = self.decode_all(bin_dir, dec_dir, posQuantscale=posQuantscale)
        if not sort_sparse_tensor(self.x_gt).C.shape[0]==sort_sparse_tensor(self.x_dec).C.shape[0]:
            print('MISMATCH'+'!'*20, '\n', self.x_gt.C.shape, self.x_dec.C.shape)
        # print("DBG!!! GPU memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')

        results = enc_results
        results.update(dec_results)

        return results


############################################################################################
# density adaptive lossless coder

class LosslessCoderDensityAdaptive(LosslessCoderBase):
    def __init__(self, model_low, model_high, device='cuda'):
        """
        model_low is for dense PCs, model_high is for sparse PCs.
        """
        self.model_low = model_low
        self.model_high = model_high
        self.min_num = 32
        self.threshold = args.threshold
        self.device = device
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.CHECK = True

    def get_num_list(self, x):
        """The number of points at each scale reflect the fractal density.
        """
        x_tp = x
        num_points_list = [len(x_tp)]
        while len(x_tp) > self.min_num:
            x_tp = self.pooling(x_tp) 
            num_points_list.append(len(x_tp))
        # print('DBG!!! lossless coder encode: num_points_list:\t', num_points_list)

        return num_points_list

    def encode(self, x):
        self.num_points_list = self.get_num_list(x)

        bitstream_list = []
        gt_list = [x]

        idx = 0
        while len(x) > self.min_num:
            
            # select optimal model according to density 
            assert len(x)==self.num_points_list[idx]
            density = round(self.num_points_list[idx]/self.num_points_list[idx+1], 3)
            if density < self.threshold: self.model = self.model_high
            else: self.model = self.model_low
            if args.DBG: print('DBG!!! LosslessCoderDensityAdaptive encode threshold', idx, self.threshold, density, 'high' if density < self.threshold else 'low', self.num_points_list[idx], '-->', self.num_points_list[idx+1])
            
            x_low = self.model.downsampler(x)
            bitstream = self.model.upsampler.encode(x_low, x)
            bitstream_list.append(bitstream)
            if args.DBG: print('DBG!!! LosslessCoderDensityAdaptive encode\t', idx, len(x), '-->', len(x_low), '\tbpp:', round(8*len(bitstream)/len(x), 3))
            
            idx = idx + 1
            x = ME.SparseTensor(features=torch.ones((len(x_low),1)).float(),
                                coordinates=torch.div(x_low.C,2,rounding_mode='floor'), 
                                device=x_low.device)
            gt_list.append(x)

        coords = x.C[:,1:].cpu().numpy().astype('int16')
        bitstream_coords = self.num2bits(coords)
        bitstream_list.append(bitstream_coords)

        if self.CHECK: self.gt_list = gt_list # DBG

        return bitstream_list

    
    def decode(self, bitstream_list):
        bitstream_list = bitstream_list[::-1]
        bitstream_coords = bitstream_list[0]
        coords = self.bit2num(bitstream_coords).astype('int')
        coords = torch.tensor(coords).int()
        feats = torch.ones((len(coords),1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x = ME.SparseTensor(features=feats, coordinates=coords, 
                            tensor_stride=1, device=self.device)
        num_points_list = self.num_points_list[::-1]
        if self.CHECK: gt_list = self.gt_list[::-1]

        assert len(x)==len(gt_list[0])

        idx = 0
        for idx, bitstream in enumerate(bitstream_list[1:]):
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            len_low = len(x)

            # select optimal model according to density 
            assert len(x) == num_points_list[idx]
            density = round(num_points_list[idx+1]/num_points_list[idx], 3)
            if density < self.threshold: self.model = self.model_high
            else: self.model = self.model_low
            if args.DBG: print('DBG!!! LosslessCoderDensityAdaptive decode threshold', idx, self.threshold, density, 'high' if density < self.threshold else 'low', num_points_list[idx+1], '-->', num_points_list[idx])
            
            assert len(x)==len(gt_list[idx])
            x = self.model.upsampler.decode(x, bitstream)
            if self.CHECK and len(x)!=len(gt_list[idx+1]):
                print('DBG!!! LosslessCoderDensityAdaptive decode MISMATCH\t', len(x), len(gt_list[idx+1]))
                # assert len(x)==len(gt_list[idx+1])
                x = gt_list[idx+1]
            if args.DBG: print('DBG!!! decode', idx, len_low, '-->', len(x), '\tbpp:', round(8*len(bitstream)/len(x), 3))
            idx += 1

        return x


############################################################################################
# class LosslessCoderPartition TODO


############################################################################################
# gpcc
class OctreeCoder(LosslessCoderBase):
    """
    """
    def __init__(self, cfgdir='dense.cfg', device='cuda'):
        self.cfgdir = cfgdir
        self.device = device
    
    def load_data(self, filedir, voxel_size=1, posQuantscale=1):
        """load data & pre-quantize if posQuantscale>1
        """
        x_raw = load_sparse_tensor(filedir, voxel_size=voxel_size, device=self.device)
        x = quantize_sparse_tensor(x_raw, factor=1/posQuantscale, quant_mode='round')
        if x.C.min() < 0:
            ref_point = x.C.min(axis=0)[0]
            print('DBG!!! min_points', ref_point.cpu().numpy())
            x = ME.SparseTensor(features=x.F, coordinates=x.C - ref_point, 
                                tensor_stride=x.tensor_stride, device=x.device)
        else: ref_point = None
        self.ref_point = ref_point# TODO

        return x
    
    def encode(self, x):
        import random
        seed = random.randint(0,100)
        filedir = 'octreeCoder_ori'+str(seed)+'.ply'
        write_ply_o3d(filedir, x.C[:,1:].cpu().numpy(), dtype='int32')
        bin_dir = 'octreeCoder'+str(seed)+'.bin'
        log_enc = gpcc_encode(filedir, bin_dir, posQuantscale=1,    
                            version=args.gpcc_version, cfgdir=self.cfgdir)
        bitstream_list = [bin_dir]
        os.system('rm '+filedir)

        return bitstream_list
    
    def decode(self, bitstream_list):
        import random
        seed = random.randint(0,100)
        dec_dir = 'octreeCoder_dec'+str(seed)+'.ply'
        bin_dir = bitstream_list[0]
        log_dec = gpcc_decode(bin_dir, dec_dir, version=args.gpcc_version)
        x = load_sparse_tensor(dec_dir, voxel_size=1, device=self.device)
        os.system('rm '+dec_dir)
        os.system('rm '+bin_dir)

        return x

    def write_bitstream(self, bitstream_list, bin_dir, dtype='uint32'):
        bin_rootdir = os.path.split(bin_dir)[0]
        os.makedirs(bin_rootdir, exist_ok=True)
        os.system('mv '+bitstream_list[0]+ ' '+ bin_dir)

        return os.path.getsize(bin_dir)*8
        
    def read_bitstream(self, bin_dir, dtype='uint32'):
        bitstream_list = [bin_dir]

        return bitstream_list
    
    def test(self, filedir, bin_dir, dec_dir, voxel_size=1, posQuantscale=1):
        start = time.time()

        x = self.load_data(filedir, voxel_size, posQuantscale)
        num_points_raw = x.shape[0]

        start_enc = time.time()

        bitstream_list = self.encode(x)

        enc_time = round(time.time() - start_enc, 3)

        bits = self.write_bitstream(bitstream_list, bin_dir)
        # print('DBG!!! bpp_list', [round(8*len(bitstream)/num_points_raw, 3) for bitstream in bitstream_list])
        bpp = round(bits / num_points_raw, 3)
        enc_time_all = round(time.time() - start, 3)
        
        start = time.time()

        bitstream_list_dec = self.read_bitstream(bin_dir)

        start_dec = time.time()

        x_dec = self.decode(bitstream_list_dec)

        dec_time = round(time.time() - start_dec, 3)

        write_ply_o3d(dec_dir, x_dec.C[:,1:].cpu().numpy(), dtype='int32')

        all_dec_time = round(time.time() - start, 3)

        if not x.shape[0]==x_dec.C.shape[0]: print('MISMATCH'+'!'*20, '\n', x.C.shape, x_dec.C.shape)
        # print("DBG!!! GPU memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')

        results = {'filedir':filedir, 'num_points_raw':num_points_raw, 'file_size':bits, 'bpp':bpp, 
                'enc_time':enc_time, 'enc_time_all':enc_time_all, 'dec_time':dec_time, 'all_dec_time':all_dec_time}

        return results
