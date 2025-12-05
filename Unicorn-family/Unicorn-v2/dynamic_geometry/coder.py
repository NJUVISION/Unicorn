# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-06

import os, sys

sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import glob, time
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME
import pandas as pd

from data_utils.geometry.partition import kdtree_partition
from data_utils.geometry.inout import write_ply_o3d
from data_utils.dataloaders.geometry_dataloader import load_sparse_tensor
from data_utils.sparse_tensor import sort_sparse_tensor
from third_party.pc_error_geo import pc_error
from basic_models.loss import get_bce, get_bits
from third_party.gpcc_geo import gpcc_encode, gpcc_decode

from model_lossless import PCCModel
from model_lossy import PCCModel as PCCModelLossy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def downscale(x):
    return ME.SparseTensor(features=x.F, coordinates=x.C // x.tensor_stride[0], tensor_stride=1, device=x.device)


def upscale(x):
    return ME.SparseTensor(features=x.F, coordinates=x.C * 2,
                           tensor_stride=x.tensor_stride[0] * 2, device=x.device)


def load_ckpt(model, ckptdir):
    if ckptdir == '':
        model = None
    else:
        ckpt = torch.load(ckptdir)
        model.load_state_dict(ckpt['model'])
    print('load model from', ckptdir)

    return model


class CoderLossless():
    def __init__(self, ckptdir, mode, outdir='output'):
        # lossless model
        self.model = PCCModel(mode=mode, stage=8, scale=6).to(device)
        self.model = load_ckpt(self.model, ckptdir)
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def gpcc_encode(self, x, filename='tp'):
        """lossless compression by G-PCC."""
        filedir = os.path.join(self.outdir, filename + '_gpcc.ply')
        write_ply_o3d(filedir, x.C[:, 1:].cpu().numpy(), dtype='int32')
        bin_dir = os.path.join(self.outdir, filename + '_gpcc.bin')
        _ = gpcc_encode(filedir, bin_dir, posQuantscale=1, version=21, cfgdir='dense.cfg')
        # bits = os.path.getsize(bin_dir)*8

        return bin_dir

    def gpcc_decode(self, bin_dir):
        filename = os.path.split(bin_dir)[-1].split('.')[0]
        dec_dir = os.path.join(self.outdir, filename + '_dec.ply')
        _ = gpcc_decode(bin_dir, dec_dir, version=21)
        x = load_sparse_tensor(dec_dir, voxel_size=1, device=device)

        return x

    @torch.no_grad()
    def encode(self, x0, x1, scale=6, filename='tp'):
        """lossless compression by 8stage SOPA model;
        """
        assert x0.tensor_stride[0] == 1 and x1.tensor_stride[0] == 1
        if self.model is None:
            gpcc_bin_dir = self.gpcc_encode(x1, filename=filename)
            bitstream_list = []
        # encode by 8stage SOPA
        scale = min(scale, np.floor(np.log2(x1.C.max().item())).astype('int') - 3)
        bitstream_list, x1_low = self.model.encode(x0, x1, scale=scale)
        # write into file
        bin_dir_list = []
        for idx_scale, bitstream in enumerate(bitstream_list):
            bin_dir = os.path.join(self.outdir, filename + '_s' + str(idx_scale) + '.bin')
            with open(bin_dir, 'wb') as f: f.write(bitstream)
            bin_dir_list.append(bin_dir)
        # encode by G-PCC
        gpcc_bin_dir = self.gpcc_encode(x1_low, filename=filename)

        bin_dir_list = bin_dir_list + [gpcc_bin_dir]

        return bin_dir_list

    @torch.no_grad()
    def decode(self, x0, bin_dir_list):
        gpcc_bin_dir = bin_dir_list[-1]
        x1 = self.gpcc_decode(gpcc_bin_dir)
        bitstream_list = []
        for idx_scale, bin_dir in enumerate(bin_dir_list[:-1]):
            with open(bin_dir, 'rb') as fin: bitstream = fin.read()
            bitstream_list.append(bitstream)
        x1 = self.model.decode(x0, x1, bitstream_list)

        return x1

    def test(self, x0, x1, scale=6, filename='tp'):
        num_points = x1.C.shape[0]
        # encode
        start = time.time()
        bin_dir_list = self.encode(x0, x1, scale=scale, filename=filename)
        enc_time = round(time.time() - start, 3)
        #
        bits_list = [os.path.getsize(bin_dir) * 8 for bin_dir in bin_dir_list]
        bpp_list = np.array(bits_list) / num_points

        # decode
        start_dec = time.time()
        x1_dec = self.decode(x0, bin_dir_list)
        dec_time = round(time.time() - start_dec, 3)
        assert (sort_sparse_tensor(x1).C == sort_sparse_tensor(x1_dec).C).all()

        return {'bits': sum(bits_list[:-1]), 'bits_gpcc': bits_list[-1],
                'enc_time': enc_time, 'dec_time': dec_time}


class CoderLossy():
    def __init__(self, ckptdir, mode, scale, outdir='output'):
        if scale in [1, 2, 3]:
            self.model = PCCModelLossy(mode=mode, scale=scale).to(device)
            self.model = load_ckpt(self.model, ckptdir)
        else:
            self.model = None
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    @torch.no_grad()
    def encode(self, pc1, filename='tp'):
        x1, bitstream, nums_list = self.model.encode(pc1)
        bin_dir = os.path.join(self.outdir, filename + '_lossy.bin')
        with open(bin_dir, 'wb') as f: f.write(bitstream)

        return x1, bin_dir, nums_list

    @torch.no_grad()
    def decode(self, pc0, x1, bin_dir, nums_list):
        with open(bin_dir, 'rb') as fin: bitstream = fin.read()
        x1_dec = self.model.decode(pc0, x1, bitstream, nums_list)

        return x1_dec

    @torch.no_grad()
    def test(self, x0, x1, filename='tp'):
        assert x0.tensor_stride[0] == 1 and x1.tensor_stride[0] == 1
        # encode
        start = time.time()
        x1_low, bin_dir, nums_list = self.encode(x1, filename=filename)
        enc_time = round(time.time() - start, 3)
        bits = os.path.getsize(bin_dir) * 8 + len(nums_list) * 32
        # decode
        start = time.time()
        x1_dec = self.decode(x0, x1_low, bin_dir, nums_list)
        print('dec time:\t', round(time.time() - start, 3))
        #
        x0_low = x0
        for i in range(self.model.scale):
            x0_low = self.pooling(x0_low)
        x1_low = downscale(x1_low)
        x0_low = downscale(x0_low)

        return {'bits': bits, 'x_dec': x1_dec, 'x1_low': x1_low, 'x0_low': x0_low}


class CoderSR():
    def __init__(self, ckptdir, mode, outdir='output'):
        # lossless model
        self.model = PCCModel(mode=mode, stage=1, scale=1).to(device)
        self.model = load_ckpt(self.model, ckptdir)
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    @torch.no_grad()
    def downscale(self, x0, x1, scale=1):
        nums_list = []
        for i in range(scale):
            nums_list.append(len(x1))
            x0 = self.pooling(x0)
            x1 = self.pooling(x1)
        x0 = downscale(x0)
        x1 = downscale(x1)

        return x0, x1, nums_list

    @torch.no_grad()
    def upscale(self, x0, x1_low, nums_list):
        scale = len(nums_list)
        assert x0.tensor_stride[0] == 1 and x1_low.tensor_stride[0] == 1
        # downscale
        x0_list = [x0]
        for i in range(scale):
            x0 = self.pooling(x0)
            x0_list.append(x0)
        # upscale
        x1_dec = x1_low
        for i in range(scale):
            x0 = x0_list[-2 - i];
            x0 = downscale(x0)  # reference frame
            num_points = nums_list[-1 - i]
            x1_dec = upscale(x1_dec)
            assert x0.tensor_stride[0] == 1 and x1_dec.tensor_stride[0] == 2
            x1_dec = self.model.upsample(x0, latent1=x1_dec, num_points=num_points)

        return x1_dec


class CoderMultiscale():
    def __init__(self, mode, ckptdir_lossless, ckptdir_lossy, ckptdir_sr,
                 scale_lossless=6, scale_lossy=3, scale_sr=0, outdir='output'):
        """   """
        self.scale_lossless = scale_lossless
        self.scale_lossy = scale_lossy
        self.scale_sr = scale_sr
        # coder
        self.coder_lossless = CoderLossless(ckptdir=ckptdir_lossless, mode=mode, outdir=outdir)
        self.coder_lossy = CoderLossy(ckptdir=ckptdir_lossy, mode=mode, scale=scale_lossy, outdir=outdir)
        self.coder_sr = CoderSR(ckptdir=ckptdir_sr, mode=mode, outdir=outdir)
        #
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.unpooling = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)

    @torch.no_grad()
    def encode(self, pc0, pc1, filename='tp'):
        x0, x1 = pc0, pc1
        # sr
        if self.scale_sr != 0:
            x0, x1, nums_list_sr = self.coder_sr.downscale(x0, x1, scale=self.scale_sr)
        else:
            nums_list_sr = []
        # lossy
        if self.scale_lossy != 0:
            x1, bin_dir_lossy, nums_list_lossy = self.coder_lossy.encode(x1, filename=filename)
            for i in range(self.scale_lossy): x0 = self.pooling(x0)
            x0 = downscale(x0)
        else:
            bin_dir_lossy, nums_list_lossy = 0, []
        # lossless
        bin_dir_list = self.coder_lossless.encode(x0, x1, scale=self.scale_lossless, filename=filename)
        # out (to-be-encoded)
        out_set = {'bin_dir_list': bin_dir_list, 'bin_dir_lossy': bin_dir_lossy,
                   'nums_list_lossy': nums_list_lossy, 'nums_list_sr': nums_list_sr}
        # log
        results = {'bits_gpcc': os.path.getsize(bin_dir_list[-1]) * 8,
                   'bits_lossless': sum([os.path.getsize(bin_dir) * 8 for bin_dir in bin_dir_list[:-1]]),
                   'bits_lossy': os.path.getsize(bin_dir_lossy) * 8 + len(nums_list_lossy + nums_list_sr) * 32}
        results['bits'] = results['bits_gpcc'] + results['bits_lossless'] + results['bits_lossy']

        return out_set, results

    @torch.no_grad()
    def decode(self, pc0, out_set):
        # downscale pc0
        x0 = pc0
        for i in range(self.scale_sr):
            x0 = self.pooling(x0)
        x0_lossy = downscale(x0)
        #
        for i in range(self.scale_lossy):
            x0 = self.pooling(x0)
        x0 = downscale(x0)
        # decode
        x1 = self.coder_lossless.decode(x0, out_set['bin_dir_list'])
        if self.scale_lossy != 0: x1 = self.coder_lossy.decode(x0_lossy, x1, out_set['bin_dir_lossy'],
                                                               out_set['nums_list_lossy'])
        if self.scale_sr != 0: x1 = self.coder_sr.upscale(pc0, x1, out_set['nums_list_sr'])

        return x1

    @torch.no_grad()
    def test_one(self, pc0, pc1, filename='tp'):
        # encode
        start = time.time()
        out_set, results = self.encode(pc0, pc1, filename=filename)
        enc_time = time.time() - start
        # deocode
        start = time.time()
        pc_dec = self.decode(pc0, out_set)
        dec_time = time.time() - start
        # bpp
        num_points = len(pc1)
        results.update({'num_points': num_points,
                        'bpp_gpcc': round(results['bits_gpcc'] / num_points, 3),
                        'bpp_lossless': round(results['bits_lossless'] / num_points, 3),
                        'bpp_lossy': round(results['bits_lossy'] / num_points, 3),
                        'bpp': round(results['bits'] / num_points, 3)})
        results.update({'enc_time': round(enc_time, 3), 'dec_time': round(dec_time, 3)})

        return pc_dec, results

    @torch.no_grad()
    def test(self, pc0, pc1, max_num=1e6, filename='tp'):
        if len(pc1) > max_num:
            n_parts = 2 ** round(np.ceil(np.log2(len(pc1) / max_num)))

            points0 = pc0.C[:, 1:].detach().cpu().numpy()
            points0_list = kdtree_partition(points0, max_num=max_num, n_parts=n_parts)

            points1 = pc1.C[:, 1:].detach().cpu().numpy()
            points1_list = kdtree_partition(points1, max_num=max_num, n_parts=n_parts)
        else:
            n_parts = 1
            points0_list = [pc0.C[:, 1:].detach().cpu().numpy()]
            points1_list = [pc1.C[:, 1:].detach().cpu().numpy()]

        #
        x_dec_list = []
        results_list = []
        for idx in range(n_parts):
            coords0 = points0_list[idx]
            coords1 = points1_list[idx]
            #
            coords0 = torch.tensor(coords0).int()
            feats0 = torch.ones((len(coords0), 1)).float()
            coords0, feats0 = ME.utils.sparse_collate([coords0], [feats0])
            x0 = ME.SparseTensor(features=feats0, coordinates=coords0,
                                 tensor_stride=1, device=pc0.device)
            #
            coords1 = torch.tensor(coords1).int()
            feats1 = torch.ones((len(coords1), 1)).float()
            coords1, feats0 = ME.utils.sparse_collate([coords1], [feats1])
            x1 = ME.SparseTensor(features=feats1, coordinates=coords1,
                                 tensor_stride=1, device=pc1.device)
            #
            x_dec, results = self.test_one(x0, x1, filename=filename + '_part' + str(idx))
            x_dec_list.append(x_dec)
            results_list.append(results)
        # collect
        points_dec = np.vstack([tp.C[:, 1:].detach().cpu().numpy() for tp in x_dec_list])
        coords_dec = torch.tensor(points_dec).int()
        feats_dec = torch.ones((len(coords_dec), 1)).float()
        coords_dec, feats_dec = ME.utils.sparse_collate([coords_dec], [feats_dec])
        x_dec = ME.SparseTensor(features=feats_dec, coordinates=coords_dec,
                                tensor_stride=1, device=pc1.device)
        #
        results = {'num_points': len(pc1), 'max_num': max_num, 'n_parts': n_parts}
        for index in ['bits', 'bits_gpcc', 'bits_lossless', 'bits_lossy', 'enc_time', 'dec_time']:
            results[index] = round(sum([results0[index] for results0 in results_list]), 3)
        #
        memoey = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        results.update({'bpp': round(results['bits'] / results['num_points'], 6),
                        'bpp_gpcc': round(results['bits_gpcc'] / results['num_points'], 6),
                        'bpp_lossless': round(results['bits_lossless'] / results['num_points'], 6),
                        'bpp_lossy': round(results['bits_lossy'] / results['num_points'], 6)})
        results['memory(GB)'] = memoey

        return x_dec, results


class CoderMultiframe():
    def __init__(self, mode='intra',
                 ckptdir_lossless0='', ckptdir_lossy0='', ckptdir_sr0='', scale_lossless0=6, scale_lossy0=3,
                 scale_sr0=0,
                 ckptdir_lossless1='', ckptdir_lossy1='', ckptdir_sr1='', scale_lossless1=6, scale_lossy1=3,
                 scale_sr1=0,
                 outdir='output', resultsdir='results'):
        """
        """
        self.outdir = outdir
        self.resultsdir = resultsdir
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.resultsdir, exist_ok=True)
        #
        self.mode = mode
        self.coder_intra = CoderMultiscale(mode='intra',
                                           ckptdir_lossless=ckptdir_lossless0, ckptdir_lossy=ckptdir_lossy0,
                                           ckptdir_sr=ckptdir_sr0,
                                           scale_lossless=scale_lossless0, scale_lossy=scale_lossy0, scale_sr=scale_sr0,
                                           outdir=outdir)
        if mode == 'inter':
            self.coder_inter = CoderMultiscale(mode='inter',
                                               ckptdir_lossless=ckptdir_lossless1, ckptdir_lossy=ckptdir_lossy1,
                                               ckptdir_sr=ckptdir_sr1,
                                               scale_lossless=scale_lossless1, scale_lossy=scale_lossy1,
                                               scale_sr=scale_sr1, outdir=outdir)

    def write_file(self, x_ori, x_dec, filename):
        # write files
        points_ori = x_ori.C[:, 1:].detach().cpu().numpy()
        ori_dir = os.path.join(self.outdir, filename + '_ori.ply')
        write_ply_o3d(ori_dir, points_ori, normal=True, dtype='int32', knn=16)
        #
        points_dec = x_dec.C[:, 1:].detach().cpu().numpy()
        dec_dir = os.path.join(self.outdir, filename + '_dec.ply')
        write_ply_o3d(dec_dir, points_dec, dtype='int32')

        return ori_dir, dec_dir

    @torch.no_grad()
    def test(self, seqs_list, voxel_size=2, quant_mode='floor', prefix='tp', test_psnr=True, test_d3=False):
        # seqs_list = sorted(glob.glob(os.path.join(rootdir, '**', f'*.ply'), recursive=True))
        for idx_file, filedir in enumerate(tqdm(seqs_list)):
            print('=' * 20, idx_file, filedir)
            filename = prefix + '_' + os.path.split(filedir)[-1][:-4]
            if idx_file == 0:
                pc1 = load_sparse_tensor(filedir, voxel_size=voxel_size, quant_mode=quant_mode)
                x_dec, results = self.coder_intra.test(pc1, pc1, filename=filename)
            else:
                pc0 = load_sparse_tensor(dec_dir)  # the previous decoded frame
                pc1 = load_sparse_tensor(filedir, voxel_size=voxel_size, quant_mode=quant_mode)
                if self.mode == 'inter':
                    x_dec, results = self.coder_inter.test(pc0, pc1, filename=filename)
                if self.mode == 'intra':
                    x_dec, results = self.coder_intra.test(pc0, pc1, filename=filename)
            results.update({'filedir': filedir})
            # psnr
            ori_dir, dec_dir = self.write_file(x_ori=pc1, x_dec=x_dec, filename=filename)
            if test_psnr:
                max_value = x_dec.C[:, 1:].detach().cpu().numpy()
                if np.log2(max_value.max()) < 10:
                    resolution = 1023
                elif np.log2(max_value.max()) < 11:
                    resolution = 2047
                psnr_results = pc_error(ori_dir, dec_dir, resolution=resolution, normal=True, show=False)
                psnr_results['resolution'] = resolution
                results.update(psnr_results)
                # d3 psnr
                if test_d3:
                    from third_party.d3metric.d3_utils import load_geometry
                    from third_party.d3metric.metrics import d3psnr
                    pts_A = load_geometry(ori_dir, False)
                    pts_B = load_geometry(dec_dir, False)
                    print("Calculate d3-PSNR (A->A|B)")
                    d3_mse1, d3_psnr1, nPtsA, nPtsB, radiusA_A, radiusB_A, blk_size = d3psnr(pts_A, pts_B,
                                                                                             bitdepth_psnr=16,
                                                                                             mapsfolder='',
                                                                                             mapsprefix='density_map',
                                                                                             diffradius=False,
                                                                                             localaverage=False)
                    d3_results = {'d3_psnr': round(d3_psnr1, 4), 'd3_mse1': round(d3_mse1, 4)}
                    results.update(d3_results)

            # collection
            print('results', results)
            results = pd.DataFrame([results])
            if idx_file == 0:
                all_results = results.copy(deep=True)
            else:
                all_results = pd.concat([all_results,results], ignore_index=True)
            csvfile = os.path.join(self.resultsdir, prefix + '_data' + str(len(seqs_list)) + '.csv')
            all_results.to_csv(csvfile, index=False)
            torch.cuda.empty_cache()
        # print('save results to ', csvfile)
        # print(all_results.mean())

        return
