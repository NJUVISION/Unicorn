# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-03

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
sys.path.append('/media/ivc3090ti/新加卷/zjz/unicorn/UnicornV1')
import time
import numpy as np
import os, glob, tqdm
import torch
import pandas as pd

from lossless_geometry.model import PCCModel as PCCModelLossless
from lossless_geometry.coder import LosslessCoderDensityAdaptive as LosslessCoder
from model import PCCModel as PCCModelLossy
from model_offset import OffsetModel
from coder import LossyCoder

from cfg.get_args import get_args 
args = get_args(component='geometry')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tester():
    def __init__(self):
        ####################### load_model #######################
        # lossless coder solid
        model_low = PCCModelLossless(channels=args.channels, kernel_size=args.kernel_size, block_layers=args.block_layers, stage=8, scale=1, block_type='conv').to(device)
        if os.path.exists(args.ckptdir_low):
            print('DBG!!! load model_low from', args.ckptdir_low)
            ckpt = torch.load(args.ckptdir_low)
            model_low.load_state_dict(ckpt['model'])
        else:
            model_low = None
        
        # lossless coder dense/sparse
        model_high = PCCModelLossless(channels=args.channels, kernel_size=5, block_layers=args.block_layers, stage=8, scale=1, block_type=args.block_type).to(device)
        if os.path.exists(args.ckptdir_high):
            print('DBG!!! load model_high from', args.ckptdir_high)
            ckpt = torch.load(args.ckptdir_high)
            model_high.load_state_dict(ckpt['model'])
        else:
            model_high = None

        # AE (dense)
        model_AE_low = PCCModelLossy(stage=1, kernel_size=args.kernel_size, enc_type='ae').to(device)
        if os.path.exists(args.ckptdir_ae_low):
            print('DBG!!! load model_AE_low from', args.ckptdir_ae_low)
            ckpt1 = torch.load(args.ckptdir_ae_low)
            model_AE_low.load_state_dict(ckpt1['model'])
        else:
            model_AE_low = None

        # AE (sparse)
        model_AE_high = PCCModelLossy(stage=1, kernel_size=5, enc_type='ae').to(device)
        if os.path.exists(args.ckptdir_ae_high):
            print('DBG!!! load model_AE_high from', args.ckptdir_ae_high)
            ckpt2 = torch.load(args.ckptdir_ae_high)
            model_AE_high.load_state_dict(ckpt2['model'])   
        else:
            model_AE_high = None

        # SR (dense)
        model_SR_low = PCCModelLossy(stage=1, kernel_size=args.kernel_size, enc_type='pooling').to(device)
        if os.path.exists(args.ckptdir_sr_low):
            print('DBG!!! load model_SR_low from', args.ckptdir_sr_low)
            ckpt1 = torch.load(args.ckptdir_sr_low)
            model_SR_low.load_state_dict(ckpt1['model'])
        else:
            model_SR_low = None

        # SR (sparse)
        model_SR_high = PCCModelLossy(stage=1, kernel_size=5, enc_type='pooling').to(device)
        if os.path.exists(args.ckptdir_sr_high):
            print('DBG!!! load model_SR_high from', args.ckptdir_sr_high)
            ckpt2 = torch.load(args.ckptdir_sr_high)
            model_SR_high.load_state_dict(ckpt2['model'])
        else:
            model_SR_high = None

        model_offset = OffsetModel(kernel_size=5).to(device)
        if os.path.exists(args.ckptdir_offset):
            print('DBG!!! load model_offset from', args.ckptdir_offset)
            ckpt = torch.load(args.ckptdir_offset)
            model_offset.load_state_dict(ckpt['model'])
        else:
            model_offset = None


        # ####################### set coder #######################
        # octree coder
        # octree_coder = OctreeCoder(device=device)
        lossless_coder = LosslessCoder(model_low, model_high, device=device)
        lossy_coder = LossyCoder(lossless_coder=lossless_coder,
                                model_AE_low=model_AE_low, model_AE_high=model_AE_high, 
                                model_SR_low=model_SR_low, model_SR_high=model_SR_high, 
                                model_offset=model_offset, device=device)
        
        self.lossy_coder = lossy_coder

        return
    
    def test_bitrates(self, filedir, idx_file):
        filename = os.path.split(filedir)[-1].split('.')[0]

        # set bitrates
        if args.bitrate_mode==-1:
            scale_AE_list = [0]
            scale_SR_list = [0]
            posQuantscale_list = [1]
        elif args.bitrate_mode==0:
            # solid/dense
            scale_SR_list =      [0, 0,1, 1,2, 2,3, 3,4]
            scale_AE_list =      [0, 1,0, 1,0, 1,0, 1,0]
            posQuantscale_list = [1, 1,1, 1,1, 1,1, 1,1]
        elif args.bitrate_mode==1:
            # sparse
            scale_SR_list =      [0, 0, 0,  0,1,1,2,2,3]
            scale_AE_list =      [0, 1, 0,  1,0,1,0,1,0]
            posQuantscale_list = [1, 1, 3,  4,4,4,4,4,4]
        
        elif args.bitrate_mode==2:
            # lidar; only scaling
            posQuantscale_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            scale_SR_list =      [0] * len(posQuantscale_list)
            scale_AE_list =      [0] * len(posQuantscale_list)

        if args.bitrate_mode==3:
            # only scaling
            posQuantscale_list = [1, 2, 4, 8, 16, 32, 64]
            scale_SR_list =      [0] * len(posQuantscale_list)
            scale_AE_list =      [0] * len(posQuantscale_list)
        
        if args.bitrate_mode==4: 
            # SparsePCGC: dense, sparse, scannet
            scale_SR_list =      [0, 0, 0, 0,1,1,2,2,3]
            scale_AE_list =      [0, 1, 1, 1,0,1,0,1,0]
            posQuantscale_list = [1, 1, 2, 4,4,4,4,4,4]

        if args.bitrate_mode==6:
            # ablation study: traverse for occupancy global with scaling
            scale_SR_list =      [0, 0, 0,0, 0,0, 0,0,1,1,2]
            scale_AE_list =      [0, 1, 0,1, 0,1, 0,1,0,1,0]
            posQuantscale_list = [1, 1, 2,2, 3,3, 4,4,4,4,4]

        print('=====set bitrates=====', '\nscale_AE_list', scale_AE_list, 
            '\nscale_SR_list', scale_SR_list, '\nposQuantscale_list', posQuantscale_list)
        
        results_list = []
        idx_rate = 0
        for scale_AE, scale_SR, posQuantscale in zip(scale_AE_list, scale_SR_list, posQuantscale_list):
            # results = {'filename':filename, 'rate':idx_rate}
            # print('DBG!!! posQuantscale', idx_rate, posQuantscale)
            bin_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.bin')
            dec_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.ply')
            idx_rate += 1
            results = self.lossy_coder.test(filedir, bin_dir, dec_dir,
                                scale_AE=scale_AE, scale_SR=scale_SR, posQuantscale=posQuantscale, 
                                psnr_resolution=args.resolution)
            torch.cuda.empty_cache()
            # results.update(results_test)
            # print('DBG!!! results', idx_rate, results)

            print('DBG!!! results', results)
            results_list.append(results)

        # collect results
        for idx_rate, results_one in enumerate(results_list):
            if idx_rate==0: filename = os.path.split(results_one['filedir'])[-1].split('.')[0]
            results_one['rate'] = idx_rate
            results_one = pd.DataFrame([results_one])
            if idx_rate==0: results = results_one.copy(deep=True)
            else: results = results.append(results_one, ignore_index=True)
            csvfile = os.path.join(args.resultsdir, str(idx_file)+'_'+filename+'.csv')
            results.to_csv(csvfile, index=False)

        return results

    def test_seqs(self, filedir_list):
        # run
        for idx_file, filedir in enumerate(tqdm.tqdm(filedir_list)):
            # skip some frame for quick test
            if idx_file < args.start_index: continue
            if args.interval > 1 and (idx_file-1) %args.interval != 0: continue
            print('filedir', idx_file, filedir)
            results = self.test_bitrates(filedir=filedir, idx_file=idx_file)

        # average
        from data_utils.pandas_utils import mean_dataframe
        csvdirs = sorted(glob.glob(os.path.join(args.resultsdir, '*.csv')))
        for i, f in enumerate(csvdirs): 
            print('\ncsvdirs:', i, f)
        results_list = [pd.read_csv(f) for f in csvdirs]
        mean_results = mean_dataframe(results_list)
        mean_results.to_csv(args.resultsdir+'.csv', index=False)
        print('DBG!!!avg:\n', mean_results)

        return mean_results


if __name__ == '__main__':
    
    args.outdir = os.path.join(args.outdir, args.prefix)
    os.makedirs(args.outdir, exist_ok=True)
    print('args.outdir\t', args.outdir)
    args.resultsdir = os.path.join(args.resultsdir, args.prefix)
    os.makedirs(args.resultsdir, exist_ok=True)
    print('DBG!!! args.resultsdir\t', args.resultsdir)

    ################# testdata #################
    filedir_list = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    
    if len(filedir_list)>args.testdata_num: 
        if args.testdata_seqs=='random':
            filedir_list = filedir_list[::len(filedir_list)//args.testdata_num]
        if args.testdata_seqs=='frame':
            filedir_list = filedir_list[:args.testdata_num]

    if args.filedir!='' and os.path.exists(args.filedir):
        filedir_list = [args.filedir]
    
    print('filedir_list length:\t', len(filedir_list))
    for i, f in enumerate(filedir_list):
        print('filedir_list', i, f)

    ################# test #################
    tester = Tester()
    tester.test_seqs(filedir_list=filedir_list)

