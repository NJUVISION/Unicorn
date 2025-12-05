# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-08

import os, sys, time, glob, tqdm
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import numpy as np
import torch
import pandas as pd
from model import PCCModel
from coder import LosslessCoderDensityAdaptive
from third_party.pc_error_geo import pc_error
from data_utils.pandas_utils import mean_dataframe

from cfg.get_args import get_args 
args = get_args(component='geometry')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pipelines.tester_base import Tester
class LosslessTester(Tester):
    def __init__(self):
        #######################  load_model  #######################
        # dense
        model_low = PCCModel(channels=args.channels, kernel_size=args.kernel_size, block_layers=args.block_layers, stage=8, scale=1, block_type='conv').to(device)
        if os.path.exists(args.ckptdir_low):
            print('DBG!!! load model_low from', args.ckptdir_low)
            ckpt = torch.load(args.ckptdir_low)
            model_low.load_state_dict(ckpt['model'])
        else:
            model_low = None
        
        # sparse
        model_high = PCCModel(channels=args.channels, kernel_size=5, block_layers=args.block_layers, stage=8, scale=1, block_type=args.block_type).to(device)
        if os.path.exists(args.ckptdir_high):
            print('DBG!!! load model_high from', args.ckptdir_high)
            ckpt = torch.load(args.ckptdir_high)
            model_high.load_state_dict(ckpt['model'])
        else:
            model_high = None

        # ####################### set coder #######################
        # lossless_coder = OctreeCoder(device=device)
        lossless_coder = LosslessCoderDensityAdaptive(model_low, model_high, device=device)
        self.lossless_coder = lossless_coder

        return
    
    def get_PSNR(self, ref_dir, dec_dir, resolution=1023, test_d3=False):
        psnr_results = pc_error(ref_dir, dec_dir, resolution=resolution, normal=True, show=False)
        if test_d3: psnr_results = self.get_d3_PSNR(ref_dir, dec_dir, psnr_results)

        return psnr_results
    
    def get_d3_PSNR(self, ref_dir, dec_dir, psnr_results):
        # d3 psnr
        from third_party.d3metric.d3_utils import load_geometry
        from third_party.d3metric.metrics import d3psnr
        pts_A = load_geometry(ref_dir, False)
        pts_B = load_geometry(dec_dir, False)
        print("LosslessTester Calculate d3-PSNR (A->A|B)")
        d3_mse1, d3_psnr1, nPtsA, nPtsB, radiusA_A, radiusB_A, blk_size = d3psnr(
            pts_A, pts_B, bitdepth_psnr=16, mapsfolder='', mapsprefix='density_map', diffradius=False, localaverage=False)
        d3_results = {'d3_psnr':round(d3_psnr1, 4), 'd3_mse':round(d3_mse1, 4)}
        psnr_results.update(d3_results)

        return psnr_results

    def test_bitrates(self, filedir):
        # set bitrates
        if args.bitrate_mode==0:
            posQuantscale_list = [1]
        elif args.bitrate_mode==1:
            # only scaling (lidar)
            posQuantscale_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        for idx_rate, posQuantscale in enumerate(posQuantscale_list):
            filename = os.path.split(filedir)[-1].split('.')[0]
            bin_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.bin')
            dec_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.ply')
            results = self.lossless_coder.test(filedir, bin_dir, dec_dir, posQuantscale=posQuantscale)
            torch.cuda.empty_cache()
            psnr_results = self.get_PSNR(filedir, dec_dir, resolution=args.resolution)
            results.update(psnr_results)
            results['rate'] = idx_rate
            print('DBG!!! LosslessTester--test_bitrates--results', results)

            # collect results
            results = pd.DataFrame([results])
            if idx_rate==0: results_bitrates = results.copy(deep=True)
            else: results_bitrates = results_bitrates.append(results, ignore_index=True)
            csvfile = os.path.join(args.resultsdir, filename+'.csv')
            results_bitrates.to_csv(csvfile, index=False)

        return results_bitrates

    def test_seqs(self, filedir_list):
        # run
        results_list = []
        for idx_file, filedir in enumerate(tqdm.tqdm(filedir_list)):
            print('DBG!!! LosslessTester--test_seqs--filedir', idx_file, filedir)
            results = self.test_bitrates(filedir=filedir)
            results_list.append(results)
        
        # average
        # csvdirs = sorted(glob.glob(os.path.join(args.resultsdir, '*.csv')))
        # for i, f in enumerate(csvdirs): print('\ncsvdirs:', i, f)
        # results_list = [pd.read_csv(f) for f in csvdirs]
        mean_results = mean_dataframe(results_list)
        mean_results.to_csv(args.resultsdir+'.csv', index=False)
        print('DBG!!! LosslessTester test_seqs avg:\n', mean_results)

        return mean_results


if __name__ == '__main__':

    args.outdir = os.path.join(args.outdir, args.prefix)
    args.resultsdir = os.path.join(args.resultsdir, args.prefix)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultsdir, exist_ok=True)
    print('DBG!!! args.outdir\t', args.outdir)
    print('DBG!!! args.resultsdir\t', args.resultsdir)

    ################# testdata #################
    
    filedir_list = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]

    if len(filedir_list)>args.testdata_num: 
        if args.testdata_seqs=='random': 
            filedir_list = filedir_list[::len(filedir_list)//args.testdata_num]
        if args.testdata_seqs=='frame': 
            filedir_list = filedir_list[args.start_index:]
            filedir_list = filedir_list[:args.testdata_num]
            filedir_list = filedir_list[::args.interval]
    if args.filedir!='' and os.path.exists(args.filedir):
        filedir_list = [args.filedir]
    for i, f in enumerate(filedir_list): print('filedir_list', i, f)

    ################# test #################
    tester = LosslessTester()
    tester.test_seqs(filedir_list=filedir_list)

