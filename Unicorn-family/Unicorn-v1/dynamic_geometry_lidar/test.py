# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-9-21

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import glob, tqdm
import torch
import pandas as pd
import argparse
from coder import InterCoder2, LossyCoder
from model import PCCModel as Model
from data_utils.pandas_utils import mean_dataframe

from lossy_geometry.model_offset import OffsetModel

from cfg.get_args import get_args 
args = get_args(component='geometry')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(ckptdir_low, ckptdir_high, ckptdir_offset, filedir_list, offset=False, octree=False):

    model_low = Model(inter_mode=args.inter_mode, scale=args.scale, block_type='conv').to(device)
    assert os.path.exists(ckptdir_low)
    ckpt = torch.load(ckptdir_low)
    model_low.load_state_dict(ckpt['model'])


    model_high = Model(inter_mode=args.inter_mode, scale=args.scale, block_type='tf').to(device)
    assert os.path.exists(ckptdir_high)
    ckpt = torch.load(ckptdir_high)
    model_high.load_state_dict(ckpt['model'])


    basic_coder = InterCoder2(model_low, model_high, device=device)
    if offset:
        model_offset = OffsetModel(kernel_size=5).to(device)
        assert os.path.exists(ckptdir_offset)
        ckpt = torch.load(ckptdir_offset)
        model_offset.load_state_dict(ckpt['model'])
    else:
        model_offset = None

    lossy_coder = LossyCoder(basic_coder, model_offset=model_offset, device=device)

    for idx_file, filedir in enumerate(tqdm.tqdm(filedir_list)):

        if idx_file<args.start_index: continue
        if args.interval > 1 and (idx_file-1) %args.interval != 0: continue

        filedir_ref = filedir_list[idx_file-1]
        print('inter coding:\t', filedir_ref, '-->', filedir)

        filename = os.path.split(filedir)[-1].split('.')[0]
        # mode1: input Ford_q1mm/or KITTI raw, adjust posQuantScale
        posQuantscale_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        results_list = []
        for idx_rate, posQuantscale in enumerate(posQuantscale_list):
            bin_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.bin')
            dec_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.ply')
            BUG = 0
            try: 
                results = lossy_coder.test(filedir, filedir_ref, bin_dir, dec_dir, posQuantscale=posQuantscale, 
                                        quant_mode='precision', quant_factor=1, psnr_mode='gpcc', test_d2=True)
            except RuntimeError:
                BUG = 1
                break            
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

    # average
    csvdirs = sorted(glob.glob(os.path.join(args.resultsdir, '*.csv')))
    for i, f in enumerate(csvdirs): print('csvdirs:\t', i, f)
    results_list = [pd.read_csv(f) for f in csvdirs]
    mean_results = mean_dataframe(results_list)
    mean_results.to_csv(args.resultsdir+'.csv', index=False)
    print('DBG!!!avg:\n', mean_results)


    return


if __name__ == '__main__':

    args.outdir = os.path.join(args.outdir, args.prefix)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultsdir, exist_ok=True)
    
    ################# test dataset ################# 

    filedir_list = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    if len(filedir_list)>args.testdata_num: filedir_list = filedir_list[:args.testdata_num]
    print('filedir_list length:\t', len(filedir_list))
    for i, f in enumerate(filedir_list):
        print('filedir_list', i, f)

    ################# test #################
    args.resultsdir = os.path.join(args.resultsdir, 
        'inter_'+'_'+args.prefix+'_frame'+str(len(filedir_list)))
    os.system('rm -r '+args.resultsdir)
    os.makedirs(args.resultsdir, exist_ok=True)


    test(ckptdir_low=args.ckptdir_low, 
            ckptdir_high=args.ckptdir_high, 
            ckptdir_offset =args.ckptdir_offset,
            filedir_list=filedir_list, 
            offset=args.offset)