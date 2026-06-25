# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-06

import os, sys, time, argparse, glob
from tqdm import tqdm
import torch
import MinkowskiEngine as ME
import numpy as np
rootdir = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(rootdir)
from third_party.pc_error_attr import pc_error
import pandas as pd

from lossy_attribute.coder import LossyAttributeCoder
from lossy_attribute.model import MultiscaleVAE as Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from cfg.get_args import get_args
args = get_args(component='attribute')


def dedupe_lmb_set(lmb_set):
    seen = set()
    deduped = []
    for lmb_list in lmb_set:
        curr = []
        for lmb in lmb_list:
            key = int(lmb)
            if key in seen:
                continue
            curr.append(lmb)
            seen.add(key)
        deduped.append(curr)
    return deduped


def get_piecewise_variable_bitrates(bitrate_config=''):
    """
    """
    if bitrate_config=='object':
        ckptdir_rootdir = './ckpts/lossy_attribute/rwtt/'
        ckptdir_list = [ckptdir_rootdir+'32k8k/epoch_last.pth',
                        ckptdir_rootdir+'8k256/epoch_last.pth',
                        ckptdir_rootdir+'2k128/epoch_last.pth']
        lmb_set = [[2**15, 2**14, 2**13],
                [2**13, 2**12, 2**11],
                [2**10, 2**9, 2**8, 2**7]]# TODO 2*13 or not

    if bitrate_config=='human':
        # ckptdir_rootdir = 'ckpts/'
        ckptdir_rootdir = '../ckpts/lossy_attribute/human/variable/'
        ckptdir_list = [ckptdir_rootdir+'32k8k/epoch_last.pth',
                        ckptdir_rootdir+'8k256/epoch_last.pth',
                        ckptdir_rootdir+'2k128/epoch_last.pth']
        lmb_set = [[2**15, 2**14, 2**13],
                [2**13, 2**12, 2**11],
                [2**10, 2**9, 2**8, 2**7]]

    if bitrate_config=='8ivfb':
        # ckptdir_rootdir = 'ckpts/'
        ckptdir_rootdir = 'ckpts/8ivfb/'
        ckptdir_list = [ckptdir_rootdir+'32k8k/epoch_last.pth',
                        ckptdir_rootdir+'8k256/epoch_last.pth',
                        ckptdir_rootdir+'2k128/epoch_last.pth']
        lmb_set = [[2**15, 2**14, 2**13],
                [2**13, 2**12, 2**11],
                [2**10, 2**9, 2**8, 2**7]]

    elif bitrate_config=='scan2cm':
        ckptdir_rootdir = './ckpts/lossy_attribute/scan2cm/variable/'
        ckptdir_list = [ckptdir_rootdir+'32k8k/epoch_last.pth',
                        ckptdir_rootdir+'8k256/epoch_last.pth',
                        ckptdir_rootdir+'2k256/epoch_last.pth']
        lmb_set = [[2**15, 2**14, 2**13],
                    [2**13, 2**12, 2**11, 2**10],
                    [2**10, 2**9, 2**8]]

    elif bitrate_config=='scan_vox9':
        ckptdir_rootdir = '../ckpts/lossy_attribute/scan_vox9/'
        # ckptdir_list = [ckptdir_rootdir+'32k8k/epoch_last.pth']
        ckptdir_list = [ckptdir_rootdir+'180k16k/epoch_last.pth',
                        ckptdir_rootdir+'32k8k/epoch_last.pth',
                        ckptdir_rootdir+'10k256/epoch_last.pth',
                        ckptdir_rootdir+'2k128/epoch_last.pth']
        lmb_set = [[2**17, 2**16, 2**15],
                    [2**15, 2**14, 2**13],
                    [2**13, 2**12, 2**11, 2**10],
                    [2**9, 2**8]]

    elif bitrate_config=='ford1mm':
        ckptdir_rootdir = '../ckpts/lossy_attribute/ford/variable/'
        ckptdir_list = [ckptdir_rootdir+'512-128/epoch_last.pth',
                        ckptdir_rootdir+'128-32/epoch_last.pth',
                        ckptdir_rootdir+'32-8/epoch_last.pth',
                        ckptdir_rootdir+'1stage/32-8/epoch_last.pth',]
        lmb_set = [[2**9, 2**8, 2**7],
                    [2**7, 2**6, 2**5],
                    [2**5, 2**4],
                    [2**5, 2**4]]

    elif bitrate_config=='ford1mm-1stage':
        ckptdir_rootdir = '../ckpts/lossy_attribute/ford/variable/1stage/'
        ckptdir_list = [ckptdir_rootdir+'512-64/epoch_last.pth',
                        ckptdir_rootdir+'256-16/epoch_last.pth',
                        ckptdir_rootdir+'64-8/epoch_last.pth']
        lmb_set = [[2**9, 2**8, 2**7],
                    [2**7, 2**6, 2**5],
                    [2**5, 2**4, 2**3]]

    elif bitrate_config=='kitti1mm':
        ckptdir_rootdir = '../ckpts/lossy_attribute/kitti/variable/'
        ckptdir_list = [ckptdir_rootdir+'512-128/epoch_last.pth',
                        ckptdir_rootdir+'128-32/epoch_last.pth',
                        ckptdir_rootdir+'32-8/epoch_last.pth',
                        ckptdir_rootdir+'32-8-1stage/epoch_last.pth',]
        lmb_set = [[2**9, 2**8, 2**7],
                    [2**7, 2**6, 2**5],
                    [2**5, 2**4],
                    [2**5, 2**4]]

    return ckptdir_list, dedupe_lmb_set(lmb_set)


############################################################################################

class Tester():
    def __init__(self):
        #  load model from args.ckptdir_list and set coder
        print('DBG!!! args.ckptdir_list\t', args.ckptdir_list)

        ########## 1. set ckptdir
        if args.ckptdir_list in ['', None] and args.init_ckpt != '':
            args.ckptdir_list = [args.init_ckpt]
        if args.Vmode==1 and args.piecewise_variable_bitrates != '':
            args.ckptdir_list, self.lmb_set = get_piecewise_variable_bitrates(
                bitrate_config=args.piecewise_variable_bitrates)
        print('DBG!!!args.ckptdir_list', args.ckptdir_list)

        # load model (coder) from ckptdir
        self.model_list = []
        for idx, ckptdir in enumerate(args.ckptdir_list):
            print("model", idx, 'ckptdir', ckptdir)
            model = Model().to(device)
            if args.piecewise_variable_bitrates in ['ford1mm', 'kitti1mm'] and idx>=3:
                model = Model(stage=1).to(device)
            assert os.path.exists(ckptdir)
            ckpt = torch.load(ckptdir)
            model_dict = model.state_dict()
            pretrained_dict = {k:v for k,v in ckpt['model'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            self.model_list.append(model)

        ########## set lmb_list
        if args.Vmode==1 and len(self.model_list)==1:
            WEIGHT_LIST = np.logspace(np.log2(args.weight_distortion), np.log2(args.weight_distortion_min), 1000000, base=2).astype('int')
            step = len(WEIGHT_LIST)//(args.num_bitrates-1)
            lmb_list = WEIGHT_LIST[::step][:(args.num_bitrates-1)][:]
            lmb_list = np.append(lmb_list, WEIGHT_LIST[-1])
            self.lmb_set = [lmb_list]
            print('DBG!!!!self.lmb_set', self.lmb_set)


    def test_seqs(self, filedir_list, intra_model, rate=0, lmb=0):
        results_list = []
        results_allfile = {}
        results_allfile = pd.DataFrame([results_allfile])
        for idx, filedir in enumerate(tqdm(filedir_list)):
            filename = os.path.split(filedir)[-1].split('.')[0]
            results_one = self.test_one(intra_model, filedir, rate=rate, lmb=lmb)
            results_one = pd.DataFrame([results_one])
            results_one.to_csv(os.path.join(args.resultsdir, 'R' + str(rate) + '_' + filename + '_intra.csv'),
                               index=False)
            results_allfile = pd.concat([results_allfile, results_one], ignore_index=True)
            csvfile = os.path.join(args.resultsdir, 'R' + str(rate) + '_' + 'all.csv')
            results_allfile.to_csv(csvfile, index=False)
            torch.cuda.empty_cache()
            print('results_one', results_one)
            results_list.append(results_one)
            torch.cuda.empty_cache()
        # average

        return results_list

    def test_bitrates(self, filedir_list):
        if args.Vmode==0:
            for idx_model, model in enumerate(self.model_list):
                results_one = self.test_seqs(filedir_list, model, rate=idx_model)
                print("results:\t", idx_model, results_one)
                results_one = pd.DataFrame([results_one])
                # merge all file
                if idx_model==0: results = results_one.copy(deep=True)
                else: results = results.append(results_one, ignore_index=True)

        if args.Vmode==1:
            idx_rate = 1

            for model, lmb_list in zip(self.model_list, self.lmb_set):
                results_allfile = {}
                results_allfile = pd.DataFrame([results_allfile])
                for _, lmb in enumerate(lmb_list):
                    print('DBG!!! variable-rate\t', idx_rate, lmb)
                    results_one = self.test_seqs(filedir_list, model, rate=idx_rate, lmb=lmb)
                    print("results:\t", idx_rate, results_one)
                    results_one = pd.DataFrame([results_one])
                    # merge all file
                    if idx_rate==1: results = results_one.copy(deep=True)
                    else: results = pd.concat([results,results_one], ignore_index=True)
                    #
                    idx_rate = idx_rate + 1

        return results

    @torch.no_grad()
    def test_one(self, model, filedir, rate=0, lmb=0):

        filename = os.path.split(filedir)[-1].split('.')[0]
        filedir_rec = os.path.join(args.outdir, filename+'_R'+str(rate)+'.ply')

        coder = LossyAttributeCoder(model=model)
        results = coder.test(filedir=filedir, filedir_rec=filedir_rec, lmb=lmb)

        psnr_results = self.get_PSNR(filedir, filedir_rec)
        for k, v in psnr_results.items(): results[k] = v

        return results

    def get_PSNR(self, filedir, filedir_rec):
        if args.in_channels==3:
            results = pc_error(filedir, filedir_rec, res=1)
            results['Y-PSNR'] = round(results['  c[0],PSNRF'], 3)
            results['YUV-PSNR'] = (results['  c[0],PSNRF']*6+results['  c[1],PSNRF']*1+results['  c[2],PSNRF']*1)/8
            results['YUV-PSNR'] = round(results['YUV-PSNR'], 3)
        if args.in_channels==1:
            results = pc_error(filedir, filedir_rec, res=1, lidar=1, show=False)
            results['PSNR'] = results['   r,PSNR   F']

        return results



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.outdir = os.path.join(args.outdir, args.prefix)
    args.resultsdir = os.path.join(args.resultsdir, args.prefix)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultsdir, exist_ok=True)
    print('DBG!!!resultsdir\t', args.resultsdir)

    filedir_list = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.ply'), recursive=True))
    if len(filedir_list)>100:
        if args.testdata_seqs=='random':
            filedir_list = filedir_list[::len(filedir_list)//100]
        if args.testdata_seqs=='frame':
            filedir_list = filedir_list[:100]

    for i, f in enumerate(filedir_list):
        print('filedir', i, f)

    ################# test #################
    tester = Tester()
    tester.test_bitrates(filedir_list=filedir_list)

