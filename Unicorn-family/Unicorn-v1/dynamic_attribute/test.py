# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-09

import os, sys, time, glob
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from data_utils.dataloaders.attribute_dataloader import load_sparse_tensor
from data_utils.pandas_utils import mean_dataframe

from lossy_attribute.model import MultiscaleVAE as IntraModel
from dynamic_attribute.model import InterModel as InterModel

from dynamic_attribute.coder import DynamicLossyAttributeCoder

from cfg.get_args import get_args 
args = get_args(component='attribute')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


from lossy_attribute.test import Tester as StaticTester
class Tester(StaticTester):
    def __init__(self):
        #########  load_model
        if args.ckptdir_intra != '' and args.ckptdir_inter != '': 
            if args.Vmode==0:
                self.intra_model = self.load_intra_model(args.ckptdir_intra)
                self.inter_model = self.load_inter_model(args.ckptdir_inter)
            if args.Vmode==1:
                    self.intra_model_list = [self.load_intra_model(args.ckptdir_intra)]
                    self.inter_model_list = [self.load_inter_model(args.ckptdir_inter)]
                    step = len(WEIGHT_LIST)//(args.num_bitrates-1)
                    lmb_list = WEIGHT_LIST[::step][:(args.num_bitrates-1)][:]
                    lmb_list = np.append(lmb_list, WEIGHT_LIST[-1])
                    self.lmb_set = [lmb_list]
                    print('DBG!!! self.lmb_set', self.lmb_set)
        else:
            self.intra_model_list, self.inter_model_list, self.lmb_set = self.get_piecewise_variable_bitrates()

        return

    def load_intra_model(self, ckptdir_intra, stage=1):
        print('DBG!!! ckptdir_intra\t', ckptdir_intra)
        intra_model = IntraModel(ref_channels=0, stage=stage).to(device)
        print('DBG!!!ckptdir_intra', ckptdir_intra)
        assert os.path.exists(ckptdir_intra)#
        ckpt = torch.load(ckptdir_intra)
        model_dict = intra_model.state_dict()
        pretrained_dict = {k:v for k,v in ckpt['model'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        intra_model.load_state_dict(model_dict)

        return intra_model

    def load_inter_model(self, ckptdir_inter, stage=1):
        print('DBG!!! ckptdir_inter\t', ckptdir_inter)
        inter_model = InterModel(stage=stage).to(device)
        assert os.path.exists(ckptdir_inter)
        ckpt = torch.load(ckptdir_inter)
        model_dict = inter_model.state_dict()
        pretrained_dict = {k:v for k,v in ckpt['model'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        inter_model.load_state_dict(model_dict)

        return inter_model

    def get_piecewise_variable_bitrates(self):
        intra_rootdir = '../ckpts/lossy_attribute/human/variable/'
        ckptdir_intra_list = [intra_rootdir+'32k8k/epoch_last.pth',
                            intra_rootdir+'8k256/epoch_last.pth',
                            intra_rootdir+'2k128/epoch_last.pth']
        intra_model_list = []
        for idx, ckptdir in enumerate(ckptdir_intra_list):
            intra_model = self.load_intra_model(ckptdir)
            intra_model_list.append(intra_model)        

        inter_rootdir = '../ckpts/dynamic_attribute/human/variable/'
        ckptdir_inter_list = [inter_rootdir+'32k8k/epoch_last.pth',
                            inter_rootdir+'8k256/epoch_last.pth',
                            inter_rootdir+'2k128/epoch_last.pth']
        inter_model_list = []
        for idx, ckptdir in enumerate(ckptdir_inter_list):
            inter_model = self.load_inter_model(ckptdir)
            inter_model_list.append(inter_model)

        lmb_set = [[2**15, 2**14, 2**13], 
                    [2**13, 2**12, 2**11], 
                    [2**10, 2**9, 2**8, 2**7]]

        return intra_model_list, inter_model_list, dedupe_lmb_set(lmb_set)

    def test_bitrates(self, filedir_list):
        if args.Vmode==0: 
            results = self.test_seqs(filedir_list, self.intra_model, self.inter_model, rate=1, lmb=args.lmb)
            results.to_csv(os.path.join(args.resultsdir + '.csv'), index=False)
        else:
            idx_rate = 1
            for intra_model, inter_model, lmb_list in zip(self.intra_model_list, self.inter_model_list, self.lmb_set):
                for _, lmb in enumerate(lmb_list):
                    print('DBG!!! variable-rate\t', idx_rate, lmb)
                    results_one = self.test_seqs(filedir_list, intra_model, inter_model, rate=idx_rate, lmb=lmb)
                    print("results:\t", idx_rate, results_one)
                    # merge all file
                    if idx_rate==1: results = results_one.copy(deep=True)
                    else: results = pd.concat([results, results_one], ignore_index=True)
                    results.to_csv(os.path.join(args.resultsdir + '.csv'), index=False)
                    #
                    idx_rate = idx_rate + 1
        
        return results

    def test_seqs(self, filedir_list, intra_model, inter_model, rate=0, lmb=0):
        results_list = []
        results_allfile = {}
        results_allfile = pd.DataFrame([results_allfile])
        for idx, filedir in enumerate(tqdm(filedir_list)):
            filename = os.path.split(filedir)[-1].split('.')[0]
            if idx == 0 or args.inter_mode == 0:
                results_one = self.test_one(intra_model, filedir, rate=rate, lmb=lmb)
                rec_dir = results_one['filedir_rec']
                results_one = pd.DataFrame([results_one])
                results_one.to_csv(os.path.join(args.resultsdir, 'R' + str(rate) + '_' + filename + '_intra.csv'),
                                   index=False)
                results_allfile = pd.concat([results_allfile, results_one], ignore_index=True)
                csvfile = os.path.join(args.resultsdir, 'all.csv')
                results_allfile.to_csv(csvfile, index=False)
                del intra_model
                torch.cuda.empty_cache()
            else:
                ref_dir = rec_dir
                print('ref_dir/filedir:', ref_dir, '\t--->\t', filedir)
                results_one = self.test_one_inter(inter_model, filedir, ref_dir, rate=rate, lmb=lmb)
                rec_dir = results_one['filedir_rec']
                results_one = pd.DataFrame([results_one])
                results_one.to_csv(os.path.join(args.resultsdir, 'R' + str(rate) + '_' + filename + '_inter.csv'),
                                   index=False)
                results_allfile = pd.concat([results_allfile, results_one], ignore_index=True)
                csvfile = os.path.join(args.resultsdir, 'R' + str(rate) + '.csv')
                results_allfile.to_csv(csvfile, index=False)
                torch.cuda.empty_cache()
            print('results_one', results_one)
            results_list.append(results_one)
            torch.cuda.empty_cache()

        # average
        mean_results = mean_dataframe(results_list)
        mean_results.to_csv(args.resultsdir + '.csv', index=False)
        print('DBG!!!avg:\n', mean_results)

        return mean_results
    
    @torch.no_grad()
    def test_one_inter(self, model, filedir, ref_dir, rate=0, lmb=0):
        results = {}
        print('DBG!!! test_one_inter:\t', 'filedir', filedir, '\tref_dir', ref_dir)
        # forward
        filename = os.path.split(filedir)[-1].split('.')[0]
        filedir_rec = os.path.join(args.outdir, filename+'_R'+str(rate)+'.ply')
        inter_coder = DynamicLossyAttributeCoder(model=model)
        results = inter_coder.test(filedir=filedir, filedir_rec=filedir_rec, 
                                   ref_dir=ref_dir, lmb=lmb)
        
        psnr_results = self.get_PSNR(filedir, filedir_rec)
        for k, v in psnr_results.items(): results[k] = v

        return results



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.outdir = os.path.join(args.outdir, args.prefix)
    os.makedirs(args.outdir, exist_ok=True)    
    args.resultsdir = os.path.join(args.resultsdir, args.prefix)
    os.makedirs(args.resultsdir, exist_ok=True)
    print('DBG!!!resultsdir\t', args.resultsdir)
    
    WEIGHT_LIST = np.logspace(np.log2(args.weight_distortion), np.log2(args.weight_distortion_min), 1000000, base=2).astype('int')
    print('DBG!!! WEIGHT_LIST\t', len(WEIGHT_LIST), WEIGHT_LIST.min(), WEIGHT_LIST.max(), WEIGHT_LIST.mean())
    
    ################# testdata #################
    
    filedir_list = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.ply'), recursive=True))
    if len(filedir_list)>args.testdata_num:
        # if args.testdata_seqs=='random':
        #     filedir_list = filedir_list[::len(filedir_list)//args.testdata_num]
        # if args.testdata_seqs=='frame':
        #     filedir_list = filedir_list[:args.testdata_num]
        filedir_list = filedir_list[:args.testdata_num]

    for i, f in enumerate(filedir_list):
        print('filedir', i, f)

    ################# test #################
    tester = Tester()
    tester.test_bitrates(filedir_list=filedir_list)











