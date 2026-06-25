# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

import  os, sys, glob
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import MinkowskiEngine as ME
from data_utils.dataloaders.attribute_dataloader import load_sparse_tensor
from basic_models.loss import get_bits
from lossless_attribute.model import Model
from coder_lossless import LosslessCoder

from cfg.get_args import get_args 
args = get_args(component='attribute')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##########################################################################################
class Tester():
    def __init__(self):
        # intra model
        model = Model(inter_mode=0).to(device)
        print('DBG!!! args.ckptdir\t', args.ckptdir)
        assert args.ckptdir!=''
        ckpt = torch.load(args.ckptdir)
        model.load_state_dict(ckpt['model'])

        self.coder = LosslessCoder(model)
        
        if args.inter_mode:
            model_inter = Model(inter_mode=1).to(device)
            print('DBG!!! args.ckptdir_inter\t', args.ckptdir_inter)
            assert args.ckptdir_inter!=''
            ckpt_inter = torch.load(args.ckptdir_inter)
            model_inter.load_state_dict(ckpt_inter['model'])
            self.coder_inter = LosslessCoder(model_inter)
    
    @torch.no_grad()
    def test_seqs(self, filedir_list):
        results_list = []
        if not args.inter_mode:
            for idx_file, filedir in enumerate(tqdm(filedir_list)):
                results = self.test_one(filedir)
                print('DBG!!!', idx_file, filedir, results)
                results = pd.DataFrame([results])
                filename = os.path.split(filedir)[-1].split('.')[0]
                csvfile = os.path.join(args.resultsdir, filename +'.csv')
                results.to_csv(csvfile, index=False)
                results_list.append(results)
        else:
            results_list = []
            for idx_file, filedir in enumerate(tqdm(filedir_list)):
                if idx_file==0:
                    results = self.test_one(filedir)
                    print(idx_file, filedir, results)
                else:
                    ref_dir = filedir_list[idx_file-1]
                    print('inter:\t', idx_file, filedir, ref_dir)
                    results = self.test_one(filedir, ref_dir=ref_dir)
                    print(idx_file, filedir, results)
                results = pd.DataFrame([results])
                filename = os.path.split(filedir)[-1].split('.')[0]
                csvfile = os.path.join(args.resultsdir, filename +'.csv')
                results.to_csv(csvfile, index=False)
                results_list.append(results)
                if idx_file==0: all_results = results.copy(deep=True)
                else: all_results = all_results.append(results, ignore_index=True)
                csvfile = os.path.join(args.resultsdir+'_frame'+str(len(filedir_list))+'.csv')
                all_results.to_csv(csvfile, index=False)

        # average
        from data_utils.pandas_utils import mean_dataframe
        mean_results = mean_dataframe(results_list)
        mean_results.to_csv(args.resultsdir+'.csv', index=False)
        print('DBG!!!avg:\n', mean_results)

        return mean_results
    
    @torch.no_grad()
    def test_one(self, filedir, ref_dir=None):
        # forward  
        x = load_sparse_tensor(filedir, color_format=args.color_format, normalize=False)      
        if ref_dir is None:
            out_set = self.coder.test(x)
        else:
            ref = load_sparse_tensor(ref_dir, color_format=args.color_format, normalize=False)
            out_set = self.coder_inter.test(x, x_refT=ref)

        # loss
        if 'likelihood_list' in out_set:
            ce_bpp_list = []
            for likelihood in out_set['likelihood_list']:
                curr_bpp = get_bits(likelihood)/float(x.__len__())
                ce_bpp_list.append(curr_bpp.item())
                # print('curr_bpp', curr_bpp)
        if 'prob_list' in out_set:
            cls_bpp_list = []
            for prob in out_set['prob_list']:
                curr_bpp = get_bits(prob)/float(x.__len__())
                cls_bpp_list.append(curr_bpp.item())
        # test
        if 'init_bits_list' in out_set:
            init_bpp_list = []
            for init_bits in out_set['init_bits_list']:
                curr_bpp = init_bits/float(x.__len__())
                init_bpp_list.append(curr_bpp)
        if 'init_prob_list' in out_set:
            init_cls_bpp_list = []
            for init_prob in out_set['init_prob_list']:
                curr_bpp = get_bits(init_prob)/float(x.__len__())
                init_cls_bpp_list.append(curr_bpp.item())
        if 'res_bits_list' in out_set:
            res_bpp_list = []
            for res_bits in out_set['res_bits_list']:
                curr_bpp = res_bits/float(x.__len__())
                res_bpp_list.append(curr_bpp)
        ce_bpp = sum(ce_bpp_list)
        init_bpp = sum(init_bpp_list)
        res_bpp = sum(res_bpp_list)
        if 'prob_list' in out_set:
            cls_bpp = sum(cls_bpp_list)
            init_cls_bpp = sum(init_cls_bpp_list)
        else:
            cls_bpp = 0
            init_cls_bpp = 0      

        bits = sum(out_set['bitstream_length'])*8
        real_bpp = round(bits/len(x), 3)

        results = {'filedir':filedir, 'bpp':ce_bpp+cls_bpp, 'ce_bpp':ce_bpp, 'cls_bpp':cls_bpp, 
                'init_bpp':init_bpp, 'init_cls_bpp':init_cls_bpp, 'res_bpp':res_bpp, 
                'runtime':out_set['runtime'], 'enctime':out_set['enctime'], 'dectime':out_set['dectime'],
                'bits':bits, 'length':len(x), 'real_ce_bpp':real_bpp, 'real_bpp':real_bpp+cls_bpp}
        for k, v in results.items():
            if isinstance(v, float):
                results[k] = round(v, 3)
        
        return results


if __name__ == '__main__':

    args.resultsdir = os.path.join(args.resultsdir, args.prefix)
    os.makedirs(args.resultsdir, exist_ok=True)
    
    filedir_list = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]

    if len(filedir_list)>args.testdata_num: 
        if args.testdata_seqs=='random':
            filedir_list = filedir_list[::len(filedir_list)//args.testdata_num]
        if args.testdata_seqs=='frame' or args.inter_mode:
            filedir_list = filedir_list[:args.testdata_num]

    for i, f in enumerate(filedir_list):
        print('filedir', i, f)

    ################# test #################
    tester = Tester()
    tester.test_seqs(filedir_list=filedir_list)
