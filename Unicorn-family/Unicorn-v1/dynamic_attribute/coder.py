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

from cfg.get_args import get_args 
args = get_args(component='attribute')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from lossy_attribute.coder import LossyAttributeCoder
class DynamicLossyAttributeCoder(LossyAttributeCoder):
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def test(self, filedir, filedir_rec, ref_dir, lmb=0, real_coding=True):

        results = {}
        filename = os.path.split(filedir)[-1].split('.')[0]
        results['filename'] = filename
        results['filedir'] = filedir
        results['filedir_rec'] = filedir_rec
        results['filedir_ref'] = ref_dir
        results['lmb'] = lmb
        results['color_format'] = args.color_format
        
        # load data
        x = load_sparse_tensor(filedir, device=device, color_format=args.color_format, normalize=bool(args.normalize))
        ref = load_sparse_tensor(ref_dir, device=device, color_format=args.color_format, normalize=bool(args.normalize))

        # forward
        if real_coding:
            x_rec, results_test = self.model.test(x, ref=ref, lmb=lmb)
            results.update(results_test)
        else:
            start = time.time()
            out_set = self.model(x, ref=ref, training=False, lmb=lmb)
            x_rec = out_set['out_list'][0]
            dectime = round(sum(out_set['dectime']),3)
            enctime = round(time.time() - start,3)
            results['enctime'] = enctime
            results['dectime'] = dectime
            # test bpp
            results = self.get_bpp(out_set, results, num_points=len(x))
            if False:
                x_rec, results_test = self.model.test(x, ref=ref, lmb=lmb)
                x_rec=out_set['out_list'][0]
                assert (x_rec_test.C==x_rec.C).all()

        memory =  round(torch.cuda.max_memory_allocated()/1024**3,3)
        results['memory'] = memory
        torch.cuda.empty_cache()
        self.write_file(x_rec=x_rec, filedir_rec=filedir_rec)

        return results

