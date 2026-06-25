# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

import os, sys, time
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
from lossless_attribute.model import list2set, list2set2

from cfg.get_args import get_args
args = get_args(component='attribute')


################################## ##################################
class LosslessCoder():
    def __init__(self, model):  
        self.model = model

    @torch.no_grad()
    def encode(self, out_set):
        strings_list, min_v_list, max_v_list = [], [], []
        for target, pred, in zip(out_set['target_list'], out_set['pred_list']):
            assert (pred.C==target.C).all()
            loc = pred.F[:, :pred.F.shape[-1]//2]
            scale = pred.F[:, pred.F.shape[-1]//2:]
            # encode
            strings, min_v, max_v = self.model.conditional_entropy_model.compress(target.F, loc, scale)
            strings_list.append(strings)
            min_v_list.append(min_v)
            max_v_list.append(max_v)

        return {'strings': strings_list, 'min_v': min_v_list, 'max_v': max_v_list}

    @torch.no_grad()
    def decode(self, out_set, enc_set):
        for target, pred, strings, min_v, max_v in zip(out_set['target_list'], out_set['pred_list'], enc_set['strings'], enc_set['min_v'], enc_set['max_v']):
            assert (pred.C==target.C).all()
            if pred.shape[-1]==target.shape[-1]*2:
                loc = pred.F[:, :pred.F.shape[-1]//2]
                scale = pred.F[:, pred.F.shape[-1]//2:]
                # encode & decode
                # strings, min_v, max_v = conditional_entropy_model.compress(target.F, loc, scale)
                target_dec = self.model.conditional_entropy_model.decompress(strings, loc, scale, min_v, max_v)
                error = (target.F.cpu()-target_dec.cpu()).abs().max().item()
                if error>0: print('DBG!!! mismatch', error)
        
        return
    
    def test(self, x, x_refT=None):
        """test real encoding & decoding
        """
        if not self.model.inter_mode:
            x_refT = None

        start_run = time.time()
        out = x
        out_refT = x_refT
        out_set_list2 = []
        cls_set_list = []
        for _ in range(args.scale):
            # print('DBG!!!', idx_scale, out.shape[0])
            for model in self.model.model_list:
                if not self.model.inter_mode or out_refT is None:
                    out, out_set_list, cls_set = model(out, out_refT)
                else:
                    out, out_refT, out_set_list, cls_set = model(out, out_refT)
                out_set_list2.append(out_set_list)
                cls_set_list.append(cls_set)

            if len(out)<32: break
        out_set, shape = list2set2(out_set_list2)
        runtime = round(time.time()-start_run, 3)
        #
        # encode
        start_enc = time.time()
        enc_set = self.encode(out_set)
        enctime = round(time.time()-start_enc, 3)
        # decode
        start_dec = time.time()
        self.decode(out_set, enc_set)
        dectime = round(time.time()-start_dec, 3)
        # test
        out_set = self.model.collect_loss(out_set)
        cls_set = list2set(cls_set_list)
        out_set.update(cls_set)

        bitstream_length = [len(strings) for strings in enc_set['strings']]
        bitstream_length.append(out.F.shape[0]*out.F.shape[1])

        out_set.update({'runtime':runtime, 'enctime':enctime, 'dectime':dectime, 'bitstream_length':bitstream_length})

        return out_set
