# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-04

import os, sys, time
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
from basic_models.loss import get_bits
from basic_models.factorized_entropy_model import get_entropy
from data_utils.dataloaders.attribute_dataloader import load_sparse_tensor
from data_utils.attribute.inout import write_ply_ascii
from data_utils.attribute.color_format import yuv2rgb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from cfg.get_args import get_args
args = get_args(component='attribute')


################################## ##################################
class LossyAttributeCoder():
    def __init__(self, model):
        self.model = model

    # @torch.no_grad()
    # def encode(self, filedir, lmb=0):
    #     """TODO
    #     """

    #     return

    # @torch.no_grad()
    # def decode(self, bitstream):
    #     """TODO
    #     """

    #     return

    @torch.no_grad()
    def test(self, filedir, filedir_rec, lmb=0, real_coding=False):

        results = {}
        filename = os.path.split(filedir)[-1].split('.')[0]
        results['filename'] = filename
        results['filedir'] = filedir
        results['filedir_rec'] = filedir_rec
        results['lmb'] = lmb
        results['color_format'] = args.color_format

        # load data
        x = load_sparse_tensor(filedir, device=device, color_format=args.color_format, normalize=bool(args.normalize))

        # forward
        if real_coding:
            x_rec, results_test = self.model.test(x, lmb=lmb)
            results.update(results_test)
        else:
            start = time.time()
            out_set = self.model(x, training=False, lmb=lmb)
            x_rec = out_set['out_list'][0]
            dectime = round(sum(out_set['dectime']),3)
            enctime = round(time.time() - start,3)
            results['enctime'] = enctime
            results['dectime'] = dectime
            # test bpp
            results = self.get_bpp(out_set, results, num_points=len(x))
            if False:
                x_rec_test, results_test = self.model.test(x, lmb=lmb)
                x_rec=out_set['out_list'][0]
                assert (x_rec_test.C==x_rec.C).all()

        memory =  round(torch.cuda.max_memory_allocated()/1024**3,3)
        results['memory'] = memory
        torch.cuda.empty_cache()
        self.write_file(x_rec=x_rec, filedir_rec=filedir_rec)

        return results

    def write_file(self, x_rec, filedir_rec):
        #
        coords = x_rec.C[:,1:].detach().cpu().numpy()
        feats = x_rec.F[:].detach().cpu()

        if args.color_format=='yuv':
            feats = torch.clamp(feats, 0, 1)
            feats = yuv2rgb(feats, out_range=255.)
        else:
            if bool(args.normalize): feats *= 100.
        if args.color_format == 'reflectance':
            feats = feats.round().int().numpy()
            feats = np.clip(feats, a_min=0, a_max=100)
        else:
            feats = feats.round().int().numpy()
            feats = np.clip(feats, a_min=0, a_max=255)

        write_ply_ascii(filedir=filedir_rec, coords=coords, feats=feats)

        return filedir_rec

    def get_bpp(self, out_set, results, num_points):
        if 'likelihood_list' in out_set:
            bpp_list = []
            for idx, likelihood in enumerate(out_set['likelihood_list']):
                bits = round(get_bits(likelihood).item())
                bpp_global = round(bits/float(num_points),3)
                bpp_local = round(bits/float(len(likelihood)),3)
                if args.DBG:
                    print('DBG!!! check likelihood idx (len(x)/len(likelihood)/bpp_global/bpp_local/bits):\n',
                        idx, num_points, len(likelihood), bpp_global, bpp_local, bits)
                bpp_list.append(bpp_global)

        if 'real_bits_list' in out_set:
            real_bpp_list = []
            for idx, real_bits in enumerate(out_set['real_bits_list']):
                real_bpp = round(real_bits/float(num_points),3)
                real_bpp_list.append(real_bpp)

        # Qbpp
        if 'Qvalue_list' in out_set:
            Qbpp_list = []
            idx_tp = 0
            for idx, Qvalue in enumerate(out_set['Qvalue_list']):
                Qbits = round(get_entropy(Qvalue.float().round().int().detach().cpu().numpy()))
                Qbpp_local = round(Qbits/len(Qvalue), 3)
                Qbpp_global = round(Qbits/num_points, 3)
                if args.DBG:
                    print('DBG!!! check Qvalue idx num_points/len(Qvalue)/bpp_global/bpp_local/bits):\n',
                        idx, num_points, len(Qvalue), Qbpp_global, Qbpp_local, Qbits)
                Qbpp_list.append(Qbpp_global)

        results['num_points'] = num_points
        if 'gpcc_bpp' in out_set:
            results['gpcc_bpp'] = out_set['gpcc_bpp']
            if args.DBG: print('DBG','!'*100, 'gpcc_bpp', results['gpcc_bpp'])
            results['bpp'] = np.array(bpp_list).sum().round(3) + out_set['gpcc_bpp']
            results['Qbpp'] = np.array(Qbpp_list).sum().round(3) + out_set['gpcc_bpp']
            results['real_bpp'] = np.array(real_bpp_list).sum().round(3) + out_set['gpcc_bpp']
        else:
            results['bpp'] = np.array(bpp_list).sum().round(3)
            
            results['Qbpp'] = np.array(Qbpp_list).sum().round(3)
            results['real_bpp'] = np.array(real_bpp_list).sum().round(3)
        # details
        if False:
            for idx, tp_bpp in enumerate(np.array(bpp_list)):
                results['bpp_s'+str(idx+1)] = round(tp_bpp, 6)
            for idx, tp_Qbpp in enumerate(np.array(Qbpp_list)):
                results['Qbpp_s'+str(idx+1)] = round(tp_Qbpp, 6)

        results['gpcc_bpp'] = round(results['gpcc_bpp'], 3)
        results['bpp'] = round(results['bpp'], 3)
        results['Qbpp'] = round(results['Qbpp'], 3)

        print('test bpp:\t',  results['bpp'], '\t=\t',  np.array(bpp_list).round(3), '+', results['gpcc_bpp'])
        # for b in np.array(bpp_list).round(3): print(b, ',')
        print('test Qbpp:\t',  results['Qbpp'], '\t=\t', np.array(Qbpp_list).round(3), '+', results['gpcc_bpp'])
        print('test real bpp:\t',  results['real_bpp'], '\t=\t',  np.array(real_bpp_list)[::-1].round(3), '+', results['gpcc_bpp'])

        return results
