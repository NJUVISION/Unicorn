# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-9-18

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
from lossless_attribute.utils import rgb2YCoCg, YCoCg2rgb, kdtree_partition, read_ply_ascii, write_ply_ascii, list2set, concat_voxel
from third_party.pc_error_attr import pc_error
from cfg.get_args import get_args
args = get_args(component='attribute')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################################################################################


class Tester():
    def __init__(self):
        # intra model
        self.model = Model(inter_mode=0).to(device)
        print('DBG!!! args.ckptdir\t', args.ckptdir)
        assert args.ckptdir != ''
        ckpt = torch.load(args.ckptdir)
        self.model.load_state_dict(ckpt['model'])

        self.coder = LosslessCoder(self.model)

        if args.inter_mode:
            self.model_inter = Model(inter_mode=1).to(device)
            print('DBG!!! args.ckptdir_inter\t', args.ckptdir_inter)
            assert args.ckptdir_inter != ''
            ckpt_inter = torch.load(args.ckptdir_inter)
            self.model_inter.load_state_dict(ckpt_inter['model'])
            self.coder_inter = LosslessCoder(self.model_inter)

    @torch.no_grad()
    def test_partition(self, filedir_list, partition_num):
        results_list = []
        if not args.inter_mode:
            for idx_file, filedir in enumerate(tqdm(filedir_list)):
                coords, feats = read_ply_ascii(filedir)
                points = np.concatenate((coords, feats), axis=1)
                part_list = kdtree_partition(points, partition_num)
                part_dec = []
                out_set_list2 = []
                for idx_part, part_slice in enumerate(part_list):
                    filename = filedir.split('/')[-1].split('.')[0] + '_part' + '_' + str(idx_part)
                    coords_part = part_slice[:,:3]
                    feats_part = rgb2YCoCg(part_slice[:,3:])
                    coords_part, feats_part = ME.utils.sparse_collate([coords_part], [feats_part])
                    x = ME.SparseTensor(features=feats_part, coordinates=coords_part, tensor_stride=1, device=device)
                    results, dec_out = self.test_part_one(self.model, x, filename, filedir)
                    part_dec.append(dec_out)
                    out_set_list2.append(results)

                results = list2set(out_set_list2)
                dec_output = concat_voxel(part_dec)
                filename = os.path.split(filedir)[-1].split('.')[0]
                save_name = os.path.join(args.resultsdir, filename)
                coords = dec_output.C[:, 1:].detach().cpu().numpy()
                feats = YCoCg2rgb(dec_output.F.detach().cpu().numpy())
                # feats = dec_output.F.detach().cpu().numpy()
                write_ply_ascii(filedir=(save_name + '_dec.ply'), coords=coords, feats=feats)
                pc_results = pc_error((save_name + '_dec.ply'), filedir, 1023, normal=False, color=True)
                gt = load_sparse_tensor(filedir)
                print(pc_results)
                results['filename'] = filename
                results['real_bpp'] = round(results['bits'] / len(gt), 3)
                results['length'] = len(gt)
                print('DBG!!!', idx_file, filedir, results)
                results = pd.DataFrame([results])
                # csvfile = os.path.join(args.resultsdir, filename +'.csv')
                # results.to_csv(csvfile, index=False)
                results_list.append(results)
                if idx_file == 0:
                    results_allfile = results.copy(deep=True)
                else:
                    results_allfile = pd.concat([results_allfile, results], ignore_index=True)

        # average
        csvfile = os.path.join(args.resultsdir, 'all.csv')
        results_allfile.to_csv(csvfile, index=False)
        print('DBG!!!all:\n', results_allfile)

        return results_allfile


    @torch.no_grad()
    def test_part_one(self, model, x, filename, filedir, ref_dir=None):
        # forward
        save_name = os.path.join(args.resultsdir, filename)
        if ref_dir is None:
            # start = time.time()
            with torch.no_grad():
                # out_set = model(x, training=False)
                out_set = model.test(x, filename=save_name)
            # runtime = round(time.time()-start, 3)
        else:
            ref = load_sparse_tensor(ref_dir, color_format=args.color_format, normalize=False)
            with torch.no_grad():
                # out_set = model(x, training=False)
                out_set = model.test(x, x_refT=ref)

        bits = out_set['bitstream_length'] * 8
        #save point cloud

        results = {'enctime': out_set['enctime'], 'dectime': out_set['dectime'],'bits': bits}
        for k, v in results.items():
            if isinstance(v, float):
                results[k] = round(v, 3)

        return results, out_set['out']

    @torch.no_grad()
    def test_seqs(self, filedir_list):
        results_list = []
        if not args.inter_mode:
            for idx_file, filedir in enumerate(tqdm(filedir_list)):
                results = self.test_one(self.model, filedir)
                print('DBG!!!', idx_file, filedir, results)
                results = pd.DataFrame([results])
                filename = os.path.split(filedir)[-1].split('.')[0]
                # csvfile = os.path.join(args.resultsdir, filename +'.csv')
                # results.to_csv(csvfile, index=False)
                results_list.append(results)
                if idx_file == 0:
                    results_allfile = results.copy(deep=True)
                else:
                    results_allfile = pd.concat([results_allfile, results], ignore_index=True)
            csvfile = os.path.join(args.resultsdir, 'all.csv')
            results_allfile.to_csv(csvfile, index=False)
            print('DBG!!!all:\n', results_allfile)
        else:
            for idx_file, filedir in enumerate(tqdm(filedir_list)):
                if idx_file == 0:
                    results = self.test_one(self.model, filedir)
                    print(idx_file, filedir, results)
                else:
                    ref_dir = filedir_list[idx_file - 1]
                    print('inter:\t', idx_file, filedir, ref_dir)
                    results = self.test_one(self.model_inter, filedir, ref_dir=ref_dir)
                    print(idx_file, filedir, results)
                results = pd.DataFrame([results])
                filename = os.path.split(filedir)[-1].split('.')[0]
                csvfile = os.path.join(args.resultsdir, filename + '.csv')
                results.to_csv(csvfile, index=False)
                if idx_file == 0:
                    results_allfile = results.copy(deep=True)
                else:
                    results_allfile = pd.concat([results_allfile,results], ignore_index=True)
                csvfile = os.path.join(args.resultsdir + '_frame' + str(len(filedir_list)) + '.csv')
                results_allfile.to_csv(csvfile, index=False)

        return results_allfile

    @torch.no_grad()
    def test_one(self, model, filedir, ref_dir=None):
        # forward
        x = load_sparse_tensor(filedir, color_format=args.color_format, normalize=False)
        filename = filedir.split('/')[-1].split('.')[0]
        save_name = os.path.join(args.resultsdir, filename)
        if ref_dir is None:
            # start = time.time()
            with torch.no_grad():
                # out_set = model(x, training=False)
                out_set = model.test(x, filename=save_name)
            # runtime = round(time.time()-start, 3)
        else:
            ref = load_sparse_tensor(ref_dir, color_format=args.color_format, normalize=False)
            with torch.no_grad():
                # out_set = model(x, training=False)
                out_set = model.test(x, x_refT=ref, filename=save_name)

        bits = out_set['bitstream_length'] * 8
        real_bpp = round(bits / len(x), 3)
        #save point cloud
        coords = out_set['out'].C[:,1:].detach().cpu().numpy()
        if args.color_format == 'ycocg':
            feats = YCoCg2rgb(out_set['out'].F.detach().cpu().numpy())
        if args.color_format == 'reflectance':
            feats = out_set['out'].F.detach().cpu().numpy()

        write_ply_ascii(filedir=(save_name + '_dec.ply'), coords=coords, feats=feats)
        pc_results = pc_error((save_name + '_dec.ply'), filedir, 1023, normal=False, color=True)
        print(pc_results)
        results = {'filedir': filedir,'enctime': out_set['enctime'], 'dectime': out_set['dectime'],
                   'bits': bits, 'real_bpp':real_bpp, 'length': len(x)}
        for k, v in results.items():
            if isinstance(v, float):
                results[k] = round(v, 3)

        return results


if __name__ == '__main__':

    args.resultsdir = os.path.join(args.resultsdir, args.prefix)
    os.makedirs(args.resultsdir, exist_ok=True)

    filedir_list = sorted(glob.glob(os.path.join(args.testdata, '**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]

    if len(filedir_list) > args.testdata_num:
        if args.testdata_seqs == 'random':
            filedir_list = filedir_list[::len(filedir_list) // args.testdata_num]
        if args.testdata_seqs == 'frame' or args.inter_mode:
            filedir_list = filedir_list[:args.testdata_num]

    for i, f in enumerate(filedir_list):
        print('filedir', i, f)

    ################# test #################
    tester = Tester()
    if args.partition:
        tester.test_partition(filedir_list=filedir_list, partition_num=args.part_nums)
    else:
        tester.test_seqs(filedir_list=filedir_list)

