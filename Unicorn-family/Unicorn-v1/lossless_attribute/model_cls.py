# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

import os, sys, time
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
from data_utils.LoD_utils import split_value, split_channels
from basic_models.backbone import make_convNet, make_linearNet
from lossless_attribute.utils import concat_channels, concat_voxel

################################ Multi Scale ################################
class ModelCls(torch.nn.Module):
    """predict the probability of rounding error by classification.
    """
    def __init__(self, channels=128, in_channels=3, kernel_size=3, block_layers=3, block_type='conv', stride=[2,2,2]):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.label_list = [i for i in range(2, np.prod(stride)+1)]# number of label
        self.sumpooling = ME.MinkowskiSumPooling(kernel_size=stride, stride=stride, dimension=3)
        self.avgpooling = ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=3)
        self.unpooling = ME.MinkowskiPoolingTranspose(kernel_size=stride, stride=stride, dimension=3)
        self.pruning = ME.MinkowskiPruning()
        # model for different channels
        self.convNet_list = torch.nn.ModuleList([make_convNet(
            channels=channels, in_channels=1, out_channels=channels, 
            kernel_size=kernel_size, block_layers=block_layers, block_type=block_type) for i in range(in_channels)])
        self.clsNet_list2 = torch.nn.ModuleList([
            torch.nn.ModuleList([make_linearNet(channels=channels, in_channels=channels, out_channels=idx)\
                for idx in self.label_list]) for j in range(in_channels)])
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)


    def predict_cls(self, data, data_num, data_gt, ctxNet, clsNet_list, DBG=False):
        """ x: input value
            x_num: the number of categories.
            x_gt: the ground truth.
        """
        # print('DBG!!! predict_cls data_num:\n', data_num)
        # 1. extract context
        ctx = ctxNet(data)
        # 2. split for different classification numbers
        ctx_list = split_value(ctx, data_num, pruning=self.pruning, value_list=self.label_list)
        # 3. obatin ground truth
        gt_list = split_value(data_gt, data_num, pruning=self.pruning, value_list=self.label_list)
        # 4. classification
        prob_list = []
        init_prob_list = []
        # flag = 0
        for ctx, gt, clsNet in zip(ctx_list, gt_list, clsNet_list):
            # 1. different channels represent different classes
            # print('DBG!!! predict_cls, gt:\n', gt.F.max().item(), gt.F.min().item(), gt.F.mean().item())
            prob = clsNet(ctx)
            prob = self.softmax_fn(prob.F)
            prob = torch.clamp(prob, min=1e-8, max=1-1e-8)

            if DBG:
                import torchac
                # to CDF
                cdf = prob.cumsum(dim=-1)
                cdf = torch.cat([torch.zeros([len(cdf),1]).to(cdf.device), cdf], dim=-1).cpu()
                value = gt.F.short().cpu().squeeze()
                bitstream = torchac.encode_float_cdf(cdf, value)
                print('DBG!!!bitstream', len(bitstream)*8)
            init_prob = torch.ones(len(prob)).float() * 1/prob.shape[-1]
            # print('DBG!!! raw prob \n', prob.sum(dim=-1).mean())
            # 2. select one channel according to the gt
            assert (ctx.C==gt.C).all()
            prob = prob.cpu()[torch.arange(len(gt)).long(), gt.F.long().squeeze().cpu()]
            prob = prob.to(ctx.device)
            # collect
            prob_list.append(prob)
            if DBG:
                from basic_models.loss import get_bits
                bits = get_bits(prob).round().item()
                print('DBG!!! entropy:\t', bits)

                from collections import Counter
                # statistic
                data = gt.F.cpu().int().numpy().reshape(-1)
            
            init_prob_list.append(init_prob)

        return {'prob_list': prob_list, 'init_prob_list':init_prob_list}   

    def forward(self, x, y, training=True, DBG=False):
        """predict the rounding error of y; 
            y is the quantized value.
            x is the ground truth.
        """
        # 1. obtain ground truth for classification by sumpooling: gt is just the offset value.
        x_num = ME.SparseTensor(torch.ones([len(x), 1]).float(), 
            coordinate_map_key=x.coordinate_map_key, 
            coordinate_manager=x.coordinate_manager, device=x.device)
        y_num = self.sumpooling(x_num)
        # obtain rounding error by subtraction.
        y_sum = self.sumpooling(x)
        assert (y.C==y_sum.C).all() and (y.C==y_num.C).all()
        y_gt = ME.SparseTensor(y_sum.F - y.F*y_num.F + torch.floor(y_num.F/2),
            coordinate_map_key=y.coordinate_map_key, 
            coordinate_manager=y.coordinate_manager, device=y.device)

        # print('DBG!!!label:\t', y_gt.F.max(), y_gt.F.min())
        # 1: [0]
        # 2: [-1,0]
        # 3: [-1,0,1]
        # 4: [-2,-1,0,1]
        # 5: [-2,-1,0,1,2]
        # 6: [-3,-2,-1,0,1,2]
        # 7: [-3,-2,-1,0,1,2,3]
        # 8: [-4,-3,-2,-1,0,1,2,3]
        # 2. predict for each channel (YCoCg) seperately
        y_list = split_channels(y, [1]*self.in_channels)
        y_gt_list = split_channels(y_gt, [1]*self.in_channels)
        
        out_set_list = []
        for data, data_gt, convNet, clsNet_list in zip(y_list, y_gt_list, self.convNet_list, self.clsNet_list2):
            out_set = self.predict_cls(data=data, data_num=y_num, data_gt=data_gt, 
                                    ctxNet=convNet, clsNet_list=clsNet_list)
            # enc_set = self.encode(data=data, data_num=y_num, data_gt=data_gt, 
            #                     ctxNet=convNet, clsNet_list=clsNet_list)
            out_set_list.append(out_set)
        out_set = list2set(out_set_list)

        return out_set
    @torch.no_grad()
    def encode(self, x, y, filename=None, y_sum=None):
        """predict the rounding error of y;
            y is the quantized value.
            x is the ground truth.
        """
        # 1. obtain ground truth for classification by sumpooling: gt is just the offset value.
        filename = filename + '_frac'
        x_num = ME.SparseTensor(torch.ones([len(x), 1]).float(),
                                coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager, device=x.device)
        y_num = self.sumpooling(x_num)  # getting voxel occupancy statues in the eight stage
        y_sum = self.sumpooling(x)
        # obtain rounding error by subtraction.
        assert (y.C == y_sum.C).all() and (y.C == y_num.C).all()
        # need coding
        y_gt = ME.SparseTensor(y_sum.F - y.F * y_num.F + torch.floor(y_num.F / 2),
                               coordinate_map_key=y.coordinate_map_key,
                               coordinate_manager=y.coordinate_manager, device=y.device)
        y_list = split_channels(y, [1] * self.in_channels)
        y_gt_list = split_channels(y_gt, [1] * self.in_channels)

        out_set_list = []
        idx = 0
        for data, data_gt, convNet, clsNet_list in zip(y_list, y_gt_list, self.convNet_list, self.clsNet_list2):
            if idx == 0:
                true_filename = filename + '_Y'
            elif idx == 1:
                true_filename = filename + '_Co'
            elif idx == 2:
                true_filename = filename + '_Cg'
            out_set = self.encode_cls(data=data, data_num=y_num, data_gt=data_gt,
                                      ctxNet=convNet, clsNet_list=clsNet_list, filename=true_filename)

            idx += 1
            out_set_list.append(out_set)
        out_set = list2set(out_set_list)

        return out_set

    @torch.no_grad()
    def encode_cls(self, data, data_num, data_gt, ctxNet, clsNet_list, filename):
        ctx = ctxNet(data)
        ctx_list = split_value(ctx, data_num, pruning=self.pruning, value_list=self.label_list)
        gt_list = split_value(data_gt, data_num, pruning=self.pruning, value_list=self.label_list)
        prob_list = []
        init_prob_list = []
        idx = 0
        Bytes = 0
        for ctx, gt, clsNet in zip(ctx_list, gt_list, clsNet_list):
            prob = clsNet(ctx)
            prob = self.softmax_fn(prob.F)
            prob = torch.clamp(prob, min=1e-8, max=1 - 1e-8)
            # print('DBG, Prob:\t', prob)
            # if DBG:
            import torchac
            # to CDF
            cdf = prob.cumsum(dim=-1)
            # cdf = cdf.cpu()
            cdf = torch.cat([torch.zeros([len(cdf), 1]).to(cdf.device), cdf], dim=-1).cpu()
            if gt.F.shape[0] == 1:
                value = gt.F.short().cpu().squeeze(1)
            else:
                value = gt.F.short().cpu().squeeze()
            bitstream = torchac.encode_float_cdf(cdf, value)
            with open(filename + '_' + str(idx) + '_A.bin', 'wb') as fout:
                fout.write(bitstream)
            Bytes += os.path.getsize(filename + '_' + str(idx) + '_A.bin')
            idx += 1
        return {'cls_bytes': Bytes}

    @torch.no_grad()
    def decode(self, x, y, filename=None):
        """predict the rounding error of y;
            y is the quantized value.
            x only a coordinates
        """
        # 1. obtain ground truth for classification by sumpooling: gt is just the offset value.
        filename = filename + '_frac'
        x_num = ME.SparseTensor(torch.ones([len(x), 1]).float(),
                                coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager, device=x.device)
        y_num = self.sumpooling(x_num)  # getting voxel occupancy statues in the eight stage
        # obtain rounding error by subtraction.
        # need coding
        y_list = split_channels(y, [1] * self.in_channels)
        dec_value_list = []
        idx = 0
        for data, convNet, clsNet_list in zip(y_list, self.convNet_list, self.clsNet_list2):
            if idx == 0:
                true_filename = filename + '_Y'
            elif idx == 1:
                true_filename = filename + '_Co'
            elif idx == 2:
                true_filename = filename + '_Cg'
            out_set = self.decode_cls(data=data, data_num=y_num,
                                      ctxNet=convNet, clsNet_list=clsNet_list, filename=true_filename)

            dec_value_list.append(out_set['value'])
            idx += 1
        y_gt_dec = concat_channels(dec_value_list)
        y_dec = self.sort(y_gt_dec, y.C)
        y_sum_true = ME.SparseTensor(y.F * y_num.F - torch.floor(y_num.F / 2) + y_dec.F,
                                     coordinate_map_key=y.coordinate_map_key,
                                     coordinate_manager=y.coordinate_manager, device=y.device)
        # print('!'*200, 'DBG!!!prob_list:\n', len(out_set['prob_list']),
        #     '\n', [tp.shape[0] for tp in out_set['prob_list']],
        #     '\n', [round(tp.mean().item(), 2) for tp in out_set['prob_list']])
        # print('!'*200, 'DBG!!!init_prob_list:\n', len(out_set['init_prob_list']),
        #     '\n', [tp.shape[0] for tp in out_set['init_prob_list']],
        #     '\n', [round(tp.mean().item(), 2) for tp in out_set['init_prob_list']])
        del y_gt_dec, dec_value_list
        torch.cuda.empty_cache()
        return y_sum_true

    @torch.no_grad()
    def decode_cls(self, data, data_num, ctxNet, clsNet_list, filename):
        ctx = ctxNet(data)
        ctx_list = split_value(ctx, data_num, pruning=self.pruning, value_list=self.label_list)
        data_list = split_value(data, data_num, pruning=self.pruning, value_list=self.label_list)
        single_list = split_value(data, data_num, pruning=self.pruning, value_list=[1])
        if len(single_list) > 0:
            single_voxel = ME.SparseTensor(features=torch.zeros(len(single_list[0]), 1).cuda(),
                                           coordinate_manager=single_list[0].coordinate_manager,
                                           coordinate_map_key=single_list[0].coordinate_map_key,
                                           device=single_list[0].device)
            dec_list = [single_voxel]
        else:
            dec_list = []
        idx = 0
        for ctx, clsNet, data_slice in zip(ctx_list, clsNet_list, data_list):
            prob = clsNet(ctx)
            prob = self.softmax_fn(prob.F)
            prob = torch.clamp(prob, min=1e-8, max=1 - 1e-8)
            # print('DBG, Prob:\t', prob)
            # if DBG:

            import torchac
            # to CDF
            cdf = prob.cumsum(dim=-1)
            # cdf = cdf.cpu()
            cdf = torch.cat([torch.zeros([len(cdf), 1]).to(cdf.device), cdf], dim=-1).cpu()
            with open(filename + '_' + str(idx) + '_A.bin', 'rb') as fin:
                strings = fin.read()
            values = torchac.decode_float_cdf(cdf, strings).float().cuda().unsqueeze(1)

            value_sparsetensor = ME.SparseTensor(features=values, coordinate_manager=data_slice.coordinate_manager,
                                                 coordinate_map_key=data_slice.coordinate_map_key,
                                                 device=data_slice.device)
            dec_list.append(value_sparsetensor)
            idx += 1
        value = concat_voxel(dec_list)
        return {'value': value}


def list2set(out_set_list):
    out_set = out_set_list[0]
    for curr_out_set in out_set_list[1:]:
        for k in curr_out_set.keys():
            out_set[k] += curr_out_set[k]
    
    return out_set
