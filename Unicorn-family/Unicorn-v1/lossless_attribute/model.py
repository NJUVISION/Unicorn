# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-04

import os, sys, time
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
from basic_models.conditional_entropy_model import SymmetricConditional
from basic_models.factorized_entropy_model import get_entropy
from data_utils.LoD_utils import quantize, split_channels, concat_channels
from lossless_attribute.model_multistage import ModelStage
from lossless_attribute.model_cls import ModelCls
from lossless_attribute.utils import sort_sparse_tensor
from cfg.get_args import get_args
args = get_args(component='attribute')


################################## ##################################
class Model(torch.nn.Module):
    """each scale shares the same model.
    """
    def __init__(self, inter_mode=None):
        super().__init__()
        if inter_mode is None: self.inter_mode = args.inter_mode
        else: self.inter_mode = inter_mode

        stride_list = [[2,2,2]]
        if args.split_channel and args.in_channels==3: channels_list=[1,2]
        else: channels_list=[args.in_channels]

        self.model_list = torch.nn.ModuleList()
        for stride in stride_list:
            self.model_list.append(ModelScale(
                kernel_size=args.kernel_size, stride=stride, 
                channels=args.channels, channels_list=channels_list, 
                split_group=args.split_group, 
                block_type=args.block_type, 
                inter_mode=self.inter_mode))
        self.conditional_entropy_model = SymmetricConditional(distribution='laplace')

    def forward(self, x, x_refT=None, training=True):
        if not self.inter_mode:
            x_refT = None
            
        out = x
        out_refT = x_refT
        out_set_list2 = []
        cls_set_list = []
        for _ in range(args.scale):
            for model in self.model_list:
                if not self.inter_mode or out_refT is None:
                    out, out_set_list, cls_set = model(out, out_refT)
                else:
                    out, out_refT, out_set_list, cls_set = model(out, out_refT)
                out_set_list2.append(out_set_list)
                cls_set_list.append(cls_set)

                # torch.cuda.empty_cache()
            if len(out)<32: break
        #
        out_set, shape = list2set2(out_set_list2)
        out_set = self.collect_loss(out_set)
        cls_set = list2set(cls_set_list)
        out_set.update(cls_set)

        return out_set

    @torch.no_grad()
    def test(self, x, x_refT=None, filename=None):
        """test real encoding & decoding
        """
        if not self.inter_mode:
            x_refT = None
        start_run = time.time()
        out = x
        out_refT = x_refT
        out_set_list2 = []
        cls_set_list = []
        y_raw_list = []
        x_refT_list = []
        start_enc = time.time()
        for idx_scale in range(args.scale):
            true_filename = filename + '_scale' + str(idx_scale)
            print('DBG!!!', idx_scale, out.shape[0])
            for model in self.model_list:
                if not self.inter_mode:
                    out, out_set_list, cls_set, y_raw_list = model.encode(out, filename=true_filename, y_raw_list=y_raw_list)
                else:
                    out, out_refT, out_set_list, cls_set, y_raw_list, x_refT_list = model.encode(out, x_refT=out_refT, filename=true_filename, y_raw_list=y_raw_list, x_refT_list=x_refT_list)
                out_set_list2.append(out_set_list)
                cls_set_list.append(cls_set)
            if len(out)<32:
                np.savetxt(filename + '_finalcolor.txt', out.F.short().detach().cpu().numpy())
                base_bytes = os.path.getsize(filename + '_finalcolor.txt')
                break
        enctime = round(time.time()-start_enc, 3)

        start_dec = time.time()

        for idx_scale in range(len(y_raw_list) - 1, -1, -1):
            out = ME.SparseTensor(features=out.F, coordinates=out.C * 2, device=out.device)
            true_filename = filename + '_scale' + str(idx_scale)
            y_raw = y_raw_list[idx_scale]
            if self.inter_mode:
                out_refT = x_refT_list[idx_scale]
            else:
                out_refT = None
            out = model.decode(y_raw, out, x_refT=out_refT, filename=true_filename)

        dectime = round(time.time()-start_dec, 3)

        out = sort_sparse_tensor(out)
        out_set = list2set(out_set_list2)
        cls_set = list2set(cls_set_list)
        bitstream_length = out_set['true_bytes'] + cls_set['cls_bytes'] + base_bytes
        # bitstream_length.append(out.F.shape[0]*out.F.shape[1])
        print('DBG!!! bitstream_length', bitstream_length)
        # print('DBG!!!:\t', 'runtime',runtime, 'enctime',enctime, 'dectime',dectime)

        out_set.update({'out':out, 'enctime':enctime, 'dectime':dectime, 'bitstream_length':bitstream_length})

        return out_set

    def collect_loss(self, out_set):
        likelihood_list, res_list, res_bits_list = [], [], []
        init_bits_list, init_res_list = [], []

        for target, pred, init_pred in zip(out_set['target_list'], out_set['pred_list'], out_set['init_pred_list']):
            assert (pred.C==target.C).all() and (init_pred.C==target.C).all()
            if pred.shape[-1]==target.shape[-1]*2:
                loc = pred.F[:, :pred.F.shape[-1]//2]
                scale = pred.F[:, pred.F.shape[-1]//2:]
                # start = time.time()
                _, likelihood = self.conditional_entropy_model(target.F, loc, scale, quantize_mode=None)
                res = loc - target.F
            elif pred.shape[-1]==target.shape[-1]:
                likelihood = pred.F
                res = torch.tensor(0).float().to(likelihood.device)
            #
            init_res = init_pred.F - target.F
            init_bits = get_entropy(init_res.round().detach().cpu().numpy())
            res_bits = get_entropy(res.round().detach().cpu().numpy())
            likelihood_list.append(likelihood)
            res_list.append(res)
            res_bits_list.append(res_bits)
            init_bits_list.append(init_bits)
            init_res_list.append(init_res)

        return_set = {'likelihood_list': likelihood_list, 'res_list': res_list, 'res_bits_list': res_bits_list, 
                'init_bits_list': init_bits_list, 'init_res_list':init_res_list}

        return return_set


################################ Scale-wise prediction ################################
class ModelScale(torch.nn.Module):
    """predict across different scales/dimension.
        similar to SR but output likelihood.
    """
    def __init__(self, channels=64, kernel_size=3, block_layers=3, channels_list=[1,2], stride=[2,2,2], 
                split_group=True, block_type='conv', inter_mode=0):
        super().__init__()
        self.stride = stride
        self.inter_mode = inter_mode
        self.avgpooling = ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=3)
        self.sumpooling = ME.MinkowskiSumPooling(kernel_size=stride, stride=stride, dimension=3)
        self.unpooling = ME.MinkowskiPoolingTranspose(kernel_size=stride, stride=stride, dimension=3)
        self.model = ModelChannel(channels=channels, kernel_size=kernel_size, block_layers=block_layers, 
                    channels_list=channels_list, stride=stride, split_group=args.split_group, 
                    block_type=block_type, inter_mode=inter_mode)
        self.model_cls = ModelCls(channels=channels, in_channels=sum(channels_list), 
                        kernel_size=kernel_size, block_layers=block_layers, block_type=block_type, stride=stride)
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

    def forward(self, x, x_refT=None, training=True):
        if not self.inter_mode:
            x_refT = None
        # quantize
        y_raw = self.avgpooling(x)
        y = quantize(y_raw)

        cls_set = self.model_cls(x, y)
        out_set_list = self.model(x, y_raw, x_refT=x_refT, training=training)
        out = ME.SparseTensor(features=y.F, coordinates=torch.div(y.C, 
                            torch.tensor([1]+y.tensor_stride).to(y.device), 
                            rounding_mode="floor").int(), tensor_stride=1, device=y.device)

        out_refT = None
        if self.inter_mode:
            assert x_refT is not None
            out_refT = self.avgpooling(x_refT)
            out_refT = ME.SparseTensor(features=out_refT.F, coordinates=torch.div(out_refT.C, 
                            torch.tensor([1]+out_refT.tensor_stride).to(out_refT.device), 
                            rounding_mode="floor").int(), tensor_stride=1, device=out_refT.device)
        
        if out_refT is None:
            return out, out_set_list, cls_set
        else:
            return out, out_refT, out_set_list, cls_set

    @torch.no_grad()
    def encode(self, x, x_refT=None, filename=None, y_raw_list=None, x_refT_list=None):
        if not self.inter_mode:
            x_refT = None
        y_raw = self.avgpooling(x)
        true_y_sum = self.sumpooling(x)
        y = quantize(y_raw)
        cls_set = self.model_cls.encode(x, y, filename=filename, y_sum=true_y_sum)
        out_set_list = self.model.encode(x, y_raw, x_refT=x_refT, filename=filename)
        if self.inter_mode:
            x_refT_list.append(x_refT)

        out = ME.SparseTensor(features=y.F, coordinates=torch.div(y.C,
                            torch.tensor([1]+y.tensor_stride).to(y.device),
                            rounding_mode="floor").int(), tensor_stride=1, device=y.device)
        y_raw_list.append(y_raw)
        if self.inter_mode:
            assert x_refT is not None
            out_refT = self.avgpooling(x_refT)
            out_refT = ME.SparseTensor(features=out_refT.F, coordinates=torch.div(out_refT.C,
                                                                                  torch.tensor(
                                                                                      [1] + out_refT.tensor_stride).to(
                                                                                      out_refT.device),
                                                                                  rounding_mode="floor").int(),
                                       tensor_stride=1, device=out_refT.device)

        if not self.inter_mode:
            return out, out_set_list, cls_set, y_raw_list
        else:
            return out, out_refT, out_set_list, cls_set, y_raw_list, x_refT_list

    @torch.no_grad()
    def decode(self, y_raw, y, x_refT=None,x_refT_ori=None,filename=None):
        torch.cuda.empty_cache()
        y = self.sort(y, y_raw.C)
        y = ME.SparseTensor(features=y.F, coordinate_manager=y_raw.coordinate_manager,
                            coordinate_map_key=y_raw.coordinate_map_key, device=y_raw.device)
        if not self.inter_mode:
            x_refT = None
        if self.inter_mode:
            x_refT= x_refT
        up_x = self.unpooling(y)
        y_sum_dec = self.model_cls.decode(up_x, y, filename=filename)
        x_num = ME.SparseTensor(torch.ones([len(up_x), 1]).float(),
            coordinate_map_key=up_x.coordinate_map_key,
            coordinate_manager=up_x.coordinate_manager, device=up_x.device)
        y_num = self.sumpooling(x_num)
        torch.cuda.empty_cache()
        #
        y_raw = ME.SparseTensor(features=y_sum_dec.F / y_num.F, coordinate_manager=y.coordinate_manager,
                                coordinate_map_key=y.coordinate_map_key, device=y.device)
        true_dec = self.model.decode(y_raw, y, x_refT=x_refT, filename=filename, true_y_sum=y_sum_dec)
        torch.cuda.empty_cache()

        return true_dec



################################ Channel-wise prediction ################################

class ModelChannel(torch.nn.Module):
    def __init__(self, channels=128, kernel_size=3, block_layers=3, channels_list=[3], stride=[2,2,2], 
                split_group=True, block_type='conv', inter_mode=0):
        super().__init__()
        self.channels_list = channels_list
        self.model_list = torch.nn.ModuleList()
        self.inter_mode = inter_mode
        for idx in range(len(channels_list)):
            if split_group:
                self.model_list.append(ModelStage(
                    channels=channels, in_channels=channels_list[idx], out_channels=channels_list[idx],
                    kernel_size=kernel_size, block_layers=block_layers, ref_channels=sum(channels_list[:idx]), 
                    stride=stride, split_group=split_group, block_type=block_type, 
                    inter_mode=inter_mode))


    def forward(self, x, y, x_refT=None, training=True):
        """y is the voxels of previous scale. for YCgCo
        """
        if not self.inter_mode: x_refT = None
        # split
        x_list = split_channels(x, channels_list=self.channels_list)
        y_list = split_channels(y, channels_list=self.channels_list)
        # progressive decode
        ref_list = []
        out_set_list = []
        for idx in range(len(x_list)):    
            curr_x = x_list[idx]
            curr_y = y_list[idx]
            curr_model = self.model_list[idx]
            x_ref = concat_channels(ref_list)
            out_set = curr_model(x=curr_x, y=curr_y, x_ref=x_ref, x_refT=x_refT, training=training)
            out_set_list.append(out_set)
            ref_list.append(curr_x)

        return out_set_list

    @torch.no_grad()
    def encode(self, x, y,  x_refT=None, filename=None):
        if not self.inter_mode: x_refT = None
        # split
        x_list = split_channels(x, channels_list=self.channels_list)
        y_list = split_channels(y, channels_list=self.channels_list)
        # progressive decode
        ref_list = []
        out_set_list = []
        for idx in range(len(x_list)):
            if idx == 0:
                true_filename = filename + '_Y'
            else:
                true_filename = filename + '_CoCg'
            curr_x = x_list[idx]
            curr_y = y_list[idx]
            curr_model = self.model_list[idx]
            x_ref = concat_channels(ref_list)
            out_set = curr_model.encode(x=curr_x, y=curr_y, filename=true_filename, x_ref=x_ref, x_refT=x_refT)
            torch.cuda.empty_cache()
            out_set_list.append(out_set)
            ref_list.append(curr_x)
        # merge results
        out_set = list2set(out_set_list)

        return {'true_bytes': out_set['true_bytes']}

    @torch.no_grad()
    def decode(self, y_raw, y, x_refT=None, filename=None, true_y_sum=None):
        if not self.inter_mode: x_refT = None
        # split
        y_raw_list = split_channels(y_raw, channels_list=self.channels_list)
        true_y_sum_list = split_channels(true_y_sum, channels_list=self.channels_list)
        # progressive decode
        ref_list = []
        for idx in range(len(y_raw_list)):
            if idx == 0:
                true_filename = filename + '_Y'
            else:
                true_filename = filename + '_CoCg'
            curr_y = y_raw_list[idx]
            curr_y_sum = true_y_sum_list[idx]
            curr_model = self.model_list[idx]
            x_ref = concat_channels(ref_list)
            # if x_ref is not None:
            #     x_ref = sort_sparse_tensor(x_ref)
            #     np.savetxt(filename + '_Y_dec.txt', x_ref.F.short().detach().cpu().numpy(), fmt='%d')
            dec = curr_model.decode(y=curr_y,y_dec=y, filename=true_filename, true_y_sum=curr_y_sum, x_ref=x_ref, x_refT=x_refT)
            ref_list.append(dec)
        # merge results
        true_dec = concat_channels(ref_list)
        del ref_list
        torch.cuda.empty_cache()
        return true_dec


################################ utils ################################

def list2set2(out_set_list2):
    """convert a 2-D list to a set.
    """
    # out_set = {'target_list':[], 'pred_list':[], 'init_pred_list':[]}
    out_set = {}
    for i, out_set_list1 in enumerate(out_set_list2):
        # 1 scale/dim
        for j, curr_set in enumerate(out_set_list1):
            # 1 channel
            for key in curr_set.keys():
                if key not in out_set: out_set[key] = []
                out_set[key] += curr_set[key]
    shape = [i+1,j+1]
    
    return out_set, shape


def list2set(out_set_list):
    out_set = out_set_list[0]
    for curr_out_set in out_set_list[1:]:
        for k in curr_out_set.keys():
            out_set[k] += curr_out_set[k]
    
    return out_set
