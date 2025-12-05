# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-9-21

import os, sys

sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import time
import numpy as np
import torch
import MinkowskiEngine as ME
from data_utils.sparse_tensor import isin, istopk_local, istopk_global, sort_sparse_tensor, array2vector
from lossless_geometry.model_upsampler import UpSampler, Classifier
from lossy_geometry.model_autoencoder import DownSampler
from basic_models.factorized_entropy_model import EntropyBottleneck
from model_compensator import CompensatorConvSimple


#########################################
class PCCModel(torch.nn.Module):
    """Dynamic PCC Model
    """

    def make_encoder(self, channels, scale):
        if scale == 3:
            # channels: [1,c,2c,    2c,2c,2c,   2c,2c,2c]
            downsampler_list = torch.nn.ModuleList([
                DownSampler(in_channels=1, channels=channels, out_channels=channels * 2,
                            kernel_size=3, block_layers0=3, block_layers1=3),
                DownSampler(in_channels=channels * 2, channels=channels * 2, out_channels=channels * 2,
                            kernel_size=3, block_layers0=0, block_layers1=3),
                DownSampler(in_channels=channels * 2, channels=channels * 2, out_channels=channels,
                            kernel_size=3, block_layers0=0, block_layers1=3)])
        if scale == 2:
            # channels: [1,c,2c,    2c,2c,c]
            downsampler_list = torch.nn.ModuleList([
                DownSampler(in_channels=1, channels=channels, out_channels=channels * 2,
                            kernel_size=3, block_layers0=3, block_layers1=3),
                DownSampler(in_channels=channels * 2, channels=channels * 2, out_channels=channels,
                            kernel_size=3, block_layers0=0, block_layers1=3)])
        if scale == 1:
            # channels: [1,c,c]
            downsampler_list = torch.nn.ModuleList([
                DownSampler(in_channels=1, channels=channels, out_channels=channels,
                            kernel_size=3, block_layers0=3, block_layers1=3)])

        return downsampler_list

    def make_inter_encoder(self, channels, scale):
        if scale == 3:
            # channels: [1,c,2c,    2c,2c,2c,   2c,2c,2c]
            downsampler_list = torch.nn.Sequential(
                DownSampler(in_channels=1, channels=channels, out_channels=channels * 2,
                            kernel_size=3, block_layers0=3, block_layers1=3),
                DownSampler(in_channels=channels * 2, channels=channels * 2, out_channels=channels * 2,
                            kernel_size=3, block_layers0=0, block_layers1=3),
                DownSampler(in_channels=channels * 2, channels=channels * 2, out_channels=channels,
                            kernel_size=3, block_layers0=0, block_layers1=3))
        if scale == 2:
            # channels: [1,c,2c,    2c,2c,c]
            downsampler_list = torch.nn.Sequential(
                DownSampler(in_channels=1, channels=channels, out_channels=channels * 2,
                            kernel_size=3, block_layers0=3, block_layers1=3),
                DownSampler(in_channels=channels * 2, channels=channels * 2, out_channels=channels,
                            kernel_size=3, block_layers0=0, block_layers1=3))
        if scale == 1:
            # channels: [1,c,c]
            downsampler_list = DownSampler(in_channels=1, channels=channels, out_channels=channels,
                                           kernel_size=3, block_layers0=3, block_layers1=3)

        return downsampler_list

    def make_decoder(self, channels, scale):
        if scale == 3:
            # channels: [c,2c,2c,    2c,2c,c,   c,c,c]
            upsampler_list = torch.nn.ModuleList([
                UpSampler(in_channels=channels, channels=channels * 2, out_channels=channels,
                          kernel_size=3, block_layers0=3, block_layers1=0, expand_coordinates=True),
                UpSampler(in_channels=channels, channels=channels * 2, out_channels=channels,
                          kernel_size=3, block_layers0=3, block_layers1=0, expand_coordinates=True),
                UpSampler(in_channels=channels, channels=channels, out_channels=channels,
                          kernel_size=3, block_layers0=3, block_layers1=3, expand_coordinates=True)])
        if scale == 2:
            # channels: [c,2c,c,   c,c,c]
            upsampler_list = torch.nn.ModuleList([
                UpSampler(in_channels=channels, channels=channels * 2, out_channels=channels,
                          kernel_size=3, block_layers0=3, block_layers1=0, expand_coordinates=True),
                UpSampler(in_channels=channels, channels=channels, out_channels=channels,
                          kernel_size=3, block_layers0=3, block_layers1=3, expand_coordinates=True)])
        if scale == 1:
            # channels: [c,2c,2c,   c,c,c]
            upsampler_list = torch.nn.ModuleList([
                UpSampler(in_channels=channels, channels=channels * 2, out_channels=channels,
                          kernel_size=3, block_layers0=3, block_layers1=0, expand_coordinates=False),
                UpSampler(in_channels=channels, channels=channels, out_channels=channels,
                          kernel_size=3, block_layers0=3, block_layers1=3, expand_coordinates=True)])

        return upsampler_list

    def __init__(self, mode='inter', channels=32, scale=3, stage=1, kernel_size=3, block_layers=3, compensator=1,
                 lossyMode='concat', enc_scale=0):
        """extract feature form pc0 & reconstruct pc1
           transport pc0's feature to pc1
        """
        super().__init__()
        self.mode = mode
        self.channels = channels
        # self.lossyMode = lossyMode
        self.scale = scale
        # intra
        self.downsampler_list = self.make_encoder(channels=channels, scale=max(scale, 2))
        self.upsampler_list = self.make_decoder(channels=channels, scale=scale)
        self.classifier_list = torch.nn.ModuleList([Classifier(channels) for i in range(self.scale)])
        #
        self.entropy_bottleneck = EntropyBottleneck(channels=channels)
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.pruning = ME.MinkowskiPruning()
        # inter
        if mode == 'inter':
            self.inter_downsampler_list = torch.nn.ModuleList([
                self.make_inter_encoder(channels=channels, scale=i) for i in range(self.scale, 0, -1)])
            self.compensator_list = torch.nn.ModuleList([
                CompensatorConvSimple(channels=channels, kernel_size=9, scale=enc_scale) for i in range(self.scale)])
            self.channel_converter_list = torch.nn.ModuleList([
                ME.MinkowskiConvolution(in_channels=channels * 2, out_channels=channels,
                                        kernel_size=1, stride=1, bias=True, dimension=3) for i in range(self.scale)])

    def intra_encode(self, x):
        out_list = []
        out = x
        for downsampler in self.downsampler_list:
            out = downsampler(out)
            out_list.append(out)
            # print('DBG!!!', out.shape)

        return out_list

    def intra_decode(self, x_low, nums_list, gt_list, training):
        out_cls_list = []
        out = x_low
        for idx in range(self.scale):
            upsampler = self.upsampler_list[idx]
            classifier = self.classifier_list[idx]
            # classifier = self.classifier
            nums = nums_list[::-1][idx]
            gt = gt_list[::-1][idx]
            #
            if self.scale == 1:
                out = self.upsampler_list[1](self.upsampler_list[0](out))
            else:
                out = upsampler(out)
            out_cls = classifier(out)
            # print('DBG!!! intra_decode', out_cls.shape, out_cls.F.abs().max(),  out_cls.F.abs().sum())
            # print('DBG!!! intra_decode:\t', out.shape)
            out = self.prune_voxel(out, out_cls, nums=nums, ground_truth=gt, training=training)
            # print('DBG!!! intra_decode:\t', out.shape, nums)

            out_cls_list.append(out_cls)

        out_cls_list = out_cls_list[::-1]
        return {'out_cls_list': out_cls_list, 'out': out}

    def inter_decode(self, x1_low, x0, nums_list, gt_list, training):
        out_cls_list = []
        out1 = x1_low
        for idx in range(self.scale):
            inter_downsampler = self.inter_downsampler_list[idx]
            compensator = self.compensator_list[idx]
            channel_converter = self.channel_converter_list[idx]
            if self.scale != 1: upsampler = self.upsampler_list[idx]
            # classifier = self.classifier_list[idx]
            classifier = self.classifier_list[idx]
            nums = nums_list[::-1][idx]
            gt = gt_list[::-1][idx]
            # inter prediction: extract & compensate(conv) & concat
            out0 = inter_downsampler(x0)
            if self.scale == 1: out1 = self.upsampler_list[0](out1)
            assert out0.tensor_stride[0] == out1.tensor_stride[0]
            out1_pred = compensator(out0, out1)
            assert (out1_pred.C == out1.C).all()
            # print('DBG!!! inter_decode:\t', idx, out1_pred.shape, out1.shape)
            out1_concat = ME.SparseTensor(torch.cat([out1.F, out1_pred.F], dim=1),
                                          coordinate_map_key=out1.coordinate_map_key,
                                          coordinate_manager=out1.coordinate_manager,
                                          device=out1.device)
            out1 = channel_converter(out1_concat)
            #
            if self.scale == 1:
                out1 = self.upsampler_list[1](out1)
            else:
                out1 = upsampler(out1)
            out1_cls = classifier(out1)
            # print('DBG!!! intra_decode:\t', out1.shape)
            out1 = self.prune_voxel(out1, out1_cls, nums=nums, ground_truth=gt, training=training)
            # print('DBG!!! intra_decode:\t', out1.shape, nums)
            out_cls_list.append(out1_cls)

        out_cls_list = out_cls_list[::-1]
        return {'out_cls_list': out_cls_list, 'out': out1}

    def prune_voxel(self, data, data_cls, nums, ground_truth=None, training=False):
        prob = torch.sigmoid(data_cls.F)
        mask = istopk_local(prob, k=1)
        prob[torch.where(mask)[0]] = 1
        mask_topk = istopk_global(prob, k=nums)
        if training:
            assert ground_truth is not None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        # mask = isin(data_cls.C, ground_truth.C)# DBG!!!
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F, quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(features=data_F, coordinate_map_key=data.coordinate_map_key,
                                 coordinate_manager=data.coordinate_manager, device=data.device)

        return data_Q, likelihood

    def forward(self, pc0, pc1, training=True):
        """
        """
        x0 = pc0
        x1 = pc1
        # encode
        x1_low_list = self.intra_encode(x1)
        x1_low, likelihood = self.get_likelihood(x1_low_list[-1], quantize_mode="noise" if training else "symbols")
        gt_list = [x1] + x1_low_list[:-1]
        gt_list = gt_list[:self.scale]
        nums_list = [len(gt) for gt in gt_list]
        # print('DBG!!! nums_list:\t', nums_list)
        # decode
        if self.mode == 'intra':
            our_set = self.intra_decode(x1_low, nums_list=nums_list, gt_list=gt_list, training=training)
        elif self.mode == 'inter':
            our_set = self.inter_decode(x1_low, x0, nums_list=nums_list, gt_list=gt_list, training=training)
        #
        our_set['likelihood'] = likelihood
        our_set['ground_truth_list'] = gt_list
        for i in range(self.scale):
            if i == 0:
                x0_low = self.pooling(x0)
                x1_low = self.pooling(x1)
            else:
                x0_low = self.pooling(x0_low)
                x1_low = self.pooling(x1_low)
        our_set['x0_low'] = x0_low
        our_set['x1_low'] = x1_low

        return [our_set]

    def pack_bitstream(self, shape, min_v, max_v, strings, dtype='int32'):
        bitstream = np.array(shape, dtype=dtype).tobytes()
        bitstream += np.array(min_v, dtype=dtype).tobytes()
        bitstream += np.array(max_v, dtype=dtype).tobytes()
        bitstream += strings

        return bitstream

    def unpack_bitstream(self, bitstream, dtype='int32'):
        s = 0
        shape = np.frombuffer(bitstream[s:s + 2 * 4], dtype=dtype)
        s += 2 * 4
        min_v = np.frombuffer(bitstream[s:s + 1 * 4], dtype=dtype)
        s += 1 * 4
        max_v = np.frombuffer(bitstream[s:s + 1 * 4], dtype=dtype)
        s += 1 * 4
        strings = bitstream[s:]

        return shape, min_v, max_v, strings

    def encode(self, pc1):
        # downscale pc1
        x1 = pc1
        # encode
        x1_low_list = self.intra_encode(x1)
        latent = x1_low_list[-1]
        latent = sort_sparse_tensor(latent)
        # num_points
        gt_list = [x1] + x1_low_list[:-1]
        gt_list = gt_list[:self.scale]
        nums_list = [len(gt) for gt in gt_list]
        # feats
        feats = latent.F
        strings, min_v, max_v = self.entropy_bottleneck.compress(feats)
        shape = feats.shape
        bitstream = self.pack_bitstream(shape, min_v, max_v, strings)
        # downscale coords
        for i in range(self.scale):
            x1 = self.pooling(x1)
        x1 = ME.SparseTensor(features=torch.ones([x1.F.shape[0], 1]).float(),
                             coordinates=x1.C // x1.tensor_stride[0],
                             tensor_stride=1, device=x1.device)

        return x1, bitstream, nums_list

    def decode(self, pc0, x1, bitstream, nums_list):
        # feats
        shape, min_v, max_v, strings = self.unpack_bitstream(bitstream)
        feats = self.entropy_bottleneck.decompress(strings, min_v, max_v, shape, channels=shape[-1])
        # coords
        assert x1.tensor_stride[0] == 1
        x1 = ME.SparseTensor(features=x1.F, coordinates=x1.C * (2 ** self.scale),
                             tensor_stride=2 ** self.scale, device=x1.device)
        if self.scale in [2, 3]:
            x1 = sort_sparse_tensor(x1)
            assert x1.shape[0] == feats.shape[0]
            x1 = ME.SparseTensor(
                features=feats,
                coordinate_map_key=x1.coordinate_map_key,
                coordinate_manager=x1.coordinate_manager,
                device=x1.device)
        if self.scale == 1:
            x1 = self.pooling(x1)
            assert x1.shape[0] == feats.shape[0]
            index = array2vector(x1.C, x1.C.max() + 1).sort()[1]
            inverse_index = index.sort()[1].cpu()
            x1 = ME.SparseTensor(
                features=feats[inverse_index],
                coordinate_map_key=x1.coordinate_map_key,
                coordinate_manager=x1.coordinate_manager,
                device=x1.device)
        #
        gt_list = [None] * self.scale
        # decode
        if self.mode == 'intra':
            our_set = self.intra_decode(x1, nums_list=nums_list, gt_list=gt_list, training=False)
        elif self.mode == 'inter':
            our_set = self.inter_decode(x1, pc0, nums_list=nums_list, gt_list=gt_list, training=False)

        return our_set['out']

    def test(self, pc0, pc1):
        # encode
        start = time.time()
        x1_low, bitstream, nums_list = self.encode(pc1)
        print('enc time:\t', round(time.time() - start, 3))
        # decode
        start = time.time()
        pc1_dec = self.decode(pc0, x1_low, bitstream, nums_list)
        print('dec time:\t', round(time.time() - start, 3))
        bits = len(bitstream) * 8
        bpp = round(bits / pc1.shape[0], 3)
        print('bpp:\t', bpp, pc1.shape[0])
        print("memoey:\t", round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2), 'GB')
        # check
        out_set = self.forward(pc0, pc1, training=False)[0]
        pc1_dec2 = out_set['out']
        assert (sort_sparse_tensor(pc1_dec).C == sort_sparse_tensor(pc1_dec2).C).all()

        return


if __name__ == '__main__':
    model = PCCModel()
    print(model)