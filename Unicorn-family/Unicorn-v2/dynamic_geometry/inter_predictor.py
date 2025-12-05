import torch
import MinkowskiEngine as ME
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import math
import pytorch3d.ops
from target_knn_module import TPM
from knn_selfattention import Point_Transformer_Last as KNN
from data_utils.sparse_tensor import isin, sort_sparse_tensor
from pytorch3d.ops import knn_points, knn_gather

class ResNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


def make_layer(block, block_layers, channels):
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))

    return torch.nn.Sequential(*layers)


class InceptionResNet(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_2 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0_0(x))
        out0 = self.relu(self.conv0_2(self.relu(self.conv0_1(out))))
        out1 = self.relu(self.conv1_1(self.relu(self.conv1_0(x))))
        out = ME.cat(out0, out1)
        return out + x



class Backbone(torch.nn.Module):
    def __init__(self, in_channels, channels, out_channels, kernel_size=3):
        super(Backbone, self).__init__()
        self.conv_in = ME.MinkowskiConvolution(in_channels=in_channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               dilation=1,
                                               bias=True,
                                               dimension=3)

        self.conv0_0 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               dilation=1,
                                               bias=True,
                                               dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               dilation=1,
                                               bias=True,
                                               dimension=3)
        self.IRN0 = InceptionResNet(channels=channels, kernel_size=kernel_size)


        self.conv_out = ME.MinkowskiConvolution(in_channels=channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=1,
                                                dilation=1,
                                                bias=True,
                                                dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.IRN0(self.conv0_1(self.relu(self.conv0_0(out))))
        out = self.conv_out(out)
        return out


class Static_Backbone(torch.nn.Module):
    def __init__(self, in_channels, channels, out_channels, kernel_size, k):
        super(Static_Backbone, self).__init__()
        self.conv_in = ME.MinkowskiConvolution(in_channels=in_channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               bias=True,
                                               dimension=3)

        self.conv0_0 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               bias=True,
                                               dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               bias=True,
                                               dimension=3)
        self.IRN0 = InceptionResNet(channels=channels, kernel_size=kernel_size)
        self.knn0 = KNN(block=1, channels=channels,head=1, k=k)


        self.conv1_0 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               bias=True,
                                               dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               bias=True,
                                               dimension=3)
        self.IRN1 = InceptionResNet(channels=channels, kernel_size=kernel_size)
        self.knn1 = KNN(block=1, channels=channels,head=1, k=k)

        self.conv2_0 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               bias=True,
                                               dimension=3)
        self.conv2_1 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               bias=True,
                                               dimension=3)
        self.IRN2 = InceptionResNet(channels=channels, kernel_size=kernel_size)
        self.knn2 = KNN(block=1, channels=channels,head=1, k=k)

        self.conv_out = ME.MinkowskiConvolution(in_channels=channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=1,
                                                bias=True,
                                                dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.IRN0(self.conv0_1(self.relu(self.knn0(self.conv0_0(out)))))
        out = self.IRN1(self.conv1_1(self.relu(self.knn1(self.conv1_0(out)))))
        out = self.IRN2(self.conv2_1(self.relu(self.knn2(self.conv2_0(out)))))
        out = self.conv_out(out)
        return out





class Fusion(torch.nn.Module):
    def __init__(self, channels, out_channels, kernel_size):
        super(Fusion, self).__init__()

        self.conv0_0 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               dilation=1,
                                               bias=True,
                                               dimension=3)

        self.conv0_1 = ME.MinkowskiConvolution(in_channels=out_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               dilation=1,
                                               bias=True,
                                               dimension=3)



        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0_1(self.relu(self.conv0_0(x)))
        return out



class Classify(torch.nn.Module):
    def __init__(self, channels, out_channels, kernel_size):
        super(Classify, self).__init__()

        self.conv0_0 = ME.MinkowskiConvolution(in_channels=channels,
                                               out_channels=channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               bias=True,
                                               dimension=3)
        # self.conv0_1 = ME.MinkowskiConvolution(in_channels=channels,
        #                                        out_channels=channels,
        #                                        kernel_size=kernel_size,
        #                                        stride=1,
        #                                        bias=True,
        #                                        dimension=3)

        # self.conv1_0 = ME.MinkowskiConvolution(in_channels=channels,
        #                                        out_channels=channels,
        #                                        kernel_size=kernel_size,
        #                                        stride=1,
        #                                        bias=True,
        #                                        dimension=3)
        # self.conv1_1 = ME.MinkowskiConvolution(in_channels=channels,
        #                                        out_channels=channels,
        #                                        kernel_size=kernel_size,
        #                                        stride=1,
        #                                        bias=True,
        #                                        dimension=3)
        self.IRN0 = InceptionResNet(channels=channels, kernel_size=kernel_size)
        self.conv_out = ME.MinkowskiConvolution(in_channels=channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=1,
                                                dilation=1,
                                                bias=True,
                                                dimension=3)


        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.IRN0(self.relu(self.conv0_0(x)))
        # out = self.conv1_1(self.relu(self.conv1_0(out)))
        out = self.conv_out(out)
        return out


class MultiscaleInterPredictor(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.inter_predictor_new = TPM(in_channels=channels, channels=channels)

        self.extract_curr = Static_Backbone(in_channels=1, channels=channels, out_channels=channels, kernel_size=3, k=8)

        self.out_block = Backbone(in_channels=channels * 2, channels=channels, out_channels=channels)

        self.merge_resnet = Backbone(in_channels=2, channels=channels, out_channels=channels)

        self.pruning = ME.MinkowskiPruning()

    def merge_two_frames(self, f1, f2):
        stride = f1.tensor_stride[0]
        f1_ = ME.SparseTensor(torch.cat([f1.F, torch.zeros_like(f1.F)], dim=-1), coordinates=f1.C,
                              tensor_stride=stride, device=f1.device)
        f2_ = ME.SparseTensor(torch.cat([torch.zeros_like(f2.F), f2.F], dim=-1), coordinates=f2.C,
                              tensor_stride=stride, coordinate_manager=f1_.coordinate_manager, device=
                              f1.device)

        merged_f = f1_ + f2_

        merged_f = ME.SparseTensor(merged_f.F, coordinates=merged_f.C, tensor_stride=stride,
                                   device=merged_f.device)

        return merged_f

    def Merge_extract(self, x_ref, x_curr):
        merged_x = self.merge_two_frames(x_curr, x_ref)
        merge_feature = self.merge_resnet(merged_x)
        mask = isin(merge_feature.C, x_curr.C)
        merge_feature = self.pruning(merge_feature, mask)
        merge_feature = sort_sparse_tensor(merge_feature, x_curr)
        return merge_feature

    def forward(self, ref, curr_x):
        stride_size = curr_x.tensor_stride
        ref_f = self.extract_curr(ref)
        merge_feature = self.Merge_extract(ref, curr_x)
        pred_knn = self.inter_predictor_new(ref_f, curr_x)

        assert (pred_knn.C == merge_feature.C).all()
        pred = ME.SparseTensor(features=torch.cat([pred_knn.F, merge_feature.F], dim=1),
                               coordinate_map_key=curr_x.coordinate_map_key,
                               coordinate_manager=curr_x.coordinate_manager, device=curr_x.device)

        pred = self.out_block(pred)

        return pred


class self_attention(nn.Module):
    def __init__(self, channels):
        super(self_attention, self).__init__()
        self.q_conv = torch.nn.Linear(channels, channels)
        self.k_conv = torch.nn.Linear(channels, channels)
        self.v_conv = torch.nn.Linear(channels, channels)
        self.d = math.sqrt(channels)

    def forward(self, x, xyz_enc, new_feature):
        x_q = x.F
        # print(x_q.shape)

        Q = self.q_conv(x_q)
        new_feature = new_feature + xyz_enc
        K = self.k_conv(new_feature)
        K = K.permute(0, 2, 1)
        attention_map = torch.einsum('ndk,nd->nk', K, Q)
        # print(attention_map.shape)
        attention_map = F.softmax(attention_map / self.d, dim=-1)
        # print(attention_map)
        V = self.v_conv(new_feature)
        attention_feature = torch.einsum('nk,nkd->nd', attention_map, V)
        x = ME.SparseTensor(features=attention_feature, coordinate_map_key=x.coordinate_map_key,
                            coordinate_manager=x.coordinate_manager)
        return x


class target_knn(nn.Module):
    def __init__(self, channels, k=16):
        super(target_knn, self).__init__()
        self.SA0 = self_attention(channels)
        self.SA1 = self_attention(channels)
        self.k = k
        self.Linear = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.pos0 = nn.Sequential(
            nn.Linear(3, channels)
        )
        self.pos1 = nn.Sequential(
            nn.Linear(3, channels)
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.bn0 = ME.MinkowskiBatchNorm(channels)
        self.bn1 = ME.MinkowskiBatchNorm(channels)

    def forward(self, x_reference, x_lossy):
        x_reference_C = x_reference.C.unsqueeze(0).float()
        x_reference_F = x_reference.F.unsqueeze(0).float()

        x_lossy_C = x_lossy.C.unsqueeze(0).float()
        dist, idx, _ = knn_points(x_lossy_C, x_reference_C, K=self.k, return_nn=False, return_sorted=True)
        # dist, idx, _ = knn_points(x_lossy_C, x_reference_C, K=self.k)
        x_lossy_xyz = x_lossy_C.squeeze(0)[:, 1:]

        x_reference_neibor = knn_gather(x_reference_C[:, :, 1:], idx).squeeze(0)
        x_reference_neibor = x_lossy_xyz[:, None, :] - x_reference_neibor[:, :, :]
        x_reference_neibor_pos_emb = self.pos0(x_reference_neibor)

        # knn
        new_feature = knn_gather(x_reference_F, idx).squeeze(0)
        out = self.SA0(x_lossy, x_reference_neibor_pos_emb, new_feature)  # self-attention
        x_reference_neibor_pos_emb_1 = self.pos1(x_reference_neibor)
        out = self.SA1(out, x_reference_neibor_pos_emb_1, new_feature)  # self-attention

        # skip-connettion
        out = self.bn0(x_lossy + out)
        out_1 = self.Linear(out)
        out = self.bn1(out_1 + out)

        return out


class MotionFeatureExtraction(torch.nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels * 2,
            out_channels=channels * 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

    def merge_two_frames(self, f1, f2, stride):
        f1_ = ME.SparseTensor(torch.cat([f1.F, torch.zeros_like(f1.F)], dim=-1), coordinates=f1.C,
                              tensor_stride=stride, device=f1.device)
        f2_ = ME.SparseTensor(torch.cat([torch.zeros_like(f2.F), f2.F], dim=-1), coordinates=f2.C,
                              tensor_stride=stride, coordinate_manager=f1_.coordinate_manager, device=
                              f1.device)

        merged_f = f1_ + f2_

        merged_f = ME.SparseTensor(merged_f.F, coordinates=merged_f.C, tensor_stride=stride,
                                   device=merged_f.device)

        return merged_f

    def forward(self, ref, pred, stride):
        merged_f = self.merge_two_frames(ref, pred, stride)
        out = self.relu(self.conv0(merged_f))
        out = self.conv1(out)
        mask = isin(out.C, pred.C)
        pred_out = self.pruning(out, mask)
        pred_out = self.sort(pred_out, pred.C)
        pred_out = ME.SparseTensor(features=pred_out.F,
                                   coordinate_map_key=pred.coordinate_map_key,
                                   coordinate_manager=pred.coordinate_manager, device=pred.device)

        return pred_out


class FeatureFusion(torch.nn.Module):
    def __init__(self, channels=128, inchannels=128, midchannels=128, only_fusion=False):
        super().__init__()
        self.only_fusion = only_fusion
        if not self.only_fusion:
            self.conv0 = ME.MinkowskiConvolution(
                in_channels=inchannels,
                out_channels=midchannels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)
            self.conv1 = ME.MinkowskiConvolution(
                in_channels=midchannels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, x_gpcc):
        out = x
        out_g = x_gpcc
        if not self.only_fusion:
            out_g = self.relu(self.conv1(self.conv0(out_g)))

        out_F = torch.cat((out.F, out_g.F), dim=1)
        out = ME.SparseTensor(features=out_F, coordinate_map_key=out.coordinate_map_key,
                              coordinate_manager=out.coordinate_manager)

        out = self.conv2(out)

        return out
