import torch
import MinkowskiEngine as ME
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import math
import pytorch3d.ops


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



class SA_Layer(torch.nn.Module):
    def __init__(self, channels, head=1, k=16):
        super(SA_Layer, self).__init__()
        self.channels = channels
        self.q_conv = torch.nn.Linear(channels, channels)
        self.k_conv = torch.nn.Linear(channels + 3, channels)
        self.v_conv = torch.nn.Linear(channels + 3, channels)
        self.d = math.sqrt(channels)
        self.head = head
        self.k = k

    def forward(self, x, knn_feature, knn_xyz):
        x_q = x.F

        new_knn_feature = torch.cat((knn_feature, knn_xyz), dim=2)

        Q = self.q_conv(x_q).view(-1, self.head, self.channels // self.head)
        K = self.k_conv(new_knn_feature).view(-1, self.head, self.k, self.channels // self.head)
        attention_map = torch.einsum('nhd,nhkd->nhk', Q, K)
        attention_map = torch.nn.functional.softmax(attention_map / self.d, dim=-1)
        # print(attention_map)

        V = self.v_conv(new_knn_feature).view(-1, self.head, self.k, self.channels // self.head)
        attention_feature = torch.einsum('nhk,nhkd->nhd', attention_map, V)
        attention_feature = attention_feature.view(-1, self.channels)

        new_x = ME.SparseTensor(features=attention_feature, coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager)

        return new_x



class Transformer_block(torch.nn.Module):
    def __init__(self, channels, head, k):
        super(Transformer_block, self).__init__()

        self.layer_norm_1 = torch.nn.LayerNorm(channels)
        self.linear = torch.nn.Linear(channels, channels)
        self.layer_norm_2 = torch.nn.LayerNorm(channels)

        self.sa = SA_Layer(channels, head, k)

    def forward(self, x, knn_feature, knn_xyz):
        x1 = x + self.sa(x, knn_feature, knn_xyz)
        x1_F = x1.F

        x1_F = self.layer_norm_1(x1_F)
        x1_F = x1_F + self.linear(x1_F)
        x1_F = self.layer_norm_2(x1_F)

        x1 = ME.SparseTensor(features=x1_F, coordinate_map_key=x1.coordinate_map_key,
                             coordinate_manager=x1.coordinate_manager)

        return x1


class Target_Point_Transformer_Last(torch.nn.Module):
    def __init__(self, block=2, channels=128, head=1, k=16):
        super(Target_Point_Transformer_Last, self).__init__()
        self.head = head
        self.k = k
        self.layers = torch.nn.ModuleList()
        # for i in range(block):
        #     self.layers.append(Transformer_block(channels, head, k))
        self.transformer_block = Transformer_block(channels, head, k)

    def forward(self, x, target_x):
        out = x
        x_C = out.C.unsqueeze(0).float()
        out_F = out.F.unsqueeze(0).float()
        target_x_C = target_x.C.unsqueeze(0).float()

        dist, idx, _ = pytorch3d.ops.knn_points(target_x_C, x_C, K=self.k)
        knn_xyz = pytorch3d.ops.knn_gather(x_C, idx)[:, :, :, 1:]
        center_xyz = target_x_C[:, :, 1:].unsqueeze(2)

        knn_xyz_norm = knn_xyz - center_xyz
        knn_xyz_norm = knn_xyz_norm.squeeze(0)
        knn_xyz_norm = knn_xyz_norm / knn_xyz_norm.max()

        init_feature = pytorch3d.ops.knn_gather(out_F, idx).squeeze(0)
        init_feature = torch.mean(init_feature, 1)

        out = ME.SparseTensor(
            features=init_feature,
            coordinate_map_key=target_x.coordinate_map_key,
            coordinate_manager=target_x.coordinate_manager)

        # print(out.shape, idx.shape)
        knn_feature = pytorch3d.ops.knn_gather(out_F, idx).squeeze(0)
        out = self.transformer_block(out, knn_feature, knn_xyz_norm)

        # out_F = out.F.unsqueeze(0).float()
        # print(out.shape, idx.shape)
        #
        # knn_feature = pytorch3d.ops.knn_gather(out_F, idx).squeeze(0)
        # out = self.transformer_block(out, knn_feature, knn_xyz_norm)

        return out


class TPM(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32):
        super(TPM, self).__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.interp = ME.MinkowskiConvolution(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=9,
                                              stride=1,
                                              dilation=1,
                                              bias=True,
                                              dimension=3)

        self.conv0 = ME.MinkowskiConvolution(in_channels=in_channels * 2,
                                             out_channels=channels,
                                             kernel_size=3,
                                             stride=1,
                                             dilation=1,
                                             bias=True,
                                             dimension=3)

        self.conv1 = ME.MinkowskiConvolution(in_channels=channels,
                                             out_channels=channels,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

        self.block = InceptionResNet(channels=channels, kernel_size=3)

        self.transformer = Target_Point_Transformer_Last(block=1, channels=in_channels, head=1, k=16)

    def forward(self, previous, current):
        context = self.interp(previous, current.C)

        # interpolation_feature = knn_interpolation(previous, current, k=1)
        interpolation_feature = self.transformer(previous, current)

        context = ME.SparseTensor(
            features=torch.cat((context.F, interpolation_feature.F), dim=-1),
            coordinate_map_key=current.coordinate_map_key,
            coordinate_manager=current.coordinate_manager,
            device=current.device)

        out = self.conv1(self.relu(self.block(self.relu(self.conv0(context)))))

        return out
