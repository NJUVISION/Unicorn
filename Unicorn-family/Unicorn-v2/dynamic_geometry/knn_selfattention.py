import torch
import MinkowskiEngine as ME
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import math
import pytorch3d.ops

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


class Point_Transformer_Last(torch.nn.Module):
    def __init__(self, block=2, channels=128, head=1, k=16):
        super(Point_Transformer_Last, self).__init__()
        self.head = head
        self.k = k
        self.layers = torch.nn.ModuleList()
        for i in range(block):
            self.layers.append(Transformer_block(channels, head, k))

    def forward(self, x):
        out = x
        x_C = out.C.unsqueeze(0).float()
        dist, idx, _ = pytorch3d.ops.knn_points(x_C, x_C, K=self.k)
        knn_xyz =  pytorch3d.ops.knn_gather(x_C[:,:,1:], idx)
        center_xyz = x_C[:, :, 1:].unsqueeze(2)

        knn_xyz_norm = knn_xyz - center_xyz
        knn_xyz_norm = knn_xyz_norm.squeeze(0)
        knn_xyz_norm = knn_xyz_norm / knn_xyz_norm.max()

        for transformer in self.layers:
            out_F = out.F.unsqueeze(0).float()
            knn_feature = pytorch3d.ops.knn_gather(out_F[:,:,:], idx).squeeze(0)
            out = transformer(x, knn_feature, knn_xyz_norm)

        return out
