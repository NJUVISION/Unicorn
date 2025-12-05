# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-9-21

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import time
import torch
import MinkowskiEngine as ME
from inter_predictor import MultiscaleInterPredictor, target_knn, MotionFeatureExtraction, FeatureFusion





class CompensatorConvSimple(torch.nn.Module):
    """ using convolution on the target coordinates (failed)
    """
    def __init__(self, channels=32, kernel_size=9, block_layers=3, scale=0):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, stride=1, bias=True, dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x0, x1):
        out = self.conv(x0, x1.C)

        out = ME.SparseTensor(
            features=out.F, coordinates=out.C,
            tensor_stride=x0.tensor_stride, device=out.device)

        return out


def create_coords_sparse(x, device):
    tensor_stride = x.tensor_stride
    feats = torch.ones((len(x.C[:,1:]), 1)).float()
    coords, feats = ME.utils.sparse_collate([x.C[:,1:]], [feats])
    coords_sparse = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=tensor_stride, device=device)
    return coords_sparse


class CompensatorConv(torch.nn.Module):
    """ using convolution on the target coordinates (failed)
    """

    def __init__(self, channels=32, kernel_size=9, block_layers=3, scale=0):
        super().__init__()
        # self.inter_pred = MultiscaleInterPredictor(channels)
        self.inter_model = target_knn(channels=channels)
        self.motion_model = MotionFeatureExtraction(channels=channels)
        self.feature_fusion = FeatureFusion(channels=channels, only_fusion=True)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x0, x1):
        stride = x1.tensor_stride
 
        pred_y1 = self.inter_model(x0, x1)
        pred_y2 = self.motion_model(x0, x1, stride)
        previous_y = self.feature_fusion(pred_y1, pred_y2)
        pred_out = ME.SparseTensor(features=previous_y.F,
                                   coordinate_map_key=x1.coordinate_map_key,
                                   coordinate_manager=x1.coordinate_manager, device=x1.device)

        return pred_out
