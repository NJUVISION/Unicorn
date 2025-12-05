# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-07

import os, sys, time
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import numpy as np
from collections import Counter
from data_utils.sparse_tensor import isin


bce_fn = torch.nn.BCEWithLogitsLoss()
ce_fn = torch.nn.CrossEntropyLoss()
softmax_fn = torch.nn.Softmax(dim=-1)


def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    if data.shape[-1]==8:
        assert groud_truth.F.shape[-1]==8
        bce = bce_fn(data.F, groud_truth.F)
    elif data.F.shape[-1]==1:
        assert groud_truth.F.shape[-1]==1
        if len(data)==len(groud_truth):
            bce = bce_fn(data.F.squeeze(), groud_truth.F.squeeze())
        else:
            mask = isin(data.C, groud_truth.C)
            bce = bce_fn(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0] * groud_truth.shape[1]
    
    return sum_bce


def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits


############################################ metric: shannon entropy
def get_entropy2(error):
    bits = 0
    for idx_ch in range(error.shape[-1]):
        bits += get_entropy(error[:,idx_ch])
    return bits

def get_entropy(error):
    # normalization
    # print("=========== entropy ===========")
    data = error.reshape(-1)
    data = data.astype('int')
    keys = np.sort(np.unique(data))
    dataN = data.copy()
    for i, k in enumerate(keys):
        dataN[data==k] = i
    # data = dataN.copy()

    statistic = Counter(dataN)
    freq_table = {}
    for _, k in enumerate(sorted(statistic)):
        freq_table[k]=statistic[k]/sum(statistic.values())
    pmf = np.array([p for p in freq_table.values()])
    pmf = pmf.astype('float32').round(8)

    probs = pmf[dataN]
    bits = -np.log2(np.array(probs))
    bits = bits.reshape(error.shape)

    return bits.sum()