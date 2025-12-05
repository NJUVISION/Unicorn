# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

import os, sys
sys.path.append(os.path.split(__file__)[0])
import argparse


from data_cfg import get_testdata_cfg, get_traindata_cfg
from data_cfg import testdata_set, traindata_set

from train_cfg import get_common_cfg, get_train_cfg
from model_cfg import get_model_geometry_cfg, get_model_attribute_cfg

from test_cfg import get_test_geometry_cfg
from test_cfg import get_test_attribute_cfg
from test_cfg import get_GPCC_cfg


def get_args(component='attribute'):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser = get_common_cfg(parser)
    parser = get_testdata_cfg(parser)
    # train
    parser = get_traindata_cfg(parser)
    parser = get_train_cfg(parser)
    # gpcc
    parser = get_GPCC_cfg(parser)

    if component=='geometry': 
        parser = get_model_geometry_cfg(parser)
        parser = get_test_geometry_cfg(parser)
    if component=='attribute': 
        parser = get_model_attribute_cfg(parser)
        parser = get_test_attribute_cfg(parser)
    # if component=='joint':
    #     parser = get_model_geometry_cfg(parser)
    #     parser = get_test_geometry_cfg(parser)
    #     parser = get_model_attribute_cfg(parser)
    #     parser = get_test_attribute_cfg(parser)

    args = parser.parse_args()

    # dataset
    if args.testdata in testdata_set: args.testdata = testdata_set[args.testdata]
    if args.traindata in traindata_set: args.traindata = traindata_set[args.traindata]
    if args.valdata in traindata_set:  args.valdata = testdata_set[args.valdata]
    
    return args  


if __name__ == '__main__':

    args = get_args()
    print(args)
