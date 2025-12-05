# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-06

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import random
import numpy as np
import torch
import MinkowskiEngine as ME
import pandas as pd
from basic_models.loss import get_bce, get_bits
from data_utils.dataloaders.dynamic_geometry_dataloader import DynamicPCDataset, make_data_loader, get_seqs_list

from model import PCCModel as Model

from cfg.get_args import get_args 
args = get_args(component='geometry')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pipelines.trainer_base import Trainer
class DPCCTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info('\n'.join(['%s:\t%s' % item for item in self.__dict__.items() if item[0]!='model']))
        self.logger.info(args.traindata)
        self.logger.info(args.testdata)
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        self.logger.info('params: '+str(para)+'\tmodel size: '+str(round(para*4/1024))+' KB')
        # self.logger.info(self.model)
        # self.dist_loss_fn = get_chamfer_distance

    def forward(self, data, training):
        coords0, feats0, coords1, feats1 = data
        # coords0 = coords1
        # feats0 = feats1

        if coords1.shape[0] > args.max_num and training==True:
            print('num_points>max_value', coords1.shape[0], args.max_num)
            return torch.autograd.Variable(torch.Tensor([0]), requires_grad=True)
        # data
        x0 = ME.SparseTensor(features=feats0.float(), coordinates=coords0, device=device)
        x1 = ME.SparseTensor(features=feats1.float(), coordinates=coords1, device=device)

        # forward
        out_set_list = self.model(x0, x1, training=training)
        # loss
        loss = 0
        bce_matrix, bpp_list = [], []
        # record
        record_set = {}
        for _, out_set in enumerate(out_set_list):
            bce, bce_list = 0, []
            if 'out_cls_list' in out_set:
                for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                    num_points = float(x1.__len__())
                    curr_bce = get_bce(out_cls, ground_truth)/num_points
                    bce += curr_bce
                    bce_list.append(curr_bce.item())

            if 'likelihood' in out_set: bpp = get_bits(out_set['likelihood'])/float(x1.__len__())
            else: bpp = torch.tensor([0]).to(x0.device)

            curr_loss = args.weight_distortion * bce +args.weight_bitrate * bpp
            loss += curr_loss
            bce_matrix.append(bce_list)
            bpp_list.append(bpp.item())

        with torch.no_grad():
            record_set.update({'loss':np.sum(bce_matrix)+np.sum(bpp_list)})
            if 'out_cls_list' in out_set_list[0]:
                record_set.update({'bce_matrix':bce_matrix})
            if 'likelihood' in out_set_list[0]:
                record_set.update({'bpp_list':bpp_list})
        # collect record
        for k, v in record_set.items():
            if k not in self.record_set: self.record_set[k]=[]
            self.record_set[k].append(v)
        
        return loss


if __name__ == '__main__':

    # model
    model = Model(inter_mode=args.inter_mode, scale=args.scale, block_type=args.block_type).to(device)

    # trainer
    trainer = DPCCTrainer(model=model, lr=args.lr, init_ckpt=args.init_ckpt, rootdir=args.ckptsdir,
                        pretrained_modules=args.pretrained_modules, frozen_modules=args.frozen_modules, 
                        prefix=args.prefix, check_time=args.check_time, device=device)
    
    # data
    seqs_list = get_seqs_list(args.traindata)
    print('seqs length:\t', len(seqs_list))
    val_dataset = DynamicPCDataset(seqs_list[::1+int(len(seqs_list)//args.testdata_num)], voxel_size=args.voxel_size, quant_mode=args.quant_mode)
    val_dataloader = make_data_loader(dataset=val_dataset, batch_size=1, shuffle=False, repeat=False)
    print('val file length:\t', len(val_dataloader))
    test_seqs_list = get_seqs_list(args.testdata)
    if len(test_seqs_list)>args.testdata_num: 
        test_seqs_list = test_seqs_list[:args.testdata_num]
    test_dataset = DynamicPCDataset(test_seqs_list, voxel_size=args.voxel_size, quant_mode=args.quant_mode)
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=1, shuffle=False, repeat=False)
    print('test file length:\t', len(test_dataloader))
    for i, f in enumerate(test_seqs_list):
        print('test_seqs_list', i, f)

    # training
    for epoch in range(0, args.epoch):
        # only test
        if args.only_test:
            trainer.test(test_dataloader, 'Test')
            trainer.test(val_dataloader, 'Validate')
            break
        # training dataset
        curr_filedirs = random.sample(seqs_list[::-1], min(len(seqs_list), args.traindata_num))
        dataset = DynamicPCDataset(curr_filedirs, voxel_size=args.voxel_size, augment=args.augment, quant_mode=args.quant_mode)
        dataloader = make_data_loader(dataset=dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
        print('train file length:\t', len(dataloader))

        for idx, seqs in enumerate(sorted(random.sample(curr_filedirs, min(10, len(curr_filedirs))))):
            print('train:\t', idx, seqs)

        # train
        if epoch>0 and epoch%2==0: args.lr =  max(args.lr/2, args.lr_min)
        if epoch>args.frozen_epoch: trainer.frozen_modules = None            
        trainer.train(dataloader)
        trainer.test(test_dataloader, 'Test')
        trainer.test(val_dataloader, 'Validate')
