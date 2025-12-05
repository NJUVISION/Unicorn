# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-03

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
import os, glob, random
from basic_models.loss import get_bce, get_bits
from data_utils.dataloaders.geometry_dataloader import PCDataset, make_data_loader

from model import PCCModel

from cfg.get_args import get_args 
args = get_args(component='geometry')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



from pipelines.trainer_base import Trainer
class PCCTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info(args.traindata)
        self.logger.info(args.testdata)
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        self.logger.info('params: '+str(para)+'\tmodel size: '+str(round(para*4/1024))+' KB')

    def forward(self, data, training):
        coords, feats = data
        # data
        x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
        # forward
        out_set_list = self.model(x, training)
        loss = 0
        bce_matrix, bpp_list = [], []
        num_points_list = []
        max_points_list = []
        for _, out_set in enumerate(out_set_list):
            # current scale
            bce, bce_list = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                # current stage
                if 'likelihood' in out_set and args.model=='AE': num_points = float(out_cls.__len__())
                else: num_points = float(x.__len__())
                curr_bce = get_bce(out_cls, ground_truth)/num_points
                bce += curr_bce 
                bce_list.append(curr_bce.item())
            
            if 'likelihood' in out_set: 
                bpp = get_bits(out_set['likelihood'])/float(x.__len__())
            else: 
                bpp = torch.tensor([0]).to(bce.device)
            if args.DBG: print('DBG!!! weight_distortion weight_bitrate', args.weight_distortion, args.weight_bitrate)
            curr_loss = args.weight_distortion * bce +args.weight_bitrate * bpp
            loss += curr_loss
            bce_matrix.append(bce_list)
            bpp_list.append(bpp.item())
            num_points_list.append(len(ground_truth))
            max_points_list.append(ground_truth.C.max().cpu().numpy() - ground_truth.C.min().cpu().numpy())

            torch.cuda.empty_cache()

        # record
        record_set = {}
        with torch.no_grad():
            record_set.update({'loss':np.sum(bce_matrix)+np.sum(bpp_list)})
            record_set.update({'bce_matrix':bce_matrix})
            record_set.update({'bpp_list':bpp_list})
            record_set.update({'num_points_list':num_points_list})
            record_set.update({'max_points_list':max_points_list})
        # collect record
        for k, v in record_set.items():
            if k not in self.record_set: self.record_set[k]=[]
            self.record_set[k].append(v)

        return loss


if __name__ == '__main__':

    # model
    model = PCCModel(channels=args.channels, kernel_size=args.kernel_size, block_layers=args.block_layers,
                    stage=args.stage, scale=args.scale,  enc_type=args.enc_type, block_type=args.block_type).to(device)
    if args.DBG:  print('model:\n', model)
    
    # trainer
    trainer = PCCTrainer(model=model, lr=args.lr, init_ckpt=args.init_ckpt, rootdir=args.ckptsdir,
                        pretrained_modules=args.pretrained_modules, frozen_modules=args.frozen_modules, 
                        prefix=args.prefix, check_time=args.check_time, device=device, 
                        cfg=args)
    # data
    all_filedirs = sorted(glob.glob(os.path.join(args.traindata,'**', f'*.*'), recursive=True))
    all_filedirs = [f for f in all_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    print('file length:\t', len(all_filedirs))

    # validate dataset
    if args.valdata=='': 
        val_filedirs = all_filedirs[::1+len(all_filedirs)//args.testdata_num]
    else:
        val_filedirs = sorted(glob.glob(os.path.join(args.valdata,'**', f'*.*'), recursive=True))
        val_filedirs = [f for f in val_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
        val_filedirs = val_filedirs[::1+int(len(val_filedirs)//args.valdata_num)]
    for idx, seqs in enumerate(sorted(random.sample(val_filedirs, min(10, len(val_filedirs))))):
        print('val:\t', idx, seqs)
    val_dataset = PCDataset(val_filedirs, voxel_size=args.voxel_size)
    val_dataloader = make_data_loader(dataset=val_dataset, batch_size=1, shuffle=False, repeat=False)
    print('validate file length:\t', len(val_filedirs))

    # test dataset
    if args.testdata=='': 
        test_filedirs = val_filedirs
    else:
        test_filedirs = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.*'), recursive=True))
        test_filedirs = [f for f in test_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
        test_filedirs = test_filedirs[::1+int(len(test_filedirs)//args.testdata_num)]
    for idx, seqs in enumerate(sorted(random.sample(test_filedirs, min(10, len(test_filedirs))))):
        print('test:\t', idx, seqs)
    print('test file length:\t', len(test_filedirs))
    test_dataset = PCDataset(test_filedirs, voxel_size=args.voxel_size)
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=1, shuffle=False, repeat=False)
    
    # training
    for epoch in range(0, args.epoch):
        # only test
        if args.only_test: 
            trainer.test(test_dataloader, 'Test')
            trainer.test(val_dataloader, 'Val')
            break
        # training dataset
        filedirs = random.sample(all_filedirs, min(len(all_filedirs), args.traindata_num))
        print('file length:\t', len(filedirs))

        for idx, f in enumerate(sorted(random.sample(filedirs, min(10, len(filedirs))))):
            print('train:\t', idx, f) 

        train_dataset = PCDataset(filedirs, voxel_size=args.voxel_size, augment=args.augment, max_num=args.max_num)
        train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, 
                                            shuffle=True, repeat=False)
        # train
        if epoch>0 and epoch%2==0: 
            args.lr = max(args.lr/2, args.lr_min)
        
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
        trainer.test(val_dataloader, 'Val')
