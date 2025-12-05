# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-03

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import torch
import MinkowskiEngine as ME
import numpy as np
import os, glob, random
from data_utils.dataloaders.geometry_dataloader import PCDataset, make_data_loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model_offset import OffsetModel


from cfg.get_args import get_args 
args = get_args(component='geometry')



from pipelines.trainer_base import Trainer
class OffsetTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MSELoss = torch.nn.MSELoss().to(device)
        self.logger.info('\n'.join(['%s:\t%s' % item for item in self.__dict__.items() if item[0]!='model']))
        self.logger.info(args.traindata)
        self.logger.info(args.testdata)
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        self.logger.info('params: '+str(para)+'\tmodel size: '+str(round(para*4/1024))+' KB')
        # self.logger.info(self.model)

    def forward(self, data, training):
        coords, feats = data
        # data
        x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
        # if (x.shape[0] < args.min_points or x.shape[0] > args.max_points) and training==True:
        #     self.logger.info('num_points:\t' + str(x.shape[0]))
        #     return torch.autograd.Variable(torch.Tensor([0]), requires_grad=True)
        # forward
        out_set_list = self.model(x)
        loss = 0
        mse_list = []
        for _, out_set in enumerate(out_set_list):
            mse = self.MSELoss(out_set['out'].F, out_set['ground_truth'].F).to(device)
            loss += mse
            mse_list.append(mse.item())
        # record
        record_set = {}
        with torch.no_grad():
            record_set.update({'loss':loss.item(), 'mse_list':mse_list})
        # collect record
        for k, v in record_set.items():
            if k not in self.record_set: self.record_set[k]=[]
            self.record_set[k].append(v)

        return loss


if __name__ == '__main__':

    # model
    model = OffsetModel(channels=args.channels, kernel_size=args.kernel_size, block_layers=args.block_layers, 
                        posQuantscaleList=args.posQuantscaleList).to(device)
    
    if args.DBG:  print('model:\n', model)

    # trainer
    trainer = OffsetTrainer(model=model, lr=args.lr, init_ckpt=args.init_ckpt, rootdir=args.ckptsdir,
                        pretrained_modules=args.pretrained_modules, frozen_modules=args.frozen_modules, 
                        prefix=args.prefix, check_time=args.check_time, device=device, 
                        cfg=args)

    # data
    all_filedirs = sorted(glob.glob(os.path.join(args.traindata,'**', f'*.*'), recursive=True))
    all_filedirs = [f for f in all_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]

    print('file length:\t', len(all_filedirs))
    # val dataset
    val_filedirs = all_filedirs[::len(all_filedirs)//args.testdata_num]
    val_dataset = PCDataset(val_filedirs, voxel_size=args.voxel_size)
    val_dataloader = make_data_loader(dataset=val_dataset, batch_size=1, shuffle=False, repeat=False)
    print('validate file length:\t', len(val_filedirs))
    # test dataset
    test_filedirs = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.*'), recursive=True))
    test_filedirs = [f for f in test_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    if len(test_filedirs)>args.testdata_num: test_filedirs = test_filedirs[::len(test_filedirs)//args.testdata_num]
    test_dataset = PCDataset(test_filedirs, voxel_size=args.voxel_size)
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=1, shuffle=False, repeat=False)
    print('test file length:\t', len(test_filedirs))
    for i, f in enumerate(test_filedirs):
        print('test_filedirs', i, f)

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

        train_dataset = PCDataset(filedirs, voxel_size=args.voxel_size, augment=args.augment)
        train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, 
                                            shuffle=True, repeat=False)
        if epoch>0  and epoch%2==0:
            args.lr =  max(args.lr/2, args.lr_min)
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
        trainer.test(val_dataloader, 'Val')