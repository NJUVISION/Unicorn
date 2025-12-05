# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-06

import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
sys.path.append('/media/ivc3090ti/新加卷/zjz/unicorn/UnicornV1')
import glob
import torch
from coder import CoderMultiframe

from cfg.get_args import get_args 
args = get_args(component='geometry')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################## cfg for each rate (ckpts, scale) ##############################
ckpts_rootdir = '../ckpts/dynamic_geometry/'

def get_intra_coder(rate):
    ckptdir_lossless0 = ckpts_rootdir+'intra_lossless/epoch_last.pth'
    # lossless
    if rate=='lossless':
        ckptdir_lossy0=''
        ckptdir_sr0='' 
        scale_lossless0, scale_lossy0, scale_sr0 = 6,0,0
    # s1        
    if rate=='lossy_s1_b1':
        ckptdir_lossy0 = ckpts_rootdir+'intra_s1/b1/epoch_last.pth'
        ckptdir_sr0=''
        scale_lossless0, scale_lossy0, scale_sr0 = 5,1,0
    if rate=='lossy_s1_b15':
        ckptdir_lossy0 = ckpts_rootdir+'intra_s1/b15/epoch_last.pth'
        ckptdir_sr0=''
        scale_lossless0, scale_lossy0, scale_sr0 = 5,1,0
    if rate=='lossy_s1_sr':
        ckptdir_lossy0=''
        ckptdir_sr0 = ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 5,0,1
    # s2
    if rate=='lossy_s2_b1':
        ckptdir_lossy0 = ckpts_rootdir+'intra_s1/b1/epoch_last.pth'
        ckptdir_sr0 = ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 4,1,1
    if rate=='lossy_s2_sr':
        ckptdir_lossy0=''
        ckptdir_sr0 = ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 4,0,2

    # s3
    if rate=='lossy_s3_b1':
        ckptdir_lossy0 = ckpts_rootdir+'intra_s1/b1/epoch_last.pth'
        ckptdir_sr0 = ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 3,1,2

    if rate=='lossy_s3_sr':
        ckptdir_lossy0=''
        ckptdir_sr0 = ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 3,0,3

    if rate=='lossy_s4_sr':
        ckptdir_lossy0=''
        ckptdir_sr0 = ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 2,0,4
    if rate=='lossy_s5_sr':
        ckptdir_lossy0=''
        ckptdir_sr0 = ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 1,0,5

    # auxiliary: inter
    ckptdir_lossless1=''
    ckptdir_lossy1=''
    ckptdir_sr1=''
    scale_lossless1, scale_lossy1, scale_sr1 = 6,0,0

    # coder
    coder = CoderMultiframe(mode='intra',
        ckptdir_lossless0=ckptdir_lossless0, ckptdir_lossy0=ckptdir_lossy0, ckptdir_sr0=ckptdir_sr0, 
        scale_lossless0=scale_lossless0, scale_lossy0=scale_lossy0, scale_sr0=scale_sr0,
        ckptdir_lossless1=ckptdir_lossless1, ckptdir_lossy1=ckptdir_lossy1, ckptdir_sr1=ckptdir_sr1, 
        scale_lossless1=scale_lossless1, scale_lossy1=scale_lossy1, scale_sr1=scale_sr1,
        outdir=args.outdir, resultsdir=args.resultsdir)

    return coder


def get_inter_coder(rate):
    # auxiliary: intra
    ckptdir_lossless0 = ckpts_rootdir+'intra_lossless/epoch_last.pth'
    ckptdir_lossless1 = ckpts_rootdir+'inter_lossless/epoch_last.pth'
    if rate=='lossless':
        ckptdir_lossy0=''
        ckptdir_sr0=''
        scale_lossless0, scale_lossy0, scale_sr0 = 6,0,0
    if rate[:8]=='lossy_s1':
        ckptdir_lossy0 = ckpts_rootdir+'intra_s1/b15/epoch_last.pth'
        ckptdir_sr0=''
        scale_lossless0, scale_lossy0, scale_sr0 = 5,1,0
    if rate[:8]=='lossy_s2':
        ckptdir_lossy0 = ckpts_rootdir+'intra_s1/b1/epoch_last.pth'
        ckptdir_sr0=ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 4,1,1
    if rate[:8]=='lossy_s3':
        ckptdir_lossy0=ckpts_rootdir+'intra_s1/b1/epoch_last.pth'
        ckptdir_sr0=ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 4,0,2
    if rate[:8]=='lossy_s4':
        ckptdir_lossy0=ckpts_rootdir+'intra_s1/b1/epoch_last.pth'
        ckptdir_sr0=ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 3,0,3
    if rate[:8]=='lossy_s5':
        ckptdir_lossy0=ckpts_rootdir+'intra_s1/b1/epoch_last.pth'
        ckptdir_sr0=ckpts_rootdir+'intra_sr/epoch_last.pth'
        scale_lossless0, scale_lossy0, scale_sr0 = 2,0,4
    #
    ckptdir_sr1=''
    if rate=='lossless':
        ckptdir_lossy1=''
        scale_lossless1, scale_lossy1, scale_sr1 = 6,0,0
    
    if rate=='lossy_s1_b15':
        scale_lossless1, scale_lossy1, scale_sr1 = 5,1,0
        ckptdir_lossy1=ckpts_rootdir+'inter_s1/b15/epoch_last.pth'
    if rate=='lossy_s1_b3':
        scale_lossless1, scale_lossy1, scale_sr1 = 5,1,0
        ckptdir_lossy1=ckpts_rootdir+'inter_s1/b3/epoch_last.pth'

    if rate=='lossy_s2_b4':
        scale_lossless1, scale_lossy1, scale_sr1 = 4,2,0
        ckptdir_lossy1=ckpts_rootdir+'inter_s2/b4/epoch_last.pth'
    if rate=='lossy_s2_b5':
        scale_lossless1, scale_lossy1, scale_sr1 = 4,2,0
        ckptdir_lossy1=ckpts_rootdir+'inter_s2/b5/epoch_last.pth'

    if rate=='lossy_s2_b1':
        scale_lossless1, scale_lossy1, scale_sr1 = 4,1,1
        ckptdir_lossy1=ckpts_rootdir+'inter_s1/b15/epoch_last.pth'
        ckptdir_sr1=ckpts_rootdir+'inter_sr/epoch_last.pth'

    if rate=='lossy_s2_sr':
        scale_lossless1, scale_lossy1, scale_sr1 = 4,0,2
        ckptdir_lossy1=''
        ckptdir_sr1=ckpts_rootdir+'inter_sr/epoch_last.pth'

    if rate=='lossy_s3_b24':
        scale_lossless1, scale_lossy1, scale_sr1 = 3,2,1
        ckptdir_lossy1=ckpts_rootdir+'inter_s2/b4/epoch_last.pth'
        ckptdir_sr1=ckpts_rootdir+'inter_sr/epoch_last.pth'

    if rate=='lossy_s3_b25':
        scale_lossless1, scale_lossy1, scale_sr1 = 3,2,1
        ckptdir_lossy1=ckpts_rootdir+'inter_s2/b5/epoch_last.pth'
        ckptdir_sr1=ckpts_rootdir+'inter_sr/epoch_last.pth'

    if rate=='lossy_s3_b1':
        scale_lossless1, scale_lossy1, scale_sr1 = 3,1,2
        ckptdir_lossy1=ckpts_rootdir+'inter_s1/b15/epoch_last.pth'
        ckptdir_sr1=ckpts_rootdir+'inter_sr/epoch_last.pth'

    if rate=='lossy_s3_sr':
        scale_lossless1, scale_lossy1, scale_sr1 = 3,0,3
        ckptdir_lossy1=''
        ckptdir_sr1=ckpts_rootdir+'inter_sr/epoch_last.pth'
    if rate=='lossy_s4_sr':
        scale_lossless1, scale_lossy1, scale_sr1 = 2,0,4
        ckptdir_lossy1=''
        ckptdir_sr1=ckpts_rootdir+'inter_sr/epoch_last.pth'

    if rate == 'lossy_s5_sr':
        scale_lossless1, scale_lossy1, scale_sr1 = 1, 0, 5
        ckptdir_lossy1 = ''
        ckptdir_sr1 = ckpts_rootdir + 'inter_sr/epoch_last.pth'
    
    coder = CoderMultiframe(mode='inter',
        ckptdir_lossless0=ckptdir_lossless0, ckptdir_lossy0=ckptdir_lossy0, ckptdir_sr0=ckptdir_sr0, 
        scale_lossless0=scale_lossless0, scale_lossy0=scale_lossy0, scale_sr0=scale_sr0,
        ckptdir_lossless1=ckptdir_lossless1, ckptdir_lossy1=ckptdir_lossy1, ckptdir_sr1=ckptdir_sr1, 
        scale_lossless1=scale_lossless1, scale_lossy1=scale_lossy1, scale_sr1=scale_sr1, 
        outdir=args.outdir, resultsdir=args.resultsdir)

    return coder


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.outdir = os.path.join(args.outdir, args.prefix)
    args.resultsdir = os.path.join(args.resultsdir, args.prefix)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultsdir, exist_ok=True)

    ################# testdata #################
    filedir_list = sorted(glob.glob(os.path.join(args.testdata,'**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    filedir_list = filedir_list[:args.testdata_num]

    print('filedir_list length:\t', len(filedir_list))
    for i, f in enumerate(filedir_list):
        print('filedir_list', i, f)

    #################### test all rates ####################
    coder_list = []
    if not args.inter_mode:
        # for rate in ['lossless', 'lossy_s1_b15', 'lossy_s1_sr', 'lossy_s2_b1', 'lossy_s2_sr', 'lossy_s3_sr']:
        for rate in ['lossy_s3_b1'][::-1]:
            coder = get_intra_coder(rate)
            coder_list.append(coder)
    if args.inter_mode:
        # for rate in ['lossy_s1_b15', 'lossy_s1_b3', 'lossy_s2_b4', 'lossy_s3_sr']:
        for rate in ['lossy_s2_b5', 'lossy_s2_b4']:
        # for rate in ['lossy_s3_sr'][::-1]:
            coder = get_inter_coder(rate)
            coder_list.append(coder)
    
    # seqs_list = sorted(glob.glob(os.path.join(args.dataset, '*.ply')))[:args.frame]

    for idx, coder in enumerate(coder_list):
        # if idx<4: continue
        coder.test(filedir_list, voxel_size=args.voxel_size, quant_mode=args.quant_mode, prefix='r'+str(idx))