# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

import os, sys
rootdir = os.path.split(os.path.split(__file__)[0])[0]


def get_common_cfg(parser):
    parser.add_argument("--DBG", type=int, default=0., help="")
    # parser.add_argument("--flag",  type=int, default=0., help="")
    parser.add_argument("--outdir", default='output')
    parser.add_argument("--resultsdir", default='results')
    parser.add_argument("--ckptsdir", default='ckpts')
    parser.add_argument("--prefix", type=str, default='tp', help="")

    return parser


def get_train_cfg(parser):
    ####################### dataset cfg #######################
    parser.add_argument("--augment", type=int, default=1, help="enable dataset augmentation in training")
    parser.add_argument("--batch_size", type=int, default=1)
    
    ###### geometry specified parameters
    parser.add_argument("--voxel_size", type=float, default=1)

    ####### attribute specified parameters
    parser.add_argument("--color_format", type=str, default='yuv')
    parser.add_argument("--normalize",  type=int, default=1, help="normalize rgb to [0,1] or not?")

    ####################### training cfg #######################
    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--init_ckpt_inter", type=str, default='')# only for dynamic PCAC
    parser.add_argument('--pretrained_modules', nargs='+')
    parser.add_argument('--frozen_modules', nargs='+')
    parser.add_argument("--frozen_epoch", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-4, help='1e-4 for PCGC, 4e-5 for PCAC')
    parser.add_argument("--lr_min", type=float, default=1e-4, help='1e-4 for PCGC, 1e-5 for PCAC')

    parser.add_argument("--clip_grad", type=int, default=0, help="clip_gradients during training?")
    parser.add_argument("--clip_value", type=float, default=100., help="clip_value.")
    parser.add_argument("--weight_decay", type=float, default=0, help="0 for PCGC, 0.0001 for PCAC")

    parser.add_argument("--limit_loss", type=int, default=1, help="limit max loss to avoid training crashes. (only for attribute)")
    parser.add_argument("--max_bpp", type=float, default=100, help="max_bpp.")
    parser.add_argument("--max_mse", type=float, default=10, help="max_mse.")

    parser.add_argument("--train_time", type=float, default=60, help='max training time (hours)')
    parser.add_argument("--save_time", type=int, default=4, help='frequency for saving checkpoints (epoch)')
    parser.add_argument("--check_time", type=float, default=10, help='frequency for testing (min).')

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--only_test", type=int, default=0, help="only test or not.")

    ####################### loss function cfg #######################
    parser.add_argument("--weight_distortion", type=float, default=1., help="weights for distortion like MSE.")
    parser.add_argument("--weight_distortion_min", type=float, default=5000., help="for variable-rate coding")
    parser.add_argument("--weight_bitrate", type=float, default=1., help="weights for bit rate.")
    parser.add_argument("--last_weight", type=float, default=0, help="weight of last scale (only for attribute)")

    return parser

