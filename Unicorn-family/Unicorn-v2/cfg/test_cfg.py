# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

import argparse

def get_test_geometry_cfg(parser):
    ####################### dataset cfg #######################
    parser.add_argument("--max_num", type=float, default=2000000, help='max number of points to avoid OOM')
    parser.add_argument("--resolution", type=int, default=1023, help='geometry resolution, 1023 for 8iVFB, 30000 for Ford')

    ####################### bitrates cfg #######################
    parser.add_argument("--bitrate_mode", type=float, default=0, help='-1: lossless, 0: solid PCs, 1: sparse PCs, 2: LiDAR PCs')
    parser.add_argument("--posQuantscale", type=float, default=1)
    # threshold for density (or fractal dimension) adaptive model selection
    parser.add_argument("--threshold", type=float, default=2, help='lossless coding')
    parser.add_argument("--threshold_lossy", type=float, default=3.2, help='lossy coding')
    parser.add_argument("--offset", type=int, default=0, help='enable offset or not')

    ####################### ckptdir cfg #######################
    parser.add_argument("--ckptdir",type=str, default='')
    parser.add_argument("--ckptdir_ae",type=str, default='')
    parser.add_argument("--ckptdir_sr",type=str, default='')
    parser.add_argument("--ckptdir_offset",type=str, default='')

    parser.add_argument("--ckptdir_low",type=str, default='')
    parser.add_argument("--ckptdir_ae_low",type=str, default='')
    parser.add_argument("--ckptdir_sr_low",type=str, default='')

    parser.add_argument("--ckptdir_high",type=str, default='')
    parser.add_argument("--ckptdir_ae_high",type=str, default='')
    parser.add_argument("--ckptdir_sr_high",type=str, default='')

    ####################### adaption study
    parser.add_argument("--only_global_topk", type=int, default=0)

    return parser


def get_test_attribute_cfg(parser):
    ####################### dataset cfg #######################
    try:
        parser.add_argument("--color_format", type=str, default='yuv for lossy, ycocg for lossless')
        parser.add_argument("--normalize",  type=int, default=1, help="normalize rgb to [0,1] or not?")
    except (argparse.ArgumentError) as e: print('conflicting option string')
    ####################### bitrates cfg #######################
    parser.add_argument("--num_bitrates", type=int, default=4)#
    parser.add_argument("--piecewise_variable_bitrates", type=str, default='')
    ####################### ckptdir cfg #######################
    parser.add_argument("--ckptdir", default='', help="lossless")
    parser.add_argument("--ckptdir_list", type=str, nargs='+', help="lossy")
    parser.add_argument("--ckptdir_intra", default='')
    parser.add_argument("--ckptdir_inter", default='')

    return parser


def get_GPCC_cfg(parser):
    parser.add_argument("--gpcc_version", type=int, default=21)
    # geometry cfg
    parser.add_argument("--posQuantscaleList", type=float, nargs='+', help="geometry")
    parser.add_argument("--gpcc_cfg", type=str, default='dense.cfg')
    # attribute cfg
    parser.add_argument("--gpcc_qp", type=int, default=22)
    parser.add_argument("--transformType", type=int, default=0, help="0: RAHT, 1: predicting, 2: lifting")

    return parser
