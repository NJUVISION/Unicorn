# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

def get_model_geometry_cfg(parser):
    # model cfg
    parser.add_argument("--model", type=str, default='lossless')
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--stage", type=int, default=8)
    # network parameters
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--block_type", type=str, default='conv')
    parser.add_argument("--block_layers", type=int, default=3)
    parser.add_argument("--enc_type", type=str, default='pooling')
    parser.add_argument("--quant_mode", type=str, default='round')# data
    # dynamic PCGC
    parser.add_argument("--inter_mode", type=int, default=0)
    parser.add_argument("--inter_scale", type=int, default=3)

    return parser
    

def get_model_attribute_cfg(parser):
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--scale", type=int, default=5, help="# 5 for lossy, 8 for lossless")
    parser.add_argument("--stage", type=int, default=1)
    # network parameters
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--block_layers", type=int, default=2, help="2 for lossy, 3 for lossless")
    parser.add_argument("--block_type", type=str, default='resnet')
    #
    parser.add_argument("--Vmode", type=int, default=1, help="variable-rate coding")
    parser.add_argument("--skip_mode", type=int, default=0, help="skip coding")
    # lossless coding
    parser.add_argument("--split_group", type=int, default=1)
    parser.add_argument("--split_channel", type=int, default=0)

    #################################### dynamic PCAC
    parser.add_argument("--inter_mode", type=int, default=0)

    return parser
