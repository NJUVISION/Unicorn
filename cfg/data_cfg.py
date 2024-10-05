
def get_traindata_cfg(parser):
    parser.add_argument("--traindata", type=str, default='', help='training dataset name, see dataset_cfg.py')
    parser.add_argument("--traindata_num", type=int, default=int(1000))
    parser.add_argument("--valdata", default='')
    parser.add_argument("--valdata_num", type=int, default=int(10))

    return parser

def get_testdata_cfg(parser):
    parser.add_argument("--testdata", type=str, default='')
    parser.add_argument("--testdata_num", type=int, default=int(10))
    parser.add_argument("--max_testdata_num", type=int, default=int(100))
    parser.add_argument("--testdata_seqs", default='random')# random or frame
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--filedir", type=str, default='')

    return parser