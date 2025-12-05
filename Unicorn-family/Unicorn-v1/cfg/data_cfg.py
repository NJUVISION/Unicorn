# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-12-3

def get_traindata_cfg(parser):
    parser.add_argument("--traindata", type=str, default='', help='training dataset name, see dataset_cfg.py')
    parser.add_argument("--traindata_num", type=int, default=int(1000))
    parser.add_argument("--valdata", default='')
    parser.add_argument("--valdata_num", type=int, default=int(10))

    return parser

def get_testdata_cfg(parser):
    parser.add_argument("--testdata", type=str, default='')
    parser.add_argument("--testdata_num", type=int, default=int(1))
    parser.add_argument("--max_testdata_num", type=int, default=int(100))
    parser.add_argument("--testdata_seqs", default='frame')# random or frame
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--filedir", type=str, default='')
    parser.add_argument("--partition", action="store_true", help="partition or not.")
    parser.add_argument("--part_nums", type=int, default=500000)

    return parser



data_rootdir = '/media/ivc3090ti/新加卷/zjz/MPEG_CFP/datasets/dynamic/model_vox10.ply+xyz+n+rgb'

lossy_rootdir = '/home/temp/wjq/unicorn/outdata/recolor/'

########################################################### 
testdata_set = {
# solid_vox10
'8ivfb': data_rootdir+'object/8iVFB/8iVFB/',
'8ivfb_part': data_rootdir+'object/8iVFB/8iVFB_part2/',
'queen': data_rootdir+'object/8iVFB/queen',
'owlii': data_rootdir+'object/Owlii/Owlii_vox10_floor',
'owlii_vox11': data_rootdir+'object/Owlii/Owlii',
'RWTT': data_rootdir+'object/RWTT_frame20/',
'mvub': data_rootdir+'object/MVUB/vox10/',
'model': data_rootdir + '/',
# dense/sparse vox12
'mpeg_dense': data_rootdir+'object/MPEG_dense_sparse/testdata/dense/',
'mpeg_dense_part': data_rootdir+'object/MPEG_dense_sparse/testdata_part_100k/dense/',
'mpeg_dense_part100k': data_rootdir+'object/MPEG_dense_sparse/testdata_part_100k/dense/',
'mpeg_dense_part300k': data_rootdir+'object/MPEG_dense_sparse/testdata_part_300k/dense/',
'mpeg_dense_part400k': data_rootdir+'object/MPEG_dense_sparse/testdata_part_400k/dense/',
'House': data_rootdir+'object/MPEG_dense_sparse/testdata_part_400k/dense/House',
'Facade': data_rootdir+'object/MPEG_dense_sparse/testdata_part_400k/dense/Facade',
'mpeg_sparse': data_rootdir+'object/MPEG_dense_sparse/testdata/sparse/',
'mpeg_sparse_part': data_rootdir+'object/MPEG_dense_sparse/testdata_part_100k/sparse/',
'mpeg_sparse_part100k': data_rootdir+'object/MPEG_dense_sparse/testdata_part_100k/sparse/',
'mpeg_sparse_part300k': data_rootdir+'object/MPEG_dense_sparse/testdata_part_300k/sparse/',
'mpeg_sparse_part400k': data_rootdir+'object/MPEG_dense_sparse/testdata_part_400k/sparse/',
'Arco': data_rootdir+'object/MPEG_dense_sparse/testdata_part_400k/sparse/Arco',
'Shiva': data_rootdir+'object/MPEG_dense_sparse/testdata_part_400k/sparse/Shiva',
'Staue': data_rootdir+'object/MPEG_dense_sparse/testdata_part_400k/sparse/Staue',
# scannet
'scan2cm':  data_rootdir+'ScanNet/scans_test_q2cm/',
'scan5cm':  data_rootdir+'ScanNet/scans_test_q5cm/',
'scan_vox9_val': data_rootdir+ 'ScanNet/scans_vox9_val/', # 3DAC datasets

# static 
'static_kitti1mm': data_rootdir+'paper/testdata_sparsepcgc/KITTI_q1mm/',
'static_kitti2cm': data_rootdir+'paper/testdata_sparsepcgc/KITTI_q2cm/',
'static_ford1mm': data_rootdir+'paper/testdata_sparsepcgc/Ford23_q1mm/',
'static_ford2cm': data_rootdir+'paper/testdata_sparsepcgc/Ford23_q1mm/',

# Sequences 100
# ford1mm
'ford1mm': data_rootdir+'Ford/Ford_q1mm_frame100/',
'ford1mm_02': data_rootdir+'Ford/Ford_q1mm_frame100/Ford_02_q_1mm_frame100/',
'ford1mm_03': data_rootdir+'Ford/Ford_q1mm_frame100/Ford_03_q_1mm_frame100/',


'ford1mm_part32': data_rootdir+'Ford/Ford_q1mm_frame100_part32/',
'ford1mm_part64': data_rootdir+'Ford/Ford_q1mm_frame100_part64/',


'ford_raw_all': data_rootdir+ 'Ford/Ford_raw/Ford_023/',

'ford_raw': data_rootdir+ 'Ford/Ford_raw/Ford_023_frame100/',

# ford2cm
'ford2cm_02': data_rootdir+'Ford/Ford_q2cm/Ford_02_q_1mm_frame100/',
'ford2cm_03': data_rootdir+'Ford/Ford_q2cm/Ford_03_q_1mm_frame100/',
# kitti1mm

'kitti_raw':data_rootdir+'KITTI/sequences1121_frame100/',

'kitti1mm':data_rootdir+'KITTI/sequences1121_frame100_q1mm/',
'kitti1mm_11':data_rootdir+'KITTI/sequences1121_frame100_q1mm//11/',
'kitti1mm_12':data_rootdir+'KITTI/sequences1121_frame100_q1mm//12/',
'kitti1mm_13':data_rootdir+'KITTI/sequences1121_frame100_q1mm//13/',
'kitti1mm_14':data_rootdir+'KITTI/sequences1121_frame100_q1mm//14/',
'kitti1mm_15':data_rootdir+'KITTI/sequences1121_frame100_q1mm//15/',
'kitti1mm_16':data_rootdir+'KITTI/sequences1121_frame100_q1mm//16/',
'kitti1mm_17':data_rootdir+'KITTI/sequences1121_frame100_q1mm//17/',
'kitti1mm_18':data_rootdir+'KITTI/sequences1121_frame100_q1mm//18/',
'kitti1mm_19':data_rootdir+'KITTI/sequences1121_frame100_q1mm//19/',
'kitti1mm_20':data_rootdir+'KITTI/sequences1121_frame100_q1mm//20/',
'kitti1mm_21':data_rootdir+'KITTI/sequences1121_frame100_q1mm//21/',
# kitti2cm
'kitti2cm':data_rootdir+'KITTI/sequences1121_frame100_q2cm/',
'kitti2cm_11':data_rootdir+'KITTI/sequences1121_frame100_q2cm//11/',
'kitti2cm_12':data_rootdir+'KITTI/sequences1121_frame100_q2cm//12/',
'kitti2cm_13':data_rootdir+'KITTI/sequences1121_frame100_q2cm//13/',
'kitti2cm_14':data_rootdir+'KITTI/sequences1121_frame100_q2cm//14/',
'kitti2cm_15':data_rootdir+'KITTI/sequences1121_frame100_q2cm//15/',
'kitti2cm_16':data_rootdir+'KITTI/sequences1121_frame100_q2cm//16/',
'kitti2cm_17':data_rootdir+'KITTI/sequences1121_frame100_q2cm//17/',
'kitti2cm_18':data_rootdir+'KITTI/sequences1121_frame100_q2cm//18/',
'kitti2cm_19':data_rootdir+'KITTI/sequences1121_frame100_q2cm//19/',
'kitti2cm_20':data_rootdir+'KITTI/sequences1121_frame100_q2cm//20/',
'kitti2cm_21':data_rootdir+'KITTI/sequences1121_frame100_q2cm//21/',


# dynamic object
'dynamic_8ivfb': data_rootdir+ 'dynamic_object/8iVFB_seqs/',
'dynamic_longdress': data_rootdir+ 'dynamic_object/8iVFB_seqs/longdress/',
'dynamic_redandblack': data_rootdir+ 'dynamic_object/8iVFB_seqs/redandblack/',
'dynamic_loot': data_rootdir+ 'dynamic_object/8iVFB_seqs/loot/',
'dynamic_soldier': data_rootdir+ 'dynamic_object/8iVFB_seqs/soldier/',
'dynamic_soldier_part': data_rootdir+ 'dynamic_object/8iVFB_part4/soldier/',
#
'dynamic_owlii': data_rootdir+ 'dynamic_object/Owlii_vox10_floor/', 
'dynamic_ball': data_rootdir+ 'dynamic_object/Owlii_vox10_floor/basketball_player_vox11/', 
'dynamic_ball_part': data_rootdir+ 'dynamic_object/Owlii_vox10_floor_part4/basketball_player_vox11/', 
'dynamic_dancer': data_rootdir+ 'dynamic_object/Owlii_vox10_floor/dancer_vox11/', 
'dynamic_model': data_rootdir+ 'dynamic_object/Owlii_vox10_floor/model_vox11/', 
'dynamic_exercise': data_rootdir+ 'dynamic_object/Owlii_vox10_floor/exercise_vox11/', 
#
'dynamic_owlii_vox11': data_rootdir+ 'dynamic_object/Owlii_vox11_frame100/', 
'dynamic_ball_vox11': data_rootdir+ 'dynamic_object/Owlii_vox11_frame100/basketball_player_vox11/', 
'dynamic_dancer_vox11':data_rootdir+ 'dynamic_object/Owlii_vox11_frame100/dancer_vox11/', 
'dynamic_model_vox11': data_rootdir+ 'dynamic_object/Owlii_vox11_frame100/model_vox11/', 
'dynamic_exersice_vox11':data_rootdir+ 'dynamic_object/Owlii_vox11_frame100/exersice_vox11/',

# lossyGlossyA
'lossy_longdress': lossy_rootdir + '8ivfb/longdress/',
'lossy_loot': lossy_rootdir + '8ivfb/loot/',
'lossy_redandblack': lossy_rootdir + '8ivfb/redandblack/',
'lossy_soldier': lossy_rootdir + '8ivfb/soldier/',

'lossy_ball': lossy_rootdir + 'owlii/basketball/',
'lossy_dancer': lossy_rootdir + 'owlii/dancer/',
'lossy_model': lossy_rootdir + 'owlii/model/',
'lossy_exercise': lossy_rootdir + 'owlii/exercise/',
}


########################################################### 
traindata_set = {
# Object
'shapenet': data_rootdir + 'ShapeNet/pc_vox8_color_n100k_ply/', # TODO: 'h5' --> 'ply'
'RWTT': data_rootdir + 'RWTT_frame562/vox10_n50k/', # 
'mpeg_dense_sparse': data_rootdir + 'object/MPEG_dense_sparse/traindata_part3/',

# dynamic Object
'8iVFB': data_rootdir + 'dynamic_object/8iVFB_part8/', 
'8iVFB_Owlii': data_rootdir +'dynamic_object/8iVFB_Owlii_hybrid/', 

'dynamic_8ivfb': data_rootdir+ 'dynamic_object/8iVFB_seqs/',

# scannet
# 'scan2cm':  data_rootdir+ 'ScanNet/scans_q2cm_n50k/', 
'scan2cm': '/home/temp/wjq/dataset/failed/ScanNet/scans_q2cm_n50k/', 
'scan2cm_small': '/home/temp/wjq/dataset/failed/ScanNet/scans_q2cm_n30k/', 
'scan5cm': data_rootdir+ 'ScanNet/scans_q5cm/', 


'scan_vox9_train': data_rootdir+ 'ScanNet/scans_vox9_train/', 

# LiDAR
'ford1mm': data_rootdir+ 'Ford/Ford_q1mm_part8/',
'ford1mm_part16': data_rootdir+ 'Ford/Ford_q1mm_part16/',
'ford1mm_part32': data_rootdir+ 'Ford/Ford_q1mm_part32/',
'ford1mm_part64': data_rootdir+ 'Ford/Ford_q1mm_part64/',
'ford2cm': data_rootdir+ 'Ford/Ford_q2cm_part8/',

'ford1mm_all': data_rootdir+ 'Ford/Ford_q1mm/Ford_01_q_1mm/',

'ford2cm_all': data_rootdir+ 'Ford/Ford_q2cm/Ford_01_q_1mm/',

'ford_raw': data_rootdir+ 'Ford/Ford_raw/Ford_01/',

'kitti_raw':data_rootdir+'KITTI/sequences0110/',

'kitti1mm': data_rootdir+ 'KITTI/sequences0110_q1mm_part16/',
'kitti1mm_part16': data_rootdir+ 'KITTI/sequences0110_q1mm_part16/',
'kitti1mm_part8': data_rootdir+ 'KITTI/sequences0110_q1mm_part8/',
'kitti2cm': data_rootdir+ 'KITTI/sequences0110_q2cm_part8/',
}
