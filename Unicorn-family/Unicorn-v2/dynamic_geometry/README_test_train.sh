# test
# ####################################################################################################################################
# inter
python test.py --inter_mode=1 --testdata='dynamic_ball' --testdata_num=100 --prefix='inter/dynamic_ball'
python test.py --inter_mode=1 --testdata='dynamic_dancer' --testdata_num=100 --prefix='inter/dynamic_dancer'
python test.py --inter_mode=1 --testdata='dynamic_model' --testdata_num=100 --prefix='inter/dynamic_dancer'
python test.py --inter_mode=1 --testdata='dynamic_exersice' --testdata_num=100 --prefix='inter/dynamic_exersice'
# intra
python test.py --inter_mode=0 --testdata='dynamic_ball' --testdata_num=100 --prefix='intra/dynamic_ball'
python test.py --inter_mode=0 --testdata='dynamic_dancer' --testdata_num=100 --prefix='intra/dynamic_dancer'
python test.py --inter_mode=0 --testdata='dynamic_model' --testdata_num=100 --prefix='intra/dynamic_model'
python test.py --inter_mode=0 --testdata='dynamic_exersice' --testdata_num=100 --prefix='intra/dynamic_exersice'


python test.py --inter_mode=1 --testdata='/media/ivc3090ti/disk1/zjz/MPEG_CFP/datasets/dynamic/testing_datasets/exercise_vox10.ply+xyz+n+rgb' --testdata_num=30 --prefix='inter_geo/exercise_312'
# train
# ####################################################################################################################################
# intra
python train.py --model='lossless' --inter_mode=0 --stage=8 --inter_scale=6 --traindata='8iVFB' --testdata='dynamic_owlii' --augment=1 --init_ckpt='../ckpts/dynamic_geometry/intra_lossless/epoch_last.pth' --weight_distortion=1 --weight_bitrate=1  --prefix='intra_lossless' --only_test=0
python train.py --model='lossless' --inter_mode=0 --stage=1 --inter_scale=6 --traindata='8iVFB' --testdata='dynamic_owlii' --augment=1 --init_ckpt='../ckpts/dynamic_geometry/intra_sr/epoch_last.pth' --weight_distortion=1 --weight_bitrate=1 --prefix='intra_sr' --only_test=1
python train.py --model='lossy' --inter_mode=0  --stage=1 --inter_scale=1 --traindata='8iVFB' --testdata='dynamic_owlii' --augment=1 --init_ckpt='../ckpts/dynamic_geometry/intra_s1/b1/epoch_last.pth' --weight_distortion=1 --weight_bitrate=1 --prefix='intra_s1' --only_test=1
# inter 
python train.py --model='lossless' --inter_mode=1 --stage=8 --inter_scale=6 --traindata='8iVFB' --testdata='dynamic_owlii' --augment=1 --init_ckpt='../ckpts/dynamic_geometry/inter_lossless/epoch_last.pth' --weight_distortion=1 --weight_bitrate=1  --prefix='inter_lossless' --only_test=1
python train.py --model='lossy' --inter_mode=1 --stage=1 --inter_scale=1 --traindata='8iVFB' --testdata='dynamic_owlii' --augment=1 --init_ckpt='../ckpts/dynamic_geometry/inter_s1/b15/epoch_last.pth' --weight_distortion=1 --weight_bitrate=1.5  --prefix='inter_s1/b15' --only_test=1
python train.py --model='lossy' --inter_mode=1 --stage=1 --inter_scale=2 --traindata='8iVFB' --testdata='dynamic_owlii' --augment=1 --init_ckpt='../ckpts/dynamic_geometry/inter_s2/b4/epoch_last.pth' --weight_distortion=1 --weight_bitrate=1  --prefix='inter_s2/b4' --only_test=1
python train.py --model='lossless' --inter_mode=1 --stage=1 --inter_scale=6 --traindata='8iVFB' --testdata='dynamic_owlii' --augment=1 --init_ckpt='../ckpts/dynamic_geometry/inter_sr/epoch_last.pth' --weight_distortion=1 --weight_bitrate=1 --prefix='inter_sr' --only_test=1
