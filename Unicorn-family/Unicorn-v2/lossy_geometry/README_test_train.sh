
# Test
# ###########################################################################################################################
# ############### optimal settings ###############
bash test_data.sh 8ivfb 4
bash test_data.sh owlii 4
bash test_data.sh mpeg_sparse 3
bash test_data.sh mpeg_dense 2

bash test_data.sh scan2cm 20

bash test_data.sh kitti1mm 100
bash test_data.sh ford1mm 100

# Train
###########################################################################################################################

################################## Solid PCs (ShapeNet)
# lossless
python train.py --traindata='shapenet' --testdata='8ivfb' --augment=1 --batch_size=4 --lr=0.0001 --lr_min=0.0001 \
    --model='lossless' --stage=8 --kernel_size=3 --scale=4 --block_type='conv' \
    --init_ckpt='../ckpts/geometry/shapenet/8stage/epoch_last.pth' --prefix='shapenet/8stage' 

################################## Dense/Sparse PCGs
# lossless
python train.py  --traindata='mpeg_dense_sparse'  --testdata='mpeg_dense_part' --valdata='mpeg_sparse_part' --batch_size=1 \
    --model='lossless' --stage=8 --kernel_size=5 --scale=4 --block_type='conv' \
    --init_ckpt='../ckpts/geometry/mpeg/conv/epoch_last.pth' --prefix='mpeg/8stage' 

# lossy_sr: --stage=1
python train.py  --traindata='mpeg_dense_sparse'  --testdata='mpeg_dense_part' --valdata='mpeg_sparse_part' --batch_size=1 \
    --model='base' --stage=1 --kernel_size=5 --scale=4 --block_type='conv' \
    --init_ckpt='../ckpts/geometry/mpeg/sr/epoch_last.pth' --prefix='mpeg/sr' 

# lossy_ae: --stage=1 --enc_type='ae'
python train.py --traindata='mpeg_dense_sparse'  --testdata='mpeg_dense_part' --valdata='mpeg_sparse_part' --batch_size=1 \
    --model='base' --enc_type='ae' --kernel_size=5 --scale=4 --block_type='conv' \
    --weight_bitrate=1 --weight_distortion=1 \
    --init_ckpt='../ckpts/geometry/mpeg/ae_s1/w1/epoch_last.pth' --prefix='mpeg/ae_s1/w1' 


################################## LiDAR PCs (KITTI)
# lossless
python train.py --traindata='kitti1mm' --testdata='kitti1mm'  --augment=1 --batch_size=1 \
    --model='lossless' --block_type='tf' --stage=8 --kernel_size=5 --scale=5 \
    --init_ckpt='../ckpts/geometry/kitti/kitti1mm/tf/epoch_last.pth' --prefix='mpeg/kitt1mm/8stage' 

python train.py --traindata='kitti2cm' --testdata='kitti2cm'  --augment=1 \
    --model='lossless' --block_type='conv' --stage=8 --kernel_size=5 --scale=5 \
    --init_ckpt='../ckpts/geometry/kitti/kitti2cm/conv/epoch_last.pth'  --prefix='mpeg/kitt2cm/8stage' 

# lossy_offset
python train_offset.py --traindata='ford1mm' --testdata='ford1mm' --batch_size=4 --voxel_size=1 --augment=1 --channels=64 \
    --posQuantscaleList 16 32 64 128 256 \
    --init_ckpt='../ckpts/geometry/kitti/kitti1mm/offset/epoch_last.pth'  --prefix='mpeg/kitti1mm/offset' 

