
# Test
# ###########################################################################################################################
################################### solid 
python test.py --threshold=0 --kernel_size=3 --block_type='conv' \
    --ckptdir_low='../ckpts/geometry/shapenet/8stage/epoch_last.pth' \
    --testdata='8ivfb' --prefix='8ivfb' 
    or 
    --testdata='owlii' --prefix='owlii' 

################################### dense/sparse (TODO: OOM!!!!!!!!!!!!!!!!!!!!!!!!!!!)
python test.py --threshold=4.2 --kernel_size=3 --block_type='conv' \
    --max_num=400000 \
    --ckptdir_low='../ckpts/geometry/shapenet/8stage/epoch_last.pth' --ckptdir_high='../ckpts/geometry/mpeg/conv/epoch_last.pth' \
    --testdata='mpeg_dense' --prefix='mpeg_dense' 
    or 
    --testdata='mpeg_sparse' --prefix='mpeg_sparse' 

################################### scannet
python test.py --threshold=0 --kernel_size=5 --block_type='conv' \
    --testdata_num=20 --testdata_seqs='random' \
    --ckptdir_low='../ckpts/geometry/scan2cm/8stage/epoch_last.pth' \
    --testdata='scan2cm' --prefix='scan2cm' 

################################### lidar
python test.py --threshold=1.15 --kernel_size=5 --block_type='tf' \
    --testdata_num=100 --testdata_seqs='random' \
    --bitrate_mode=1 \
    --ckptdir_low='../ckpts/geometry/ford/ford2cm/conv/epoch_last.pth' --ckptdir_high='../ckpts/geometry/ford/ford1mm/tf/epoch_last.pth' \
    --testdata='ford1mm' --prefix='ford1mm' 
    or
    --ckptdir_low='../ckpts/geometry/kitti/kitti2cm/conv/epoch_last.pth' --ckptdir_high='../ckpts/geometry/kitti/kitti1mm/tf/epoch_last.pth' \
    --testdata='kitti1mm' --prefix='kitti1mm' 


# Train
###########################################################################################################################

################################## Solid PCs (ShapeNet)
# lossless
python train.py --traindata='shapenet' --testdata='8ivfb' --augment=1 --batch_size=4 --lr=0.0001 --lr_min=0.0001 \
    --stage=8 --kernel_size=3 --scale=4 --block_type='conv' \
    --init_ckpt='../ckpts/geometry/shapenet/8stage/epoch_last.pth' --prefix='shapenet/8stage' 

################################## Dense/Sparse PCGs
# lossless
python train.py  --traindata='mpeg_dense_sparse'  --testdata='mpeg_dense_part' --valdata='mpeg_sparse_part' --batch_size=1 \
    --stage=8 --kernel_size=5 --scale=4 --block_type='conv' \
    --init_ckpt='../ckpts/geometry/mpeg/conv/epoch_last.pth' --prefix='mpeg/8stage' 

################################## LiDAR PCs (KITTI)
# lossless
python train.py --traindata='kitti1mm' --testdata='kitti1mm'  --augment=1 --batch_size=1 \
    --block_type='tf' --stage=8 --kernel_size=5 --scale=5 \
    --init_ckpt='../ckpts/geometry/kitti/kitti1mm/tf/epoch_last.pth' --prefix='kitt1mm/8stage' 

python train.py --traindata='kitti2cm' --testdata='kitti2cm'  --augment=1 \
    --block_type='conv' --stage=8 --kernel_size=5 --scale=5 \
    --init_ckpt='../ckpts/geometry/kitti/kitti2cm/conv/epoch_last.pth'  --prefix='kitt2cm/8stage' 

