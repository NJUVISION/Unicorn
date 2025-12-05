

# test 
# #########################################################################################################################################################################
bash test_data.sh ford 100
bash test_data.sh kitti 100

# train
#########################################################################################################################################################################

python train.py --augment=1 --traindata='ford1mm' --testdata='ford1mm' \
    --inter_mode=1  --scale=5 --block_type='tf' \
    --init_ckpt='../ckpts/dynamic_geometry_lidar/ford1mm/tf_conv/epoch_last.pth' --prefix='ford1mm' --only_test=1

# 
python train.py --augment=1 --traindata='ford2cm' --testdata='ford2cm' \
    --inter_mode=1  --scale=5 --block_type='conv' \
    --init_ckpt='../ckpts/dynamic_geometry_lidar/ford2cm/conv_conv/epoch_last.pth' --prefix='ford2cm' --only_test=1

