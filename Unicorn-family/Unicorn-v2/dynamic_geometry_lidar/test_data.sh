# bash test.sh lossy ford 100
# bash test.sh lossy kitti 100
###############################################################################

testdata=$1; # kitti, ford
testdata_num=$2; # 110 30, 100
ckpt_rootdir='../ckpts/'

if [ $testdata == 'ford' ]; then
    for dataname in '02' '03'
    do
    python test.py --inter_mode=1 \
        --kernel_size=5 --block_type='conv' \
        --testdata_num=${testdata_num} \
        --testdata='ford1mm_'${dataname} \
        --prefix='ford1mm_'${dataname} \
        --ckptdir_low=${ckpt_rootdir}'dynamic_geometry_lidar/ford2cm/conv_conv/epoch_last.pth' \
        --ckptdir_high=${ckpt_rootdir}'dynamic_geometry_lidar/ford1mm/tf_conv/epoch_last.pth' \
        --ckptdir_offset=${ckpt_rootdir}'geometry/ford/ford1mm/offset/epoch_last.pth' --offset=1 \
        --start_index=1 --interval=10   
    done

elif [  $testdata  ==  'kitti'  ]; then
    for dataname in '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21'
    do
    python test.py --inter_mode=1 \
        --kernel_size=5 --block_type='conv' \
        --testdata_num=${testdata_num} \
        --testdata='kitti1mm_'${dataname} \
        --prefix='kitti1mm_'${dataname} \
        --ckptdir_low=${ckpt_rootdir}'dynamic_geometry_lidar/kitti2cm/conv_conv/epoch_last.pth' \
        --ckptdir_high=${ckpt_rootdir}'dynamic_geometry_lidar/kitti1mm/tf_conv/epoch_last.pth' \
        --ckptdir_offset=${ckpt_rootdir}'geometry/kitti/kitti1mm/offset/epoch_last.pth' --offset=1 \
        --start_index=1 --interval=10 
    done
fi

