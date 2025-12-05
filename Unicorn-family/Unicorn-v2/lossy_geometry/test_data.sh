testdata=$1; # ford, kitti
testdata_num=$2; # 110 30


################################# set model #################################
ckpts_rootdir='../ckpts/geometry' 

if [  $testdata  ==  '8ivfb'  ] || [  $testdata  ==  'owlii'  ] || [  $testdata  ==  'model'  ] || [  $testdata  ==  'mpeg_dense'  ] || [  $testdata  ==  'mpeg_sparse'  ]; then
    # SparsePCGC
    # ckptdir_low=${ckpts_rootdir}'/SparsePCGC/shapenet_k3/8stage/epoch_last.pth' 
    # ckptdir_sr_low=${ckpts_rootdir}'/SparsePCGC/shapenet_k3/1stage/epoch_last.pth' 
    # ckptdir_ae_low=${ckpts_rootdir}'/SparsePCGC/shapenet_k3/slne/epoch_last.pth' 
    # retrain 
    ckptdir_low=${ckpts_rootdir}'/shapenet/8stage/epoch_last.pth' 
    ckptdir_sr_low=${ckpts_rootdir}'/shapenet/sr/epoch_last.pth' 
    ckptdir_ae_low=${ckpts_rootdir}'/shapenet/ae_s1/w1/epoch_last.pth' 
    # RWTT
    # ckptdir_low=${ckpts_rootdir}'/RWTT/8stage/epoch_last.pth' 
    # ckptdir_sr_low=${ckpts_rootdir}'/RWTT/sr/epoch_last.pth' 
    # ckptdir_ae_low=${ckpts_rootdir}'/RWTT/ae_s1/w1/epoch_last.pth' 

elif [  $testdata  ==  'scan2cm'  ]; then
    ckptdir_low=${ckpts_rootdir}'/scan2cm/8stage/epoch_last.pth' 
    ckptdir_sr_low=${ckpts_rootdir}'/scan2cm/sr/epoch_last.pth' 
    ckptdir_ae_low=${ckpts_rootdir}'/scan2cm/ae_s1/w1/epoch_last.pth' 
elif [  $testdata  ==  'kitti1mm'  ]; then
    ckptdir_low=${ckpts_rootdir}'/kitti/kitti2cm/conv/epoch_last.pth'
    ckptdir_sr_low='' 
    ckptdir_ae_low=''
elif [  $testdata  ==  'ford1mm'  ]; then
    ckptdir_low=${ckpts_rootdir}'/ford/ford2cm/conv/epoch_last.pth' 
    ckptdir_sr_low='' 
    ckptdir_ae_low=''
fi


if [  $testdata  ==  'mpeg_dense'  ] || [  $testdata  ==  'mpeg_sparse'  ]; then
    # ckptdir_high=${ckpts_rootdir}'/mpeg/8stage/epoch_last.pth'    
    ckptdir_high=${ckpts_rootdir}'/mpeg/conv/epoch_last.pth'    
    ckptdir_sr_high=${ckpts_rootdir}'/mpeg/sr/epoch_last.pth'     
    ckptdir_ae_high=${ckpts_rootdir}'/mpeg/ae_s1/w1/epoch_last.pth'
    # ckptdir_offset=${ckpts_rootdir}'mpeg/offset/epoch_last.pth' 
    ckptdir_offset=''
elif [  $testdata  ==  'kitti1mm'  ]; then
    ckptdir_high=${ckpts_rootdir}'/kitti/kitti1mm/tf/epoch_last.pth'
    ckptdir_sr_high='' 
    ckptdir_ae_high=''
    ckptdir_offset=${ckpts_rootdir}'/kitti/kitti1mm/offset/epoch_last.pth'
elif [  $testdata  ==  'ford1mm'  ]; then
    ckptdir_high=${ckpts_rootdir}'/ford/ford1mm/tf/epoch_last.pth'
    ckptdir_sr_high='' 
    ckptdir_ae_high=''
    ckptdir_offset=${ckpts_rootdir}'/ford/ford1mm/offset/epoch_last.pth'
else
    ckptdir_high='' 
    ckptdir_sr_high='' 
    ckptdir_ae_high=''
    ckptdir_offset=''
fi


################################# set parameters #################################
if [  $testdata  ==  '8ivfb'  ] || [  $testdata  ==  'owlii'  ] || [  $testdata  ==  'model'  ]; then
    # model parameters
    kernel_size=3
    block_type='conv'
    # lossy modes
    threshold=0
    threshold_lossy=0
    bitrate_mode=0
    # test setip
    testdata_seqs='random'
    resolution=1023
    max_num=1200000
    start_index=0
    interval=1


elif [  $testdata  ==  'mpeg_dense'  ]; then
    # model parameters
    kernel_size=3
    block_type='conv'
    # lossy modes
    threshold=4.2
    threshold_lossy=4.2
    bitrate_mode=0 #!!!!!!!!!!!!!!!!!!!!!!!!!
    # test setip
    testdata_seqs='frame'
    resolution=4095
    max_num=400000
    start_index=0
    interval=1

elif [  $testdata  ==  'mpeg_sparse'  ]; then
    # model parameters
    kernel_size=3
    block_type='conv'
    # lossy modes
    threshold=4.2
    threshold_lossy=4.2
    bitrate_mode=1
    # test setip
    testdata_seqs='frame'
    resolution=4095
    max_num=400000
    start_index=0
    interval=1

elif [  $testdata  ==  'scan2cm'  ];  then
    # model parameters
    kernel_size=5
    block_type='conv'
    # lossy modes
    threshold=0
    threshold_lossy=0
    bitrate_mode=0
    # test setip
    testdata_seqs='random'
    resolution=1023
    max_num=1200000
    start_index=0
    interval=1

elif  [  $testdata  ==  'kitti1mm'  ] || [  $testdata  ==  'ford1mm'  ]; then
    # model parameters
    kernel_size=5
    block_type='tf'
    # lossy modes
    threshold=1.15
    threshold_lossy=0
    bitrate_mode=2
    # test setip
    testdata_seqs='random'
    resolution=30000
    max_num=120000
    start_index=1 
    interval=10 
fi


#################################
python test.py \
    --testdata=${testdata} \
    --testdata_num=${testdata_num} \
    --testdata_seqs=${testdata_seqs} \
    --max_num=${max_num} \
    --start_index=${start_index} \
    --interval=${interval} \
    --resolution=${resolution} \
    --threshold=${threshold} \
    --threshold_lossy=${threshold_lossy} \
    --kernel_size=${kernel_size} \
    --block_type=${block_type} \
    --ckptdir_low=${ckptdir_low} \
    --ckptdir_sr_low=${ckptdir_sr_low} \
    --ckptdir_ae_low=${ckptdir_ae_low} \
    --ckptdir_high=${ckptdir_high} \
    --ckptdir_sr_high=${ckptdir_sr_high} \
    --ckptdir_ae_high=${ckptdir_ae_high} \
    --ckptdir_offset=${ckptdir_offset} \
    --bitrate_mode=${bitrate_mode} \
    --prefix=${testdata} 


