# 1. sample points from mesh 
# Here we take shapenet as an example. We densely sample each mesh into a point cloud with the number of points of 1000000. 
# The we radnomly quantize them with the resolution of 256.
python geometry_dataset.py --process='mesh2pc' \
    --input_rootdir='../../../dataset/shared/ShapeNet/mesh/ShapeNet10k/' \
    --output_rootdir='./dataset/shapenet/' \
    --input_format='obj' --output_format='ply' \
    --resolution=256 --num_points=1000000 \
    --output_length=10

# 2. paritition points into blocks with the max number of points of 100000 using kdtree.
python geometry_dataset.py --process='partition' \
    --input_rootdir='./dataset/shapenet/' \
    --output_rootdir='./dataset/shapenet_100k/' \
    --input_format='ply' --output_format='ply' \
    --num_points=100000 \
    --output_length=10

# 3. quantize points clouds with a fixed precision.
# here we take KITTI as the example, we quantize the raw point clouds into 1mm.
python geometry_dataset.py --process='quantize' \
    --input_rootdir='../../../dataset/shared/KITTI/sequences0110/' \
    --output_rootdir='./dataset/kitti1mm/' \
    --input_format='bin' --output_format='ply' \
    --precision=0.001 \
    --output_length=10



################## point clouds with attribute #################
# 1. paritition points into several parts using kdtree.
# here we take longdress as the example and partition each point cloud into 8 parts.

python attribute_dataset.py --process='partition' \
    --input_rootdir='../../../dataset/shared/dynamic_object/8iVFB_seqs/longdress/'  \
    --output_rootdir='./dataset/longdress' \
    --input_format='ply' --output_format='ply' \
    --output_length=10 --n_parts=8

# 2. quantize points into several parts using kdtree.
# here we take ford1mm as an example and quantize them into 2cm.
python attribute_dataset.py --process='quantize' \
    --input_rootdir='../../../dataset/shared/Ford/Ford_q1mm/' \
    --output_rootdir='./dataset/ford2cmm/' \
    --input_format='ply' --output_format='ply' \
    --precision=20 --output_length=10