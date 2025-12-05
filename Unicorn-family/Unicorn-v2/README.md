# AI-PCC CfP Response Proposal

## Abstract

This document presents proposal in response to the Call for Proposals (CfP) for AI-based Point Cloud Coding. The method proposed in this document is a learning-based solution that is capable of compressing the geometry of the input point cloud. It supports static as well as dynamic point cloud input of diverse characteristics in lossy or lossless modes. This response proposal corresponds to the point cloud coding to address track 1 (geometry-only) and track 2 (geometry+attribute) of the CfP.

For more information, please visit our homepage: https://njuvision.github.io/Unicorn/ and [MPEG](https://dms.mpeg.expert/).

## News

* 2025.12.05 **Open source [Unicorn version 1](https://github.com/NJUVISION/Unicorn/tree/main/Unicorn-family/Unicorn-v1) and [version 2](https://github.com/NJUVISION/Unicorn/tree/main/Unicorn-family/Unicorn-v2)!*** 2025.12.04 Unicorn version 3 was accepted by TCSVT.
* 2024.12.06 Open source Unicorn Pre ([SparsePCGC](https://github.com/NJUVISION/SparsePCGC))!
* 2024.10.28 Unicorn version 2 has responded to the Call for Proposals for AI-based Point Cloud Coding (m70061 & m70062 in [MPEG](https://dms.mpeg.expert/)).
* 2024.10.05 Initial release of part of the code and results. (The entire source code will be released to the public after the approval from the funding agency.)
* 2024.09.12 Unicorn version 1 was accepted by TPAMI. (https://ieeexplore.ieee.org/document/10682571 and https://ieeexplore.ieee.org/document/10682566)

## Environment

* pytorch, MinkowskiEngine, etc. 
    * You can use docker to simply configure the environment: `docker pull jianqiang1995/pytorch:1.10.0-cuda11.1-cudnn8-devel`


## Dataset

* **ShapeNet**: https://shapenet.org/ 
* **RWTT**: https://texturedmesh.isti.cnr.it/ 
* **MPEG Dataset (Static Objects)**: http://mpegfs.int-evry.fr/MPEG/PCC/DataSets/pointCloud/CfP/datasets/ (MPEG password is required) 
(You can also access some of them on our NJU BOX. ( https://box.nju.edu.cn/d/51327ae7c2644c0fa1c4/ ))
* **MPEG Dataset (Dynamic Objects)**: https://mpeg-pcc.org/index.php/pcc-content-database/
* **KITTI**: https://www.cvlibs.net/datasets/kitti/
* **Ford**: https://mpegfs.int-evry.fr/ws-mpegcontent/MPEG-I/Part05-PointCloudCompression/dataSets_new/Dynamic_Acquisition/Ford  (MPEG password is required) 
(You can also access some of them on our NJU BOX. ( https://box.nju.edu.cn/d/2739fe997265478c8673/ ))


(Note: The training dataset generation methods and the amount of training dataset are not required to be fixed. We provide some examples in `data_utils/datasets/README.sh` to show how to perform sampling, partition, quantization, and other operations on raw mesh or point cloud data to generate the training datasets.)


## Pretrained Models

* **ckpt**: https://box.nju.edu.cn/f/c40a24468034424784eb/?dl=1

## Authors

These files are provided by Nanjing University [Vision Lab](https://vision.nju.edu.cn/). Thanks to Prof. Dandan Ding from Hangzhou Normal University.