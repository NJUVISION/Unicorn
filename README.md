# Unicorn: A Versatile Point Cloud Compressor Using Universal Multiscale Conditional Coding

## Abstract

A universal multiscale conditional coding framework, Unicorn, is proposed to compress the geometry and attribute of any given point cloud. Geometry compression is addressed in [Part I](https://ieeexplore.ieee.org/document/10682571) of this paper, while attribute compression is discussed in [Part II](https://ieeexplore.ieee.org/document/10682566).

For geometry compression, we construct the multiscale sparse tensors of each voxelized point cloud frame and properly leverage lower-scale priors in the current and (previously processed) temporal reference frames to improve the conditional probability approximation or content-aware predictive reconstruction of geometry occupancy in compression.

For attribute compression, Since attribute components exhibit very different intrinsic characteristics from the geometry element, e.g., 8-bit RGB color versus 1-bit occupancy, we process the attribute residual between lower-scale reconstruction and current-scale data. Similarly, we leverage spatially lower-scale priors in the current frame and (previously processed) temporal reference frame to improve the probability estimation of attribute intensity through conditional residual prediction in lossless mode or enhance the attribute reconstruction through progressive residual refinement in lossy mode for better performance.

The proposed Unicorn is a versatile, learning-based solution capable of compressing static and dynamic point clouds with diverse source characteristics in both lossy and lossless modes. Following the same evaluation criteria, Unicorn significantly outperforms standard-compliant approaches like MPEG G-PCC, V-PCC, and other learning-based solutions, yielding state-of-the-art compression efficiency while presenting affordable complexity for practical implementations.

For more information, please visit our homepage: https://njuvision.github.io/Unicorn/ 


## News

* 2025.12.05 **Open source [Unicorn version 1](https://github.com/NJUVISION/Unicorn/tree/main/Unicorn-family/Unicorn-v1) and [version 2](https://github.com/NJUVISION/Unicorn/tree/main/Unicorn-family/Unicorn-v2)!**
* 2025.12.04 Unicorn version 3 was accepted by TCSVT.
* 2024.12.06 Open source Unicorn Pre ([SparsePCGC](https://github.com/NJUVISION/SparsePCGC))!
* 2024.10.28 Unicorn version 2 has responded to the Call for Proposals for AI-based Point Cloud Coding (m70061 & m70062 in [MPEG](https://dms.mpeg.expert/)).
* 2024.10.05 Initial release of part of the code and results. (The entire source code will be released to the public after the approval from the funding agency.)
* 2024.09.12 Unicorn version 1 was accepted by TPAMI. (https://ieeexplore.ieee.org/document/10682571 and https://ieeexplore.ieee.org/document/10682566)


## Acknowledgements and Contact

These files are provided by Nanjing University [Vision Lab](https://vision.nju.edu.cn/). Thanks to Prof. Dandan Ding from Hangzhou Normal University and Prof. Yi Lin from Fudan University for their help. Please contact us (mazhan@nju.edu.cn) if you have any questions.

## Our Team

* **Zhan Ma**
* **Dandan Ding**
* **Tong Chen**
* **Jianqiang Wang**
* **Ruixiang Xue**
* **Jiaxin Li**
* **Junteng Zhang**
* **Junzhe Zhang**
* **Kang You**
* **Wenxi Ma**
* **Jiahao Zhu**
* **Zehong Li**
* **Chengfeng Han**
