# Unicorn v3: Improving Occupancy Prediction for Multiscale Point Cloud Geometry Compression

## Abstract

Multiscale sparse representation offers significant advantages in point cloud geometry compression, delivering state-of-the-art performance compared to both standardized solutions and other learned approaches. A crucial component of this framework is the cross-scale occupancy prediction, which employs the lower-scale reference representation either from the current frame alone or from both the current and temporal reference frames to establish conditional priors for either static or dynamic coding. However, existing works mainly use local computations, e.g., sparse convolutions and kNN attention, to exploit correlations in such a representation; these methods usually fail to adequately capture global coherence. In addition, the fixed configuration of lossless-lossy scales cannot adapt to temporal dynamics, which limits the reconstruction quality of temporal references in dynamic coding. These limitations constrain the generation of more effective priors used for conditional coding. To address these issues, we propose two new techniques. The first is KPA (Key Point-driven Attention), which integrates both local and global characteristics. The second is AdaScale (Adaptive Lossy/Lossless Scale), which decides whether the transitional scale should be in lossless or lossy mode based on temporal displacement, thereby enhancing the reconstruction quality of the temporal reference. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art methods, including rules-based standard codecs like G-PCC and V-PCC, as well as learning-based approaches like Unicorn and TMAP, across both static/dynamic and lossy/lossless coding scenarios.

For more information, please visit our homepage: https://njuvision.github.io/Unicorn/ 


## News

* 2025.12.05 **Open source [Unicorn version 1](https://github.com/NJUVISION/Unicorn) and [version 2](https://github.com/NJUVISION/Unicorn)!**
* 2025.12.04 Unicorn version 3 was accepted by TCSVT.
* 2024.12.06 Open source Unicorn Pre ([SparsePCGC](https://github.com/NJUVISION/SparsePCGC))!
* 2024.10.28 Unicorn version 2 has responded to the Call for Proposals for AI-based Point Cloud Coding (m70061 & m70062 in [MPEG](https://dms.mpeg.expert/)).
* 2024.10.05 Initial release of part of the code and results. (The entire source code will be released to the public after the approval from the funding agency.)
* 2024.09.12 Unicorn version 1 was accepted by TPAMI. (https://ieeexplore.ieee.org/document/10682571 and https://ieeexplore.ieee.org/document/10682566)

## Results

`./results`


## Authors

These files are provided by Nanjing University [Vision Lab](https://vision.nju.edu.cn/). Thanks to Prof. Dandan Ding from Hangzhou Normal University.
