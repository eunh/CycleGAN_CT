Paper
===============
* Cycle-consistent adversarial denoising network for multiphase coronary CT angiography
  * Authors: Eunhee Kang, Hyun Jung Koo, Dong Hyun Yang, Joon Bum Seo, and Jong Chul Ye
  * published in Medical Physics (2018): [https://doi.org/10.1002/mp.13284]

Implementation
===============
A PyTorch implementation of cycleGAN for multiphase coronary CT angiography based on original cycleGAN code.
[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix] *Thanks for Jun-Yan Zhu and Taesung Park, and Tongzhou Wang.

* Requirements
  * OS: The package development version is tested on Windows operating systems with Anaconda.
  * Python 3.5.5
  * PyTorch 0.3.1.post2

Datasets
===============
* The whole data used in the paper are private data from ASAN medical center, so only 3 test samples are uploaded.
* CT image files are formated in *.mat.

Main files
===============
* Training: train.py which is handled by scripts/train_for_cardiac.sh
* Test: test_for_cardiac.py which is handled by scripts/test_for_cardiac.sh
  * Learned network for Paper is uploaded.
