# Multi-hop Modulated Graph Convolutional Networks for 3D Human Pose Estimation (BMVC2022)
## Introduction
This repository holds the Pytorch implementation of [Multi-hop Modulated Graph Convolutional Networks for 3D Human Pose Estimation](https://bmvc2022.mpi-inf.mpg.de/0207.pdf) by Jae Yung Lee and I Gil Kim.

## Quick Start
This repository is build upon Python v3.6 and Pytorch v1.8.2 on Ubuntu 18.04. All experiments are conducted on a single NVIDIA RTX QUADRO 6000 GPU. See requirements.txt for other dependencies. We recommend installing Python v3.6 from Anaconda and installing Pytorch (>= 1.8.0) following guide on the official instructions according to your specific CUDA version. Then you can install dependencies with the following commands.

### Dataset 
2D detections for Human3.6M datasets are provided by [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) Pavllo et al.

### Pre-trained models
The pre-trained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1XLr6CHkhMEldYkAA74EY6Wg6KWFLwD3z/view?usp=share_link).

### Evaluation (GT)
```c
Human3.6M Dataset
python test.py -d Human36M -k gt -sk {HOP_NUM} -c ${CHECKPOINT_PATH} --test_model {MODEL_PATH} -ch {CHANNEL_NUM} -j_out 17 -g {GPU_IDX}
```

## Reference
```c
@inproceedings{lee22multi,
 author = {Jae Yung Lee and I Gil Kim},
 booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
 title = {Multi-hop Modulated Graph Convolutional Networks for 3D Human Pose Estimation},
 year = {2022}
}
```

## Acknowledgement
Part of our code is borrowed from the following repositories.

- [Modulated GCN](https://github.com/ZhimingZo/Modulated-GCN)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)

e thank to the authors for releasing their codes. Please also consider citing their works.
