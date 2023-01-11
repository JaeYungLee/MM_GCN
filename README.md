# Multi-hop Modulated Graph Convolutional Networks for 3D Human Pose Estimation (BMVC2022)
This repository holds the Pytorch implementation of [Multi-hop Modulated Graph Convolutional Networks for 3D Human Pose Estimation](https://bmvc2022.mpi-inf.mpg.de/0207.pdf) by Jae Yung Lee and I Gil Kim.

## Quick Start
### Evaluating our pre-trained models
### GT Evaluation
```c
Human3.6M Dataset
python test.py -d Human36M -k gt -sk 2 -c ${CHECKPOINT_PATH} --test_model {MODEL_PATH} -ch {CHANNEL_NUM} -j_out 17 -g {GPU_IDX}
```
