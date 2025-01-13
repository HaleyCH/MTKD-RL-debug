# Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition

This project provides source code for official implementation of  Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition (MTKD-RL):

## Installation

### Requirements

Ubuntu 20.04 LTS

Python 3.9

CUDA 11.8

please install python packages:

```
pip install -r requirements.txt
```

## Perform experiments on CIFAR-100 dataset

### Dataset

CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

### Training teacher networks

```
python train_baseline.py --model [model_name] \
    --data-folder [your dataset path] \
    --checkpoint-dir [your checkpoint saved path]
```
* `--model`: specify the teacher model, such as `RegNetY_400MF`,`RegNetX_400MF`, `resnet32x4`, `wrn_28_4`
* `--data-folder`: specify the folder of dataset
* `--checkpoint-dir`: specify the folder for storing checkpoints


### Training the student network with Multi-teacher KD
configure the `setting.py` with `teacher_name:teacher_path`:
```
teacher_model_path_dict = {
    'RegNetY_400MF': '[pre-traiend model pth path]',
    'RegNetX_400MF': '[pre-traiend model pth path]',
    'resnet32x4': '[pre-traiend model pth path]',
    'wrn_28_4': '[pre-traiend model pth path]',
    }
```
#### AVER: Multi-teacher KD with equal weights
Distill a student ShuffleV2 with the teacher pool (RegNetY_400MF, RegNetX_400MF, resnet32x4, wrn_28_4)

```
python train_student_avg.py \
    --data [your dataset path] \
    --arch ShuffleV2 \
    --checkpoint-dir [your checkpoint saved path] \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --rank 0 
```

* `--data`: specify the folder of dataset
* `--arch`: specify the student architecture, e.g. `ShuffleV2`
* `--checkpoint-dir`: specify the folder for storing checkpoints
* `--teacher-name-list`: specify the teacher names to construct the teacher pool, e.g. `RegNetY_400MF`, `RegNetX_400MF`, `resnet32x4`





#### MTKD-RL (Ours)
Distill a student ShuffleV2 with the teacher pool (RegNetY_400MF, RegNetX_400MF, resnet32x4, wrn_28_4)
```
python train_student_rl.py \
    --data [your dataset path] \
    --arch ShuffleV2 \
    --dynamic \
    --checkpoint-dir [your checkpoint saved path] \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --rank 0 
```
* `--dynamic`: specify whether using dynamic weight aggregation strategy. We found that various networks may achieve different performance under with or without  `dynamic`.

| Student | W/ dynamic | W/O dynamic |
| :--: | :--: | :--: |
|  RegNetX-200MF | 79.93| 80.58|
|  MobileNetV2 | 74.77 | 74.29|
|  ShuffleNetv2 | 78.74 | 78.35 |
|  ResNet-56 | 74.95 | 75.51 |

