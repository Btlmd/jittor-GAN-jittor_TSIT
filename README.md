# Jittor Landscape Generation with TSIT

![main result](assets/img.jpg)

## Introduction

This repository provides the implementation of Team **GAN!** in
- [Jittor AI Contest](https://www.educoder.net/competitions/index/Jittor-3) Track 1: Landscape Generation

We implemented our model based on [TSIT](https://github.com/EndlessSora/TSIT) network architecture, and have achieved a score of 0.5189 in Track 1, ranking 15 in Board A.

Download our [results](https://cloud.tsinghua.edu.cn/f/3d180eba21024b3bbe72/?dl=1).

Assignment report see `assets/REPORT.pdf`.

## Install and Validate
### Environments

We train and evaluate our model in the following environments.

The total training time is estimated to be 65 ~ 70 hours and inference time is about several minutes.

| -                    | Training                                                                                                                                                           | Evaluation                                                                               |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Environment          | Ubuntu 20.04 LTS<br /> Python 3.8.13<br />Jittor 1.3.4.15<br />CUDA 11.6<br />Open MPI 4.0.3                                                                       | Ubuntu 20.04 LTS<br />Python 3.7.13<br />Jittor 1.3.4.9<br />CUDA 11.6<br /> NO Open MPI |
| GPU                  | NVIDIA A100-SXM4-40GB                                                                                                                                              | NVIDIA GeForce RTX 3090                                                                  |
| Jittor<br />Unittest | Failed <br >`test_conv_transpose3d` <br />`test_conv3d`<br />due to low precision. | Pass                                                                                     |

### Packages

```
pip install -r requirements.txt
```

### Testing Pretrained Models

We trained **two separate models** and **manually mixed their result** to form our final submission. To reproduce our result, you can

1. Download our [pretrained models](https://cloud.tsinghua.edu.cn/d/00b780fc19144de1980e/) and unzip them to `./checkpoints` so that the directory looks like
   ```bash
   .
   └──checkpoints
      └── sis_landscape
          ├── aux_net_E.pkl
          ├── aux_net_G.pkl
          ├── main_net_E.pkl
          └── main_net_G.pkl
   ```
   Note that `*_net_D.pkl` (Discriminator) is not necessary at evaluation.
2. Download the [test dataset](https://cloud.tsinghua.edu.cn/f/c1618c846a7842da94e3/?dl=1).
3. Config the path of dataset in `validation.sh`. It would evaluate the models on test dataset and call `selection.py` to reproduce our manual selection process.
4. Run
   ```bash
   bash validation.sh
   ```
5. The result will be ready at `./result.zip`

## Dataset Preprocessing

We made no modifications to the images provided before they're fed into our network, but we manually constructed three subsets of the training set, i.e.

1. `Total`. Containing the original 10,000 images.
2. `Selection I`. Manually remove some images from `Total`, 8115 images left.
3. `Selection II`. Based on `Selection I`, removed more images. Contains 7331 images.

Download our [preprocessed training sets](https://cloud.tsinghua.edu.cn/d/6575d52e2b404e7895a6/)

## Training Scripts

Train on single GPU
```
bash ./train.sh
```

Train on multiple GPUs.
```
bash ./multi.sh
```

### About our training process
Our training for model `main` involves 4 phases

| Phase | Epoch     | batch_size | training set   | learning rate |
|-------|-----------|------------|----------------|---------------|
| I     | [1, 38]   | 2          | `Total`        | 2e-4          |
| II    | (38, 71]  | 30         | `Selection I`  | 1.2e-3        |
| III   | (71, 95]  | 5          | `Selection II` | 4e-4          |
| IV    | (95, 110] | 5          | `Selection I`  | 2e-4          |

This is **not** a carefully designed schedule. It is a compromise of our remaining time, access to calculation power and temporary thoughts.

## Inference Scripts
Config and run
```
bash ./test.sh
```
And results will be compressed into a 7zip file.


## Acknowledgement

The implementation of this repository is based on TSIT ([[Code Base]](https://github.com/EndlessSora/TSIT)   [[Paper]](https://arxiv.org/abs/2007.12072)). You may somehow view it as an incomplete "style-transfer" from its original pytorch implementation to jittor framework.

Our spectral normalization uses the implementation of  [[PytorchAndJittor]](https://github.com/Lewis-Liang/PytorchAndJittor).

We implement our model with Jittor. Jittor is a deep learning framework based on dynamic compilation (Just-in-time), using innovative meta-operators and unified computational graphs internally. Meta-operators are as easy to use as Numpy, and beyond Numpy can achieve more complex and more efficient operations. The unified computing graph combines the advantages of static and dynamic computing graphs, and provides high-performance optimization. Deep learning models developed based on Jittor can be automatically optimized in real time and run on CPU or GPU.
