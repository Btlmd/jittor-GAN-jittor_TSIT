#!/usr/bin/env bash

set -x

NAME='sis_landscape'
TASK='SIS'
DATA='landscape'
CROOT='./datasets/landscape'
SROOT='./datasets/landscape'
CKPTROOT='./checkpoints'
WORKER=4

python train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 2 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --gan_mode hinge \
    --num_upsampling_layers more \
    --use_vae \
    --alpha 1.0 \
    --display_freq 200 \
    --save_epoch_freq 1 \
    --niter 100 \
    --niter_decay 100 \
    --lambda_vgg 20 \
    --lambda_feat 10
#    --which_epoch 38 \
#    --continue_train

