#!/usr/bin/env bash

set -x

NAME='sis_landscape'
TASK='SIS'
DATA='landscape'
CROOT='/DATA2/gaoha/tsit/datasets/landscape'
SROOT='/DATA2/gaoha/tsit/datasets/landscape'
CKPTROOT='./checkpoints'
WORKER=20

python3 train.py \
    --name $NAME \
    --task $TASK \
    --checkpoints_dir $CKPTROOT \
    --batchSize 5 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --gan_mode hinge \
    --num_upsampling_layers more \
    --use_vae \
    --alpha 1.0 \
    --display_freq 20 \
    --save_epoch_freq 5 \
    --niter 100 \
    --lr 0.0004 \
    --niter_decay 100 \
    --lambda_vgg 20 \
    --lambda_feat 10 \
    --which_epoch 115 \
    --continue_train \
    --remote "lambda@166.111.227.254:/work/lambda/sf115" \
    --remote_port 18010

