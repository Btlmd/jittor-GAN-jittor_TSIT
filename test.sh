#!/usr/bin/env bash

set -x

NAME='sis_landscape'
TASK='SIS'
DATA='landscape'
CROOT='/dataset'
SROOT='/dataset'
CKPTROOT='./checkpoints'
WORKER=8
RESROOT='./results'
EPOCH="39"

python3 test.py \
    --name $NAME \
    --task $TASK \
    --checkpoints_dir $CKPTROOT \
    --batchSize 10 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --num_upsampling_layers more \
    --use_vae \
    --alpha 1.0 \
    --results_dir $RESROOT \
    --which_epoch $EPOCH

7z a e$EPOCH.7z $RESROOT/$NAME/test_$EPOCH/*