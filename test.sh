#!/usr/bin/env bash

set -x

NAME='sis_landscape'
TASK='SIS'
DATA='landscape'
CROOT='/work/data/data'
SROOT='/work/data/data'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results'
EPOCH="57"

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
#    --show_input
