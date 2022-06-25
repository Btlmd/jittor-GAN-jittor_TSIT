#!/usr/bin/env bash

set -x

NAME='sis_landscape'
TASK='SIS'
DATA='landscape'
CROOT='/DATA2/gaoha/tsit/datasets/landscape'
SROOT='/DATA2/gaoha/tsit/datasets/landscape'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results'
EPOCH="total"

python3 test.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
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
