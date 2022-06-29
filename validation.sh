#!/usr/bin/env bash
set -x

###### Begin Configurations ######
DATASET='/work/data/data'
###### End Configurations ######

NAME='sis_landscape'
TASK='SIS'
DATA='landscape'
CROOT=$DATASET
SROOT=$DATASET
CKPTROOT='./checkpoints'
WORKER=8
RESROOT='./results'

EPOCH="main"
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

EPOCH="aux"
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

python3 selection.py

zip -j result.zip $RESROOT/$NAME/test_mixed/*