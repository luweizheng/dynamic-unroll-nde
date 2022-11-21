#!/bin/bash


# set up environment
BATCH_SIZE=256
NUM_TIMESTEPS=200

export CUDA_VISIBLE_DEVICES=1

python -u synthetic_mu_sigma_eqx.py --batch_size $BATCH_SIZE \
    --num_timesteps $NUM_TIMESTEPS > ./data/titan_bs_${BATCH_SIZE}_ts_${NUM_TIMESTEPS}.csv 2>&1 &