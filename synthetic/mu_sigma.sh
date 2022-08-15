#!/bin/bash

#SBATCH --job-name=256-200
#SBATCH --nodes=1
#SBATCH --partition=titan
#SBATCH --gpus=1

# set up environment
source activate jax
BATCH_SIZE=256
NUM_TIMESTEPS=200

export CUDA_VISIBLE_DEVICES=1

python -u synthetic_mu_sigma_eqx.py --batch_size $BATCH_SIZE \
    --num_timesteps $NUM_TIMESTEPS > ./data/titan_bs_${BATCH_SIZE}_ts_${NUM_TIMESTEPS}.csv 2>&1 &