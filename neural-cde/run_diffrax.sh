#!/bin/bash
export CUDA_VISIBLE_DEVICES=1


indexs=(1 2 3)

for index in "${indexs[@]}"; do
    python -u neural_cde.py --diffrax-solver --num-iters=1000 --num-timesteps=100 --print-time-use > logs/ncde_diffrax_bosh3_"$index".log 2>&1 &
    wait
done