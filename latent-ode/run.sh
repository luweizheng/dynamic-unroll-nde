#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

unrolls=(200)
indexs=(1)

for index in "${indexs[@]}"; do
    for unroll in "${unrolls[@]}"; do
        python -u run_latent_ode.py --unroll="$unroll" --num-iters=1000 --num-timesteps=200 --print-time-use > logs/the_lode_euler_unroll="$unroll"_"$index".log 2>&1 &
        wait
    done
done