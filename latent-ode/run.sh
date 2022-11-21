#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

unrolls=(1, 2, 5, 10, 20, 50, 100)
indexs=(1, 2, 3)

for index in "${indexs[@]}"; do
    for unroll in "${unrolls[@]}"; do
        python -u run_latent_ode.py --unroll="$unroll" --num-iters=1000 --num-timesteps=200 --print-time-use > logs/lode_euler_unroll="$unroll"_"$index".log 2>&1 &
        wait
    done
done
