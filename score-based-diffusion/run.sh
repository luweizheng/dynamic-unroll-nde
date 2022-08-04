#!/bin/bash

source activate jax

# !!! we must specify the path to help jax find CUDNN
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=~/.conda/envs/jaxpde/lib/:$LD_LIBRARY_PATH

python -u score-based-diffusion.py --train_iters 1000 > logs/euler_1000.log 2>&1 &