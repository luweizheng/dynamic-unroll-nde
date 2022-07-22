#!/bin/bash

source activate jax

# !!! we must specify the path to help jax find CUDNN
export LD_LIBRARY_PATH=~/.conda/envs/jax/lib/:$LD_LIBRARY_PATH

python score-based-diffusion.py > logs/a.log 2>&1 &