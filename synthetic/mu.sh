#!/bin/bash

source activate jax
export CUDA_VISIBLE_DEVICES=0

python -u synthetic.py > ./logs/mu_unroll_0_5.csv &