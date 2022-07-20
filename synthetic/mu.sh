#!/bin/bash

source activate jax
export CUDA_VISIBLE_DEVICES=1

python -u synthetic_mu_eqx.py > ./logs/mu.csv &