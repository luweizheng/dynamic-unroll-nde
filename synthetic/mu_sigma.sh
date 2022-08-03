#!/bin/bash

source activate jax
export CUDA_VISIBLE_DEVICES=1

python -u synthetic_mu_sigma_eqx.py > ./logs/mu_sigma.csv 2>&1 &