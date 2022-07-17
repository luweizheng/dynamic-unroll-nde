#!/bin/bash

source activate jax
export CUDA_VISIBLE_DEVICES=1

python latent_sde.py --show-prior False --unroll 1