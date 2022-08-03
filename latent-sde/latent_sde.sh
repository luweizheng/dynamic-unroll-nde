#!/bin/bash

source activate jax
export CUDA_VISIBLE_DEVICES=0

python latent_sde.py --show-prior False --plot True --unroll 1