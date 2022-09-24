export CUDA_VISIBLE_DEVICES=0

python latent_ode.py --diffrax-solver --print-time-use > logs/lode.log 2>&1 &