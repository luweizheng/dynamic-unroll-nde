export CUDA_VISIBLE_DEVICES=0

python latent_ode.py --print-time-use > logs/lode.log 2>&1 &