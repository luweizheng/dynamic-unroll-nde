export CUDA_VISIBLE_DEVICES=1

python latent_ode.py --num-iters=1000 --diffrax-solver --print-time-use > logs/lode_diffrax.log 2>&1 &