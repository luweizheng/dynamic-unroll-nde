export CUDA_VISIBLE_DEVICES=0

python latent_ode.py --print-time-use > logs/lode_jax_rk4_2.log 2>&1 &