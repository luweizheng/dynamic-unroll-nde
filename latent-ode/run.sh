export CUDA_VISIBLE_DEVICES=1
python latent_ode.py --num-iters=1000 --num-timesteps=200 --diffrax-solver --print-time-use > logs/lode_bosh3_1.log 2>&1 &