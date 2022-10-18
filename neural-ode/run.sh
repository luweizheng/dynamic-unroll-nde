export CUDA_VISIBLE_DEVICES=0
python -u neural_ode.py --diffrax-solver --num-iters=1000 --num-timesteps=100 --print-time-use > logs/node_diffrax_bosh3_3.log
