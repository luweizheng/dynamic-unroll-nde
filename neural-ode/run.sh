export CUDA_VISIBLE_DEVICES=1

python -u neural_ode.py --num-iters=2000 --num-timesteps=100 --diffrax-solver --print-time-use > logs/node_bosh3.log 2>&1 &