export CUDA_VISIBLE_DEVICES=0

python -u neural_ode.py --num-iters=1000 --num-timesteps=200 --print-time-use > logs/node_ralston_1.log 2>&1 &