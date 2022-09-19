export CUDA_VISIBLE_DEVICES=0

python neural_ode.py --diffrax-solver --num-timesteps=100 --num-iters=1 > logs/test5.log 2>&1 &