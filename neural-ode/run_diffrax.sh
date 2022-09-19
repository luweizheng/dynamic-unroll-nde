export CUDA_VISIBLE_DEVICES=1

python neural_ode.py --diffrax-solver --num-timesteps=100 --num-iters=1000 > logs/test8.log 2>&1 &