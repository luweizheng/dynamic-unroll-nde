export CUDA_VISIBLE_DEVICES=0

python -u neural_ode.py --num-iters=20 --print-time-use > logs/ode_rk.log 2>&1 &