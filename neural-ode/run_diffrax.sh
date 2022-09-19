export CUDA_VISIBLE_DEVICES=0

python neural_ode.py --diffrax-solver > logs/test4.log 2>&1 &