export CUDA_VISIBLE_DEVICES=0

python neural_ode.py --diffrax-solver=True --print-time-use=True > logs/node_diffrax2.log 2>&1 &