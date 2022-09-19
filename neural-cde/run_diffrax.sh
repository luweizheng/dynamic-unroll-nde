export CUDA_VISIBLE_DEVICES=0

python neural_cde.py --diffrax-solver=True --print-time-use=True > logs/ncde_diffrax3.log 2>&1 &