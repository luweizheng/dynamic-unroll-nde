export CUDA_VISIBLE_DEVICES=0

python -u synthetic_mlp_unroll.py > ./logs/mlp10.csv 2>&1 &