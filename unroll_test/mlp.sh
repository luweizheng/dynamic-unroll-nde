export CUDA_VISIBLE_DEVICES=0

python -u eqx_mlp_unroll.py > ./logs/mlp.csv 2>&1 &