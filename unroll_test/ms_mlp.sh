export CUDA_VISIBLE_DEVICES=0

python -u ms_syn_mlp_unroll.py > ./logs/mlp_msd.csv 2>&1 &