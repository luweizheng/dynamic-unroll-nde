export CUDA_VISIBLE_DEVICES=1

python -u synthetic_mlp_unroll.py > ./logs/mlp_titan_hidden64_depth4_width128_steps2000_iters1000_2.csv 2>&1 &