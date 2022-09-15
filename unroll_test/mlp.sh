export CUDA_VISIBLE_DEVICES=1

python -u synthetic_mlp_unroll.py > ./logs/mlp_titan_hidden64_depth4_width128_steps100_iters1000.csv 2>&1 &