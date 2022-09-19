export CUDA_VISIBLE_DEVICES=1
STEPS=100
ITERS=200
python -u synthetic_mlp_unroll.py --num-timesteps=100 --num-iters=200 > ./logs/mlp_titan_hidden64_depth4_width128_steps100_iters200.csv 2>&1 &