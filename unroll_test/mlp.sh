export CUDA_VISIBLE_DEVICES=0
python -u synthetic_mlp_unroll.py --mu-depth=3 --sigma-depth=3 \
                    --mu-width-size=64 --sigma-width-size=64 \
                     --num-timesteps=100 --num-iters=500 > ./logs/mlp_titan_hidden64_depth3_width64_steps100_iters500_3.csv 2>&1 &