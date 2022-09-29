export CUDA_VISIBLE_DEVICES=1
python -u neural_cde.py --diffrax-solver --num-iters=1000 --num-timesteps=200 --print-time-use > logs/ncde_bosh3_3.log 2>&1 &