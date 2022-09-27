export CUDA_VISIBLE_DEVICES=1
python -u neural_cde.py --num-iters=1 --print-time-use > logs/neural-cde3.log 2>&1 &