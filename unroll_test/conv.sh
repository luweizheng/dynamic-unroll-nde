export CUDA_VISIBLE_DEVICES=1

python -u eqx_conv_unroll.py > logs/conv.csv 2>&1 &