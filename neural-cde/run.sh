export CUDA_VISIBLE_DEVICES=0
python -u neural_cde.py --unroll=2 > logs/neural-cde2.log 2>&1 &