export CUDA_VISIBLE_DEVICES=0
nvprof --csv --print-api-summary --log-file output_unroll_1.log python neural_cde.py --unroll=1  > logs/cde_unroll_1.log 2>&1 &