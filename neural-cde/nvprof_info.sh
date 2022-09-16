export CUDA_VISIBLE_DEVICES=0
UNROLL=200
nvprof --csv --print-api-summary --log-file output_unroll_$UNROLL.log python neural_cde.py --unroll=$UNROLL  > logs/cde_unroll_$UNROLL.log 2>&1 &