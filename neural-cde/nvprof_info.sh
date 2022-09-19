export CUDA_VISIBLE_DEVICES=0
UNROLL=40
nvprof --csv --print-api-summary --log-file output2_unroll_$UNROLL.log python neural_cde.py --unroll=$UNROLL  > logs/cde2_unroll_$UNROLL.log 2>&1 &