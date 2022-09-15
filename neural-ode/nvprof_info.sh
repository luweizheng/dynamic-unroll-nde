export CUDA_VISIBLE_DEVICES=0
nvprof --print-api-summary --log-file output.log python neural_ode.py  > logs/ode5.log 2>&1 &
