export CUDA_VISIBLE_DEVICES=0

# unrolls=(1 5 10 20 50 100)
# indexs=(1 2 3)

# for unroll in "${unrolls[@]}"; do
#     for index in "${indexs[@]}"; do
#         python -u neural_ode.py --unroll "$unroll" --num-iters=1000 --num-timesteps=100 --print-time-use > logs/node_ralston_unroll="$unroll"_"$index".log 2>&1 &
#         wait
#     done
# done


python -u neural_ode.py --unroll=1 --num-iters=1000 --num-timesteps=100 --print-time-use --plot > logs/node_ralston.log 2>&1 &
