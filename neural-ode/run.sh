export CUDA_VISIBLE_DEVICES=0

python -u nerual_ode.py --plot=true > logs/ode4.log 2>&1 &