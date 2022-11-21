export CUDA_VISIBLE_DEVICES=0

python -u fbsde_raissi_eqx.py --plot=True > logs/high-dim-pde.log 2>&1 &