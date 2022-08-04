export CUDA_VISIBLE_DEVICES=1

python -u synthetic_mu_sigma_eqx_pure_mlp.py > ./data/mu_sigma_pure_mlp.csv 2>&1 &