CUDA_VISIBLE_DEVICES=0 python main.py --env-id Reacher-v1 --kl-desired 0.002 --lr-vf 0.001 --seed 1 --max-timesteps 1000000 --timesteps-per-batch 2500
