CUDA_VISIBLE_DEVICES=0 python main.py --env-id Reacher-v1 --kl-desired 0.002 --kfac-update-vf 2 --lr-vf 0.001 --seed 1 --max-timesteps 40000000 --timesteps-per-batch 8000 --use-pixels True
# or try
#CUDA_VISIBLE_DEVICES=0 python main.py --env-id Reacher-v1 --kl-desired 0.004 --kfac-update-vf 2 --lr-vf 0.001 --seed 1 --max-timesteps 40000000 --timesteps-per-batch 8000 --use-pixels True
