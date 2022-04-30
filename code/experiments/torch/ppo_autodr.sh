#!/bin/bash -l

#SBATCH --time=02-00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=ppo_autodr_torch
#SBATCH --output=results/logs/ppo_autodr.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco/2.0

python main.py --mode=1 --verbose=1 --dr_type=auto_dr --agent_alg=ppo --env_key=halfcheetah_friction