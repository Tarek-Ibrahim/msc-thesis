#!/bin/bash -l

#SBATCH --time=00:05:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=ppo_torch
#SBATCH --output=results/logs/hc_ppo.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load mesa/21.2.3-opengl-osmesa-python3-llvm
module load anaconda
module load mujoco/2.1.0

python main.py --mode=0 --verbose=1 --agent_alg=ppo --env_key=hopper_friction
