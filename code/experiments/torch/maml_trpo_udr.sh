#!/bin/bash -l

#SBATCH --time=20:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=maml_trpo_udr_torch
#SBATCH --output=results/logs/maml_trpo_udr.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python main.py --mode=1 --verbose=1 --dr_type=uniform_dr --maml --agent_alg=trpo --env_key=hopper_friction