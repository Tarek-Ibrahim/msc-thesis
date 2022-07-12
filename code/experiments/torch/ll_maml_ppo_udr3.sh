#!/bin/bash -l

#SBATCH --time=02-00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=maml_ppo_udr_torch
#SBATCH --output=results/logs/maml_ppo_udr_3.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load mesa/21.2.3-opengl-osmesa-python3-llvm
module load anaconda
module load mujoco/2.1.0

python main2.py --mode=1 --verbose=1 --dr_type=uniform_dr --maml --agent_alg=ppo --env_key=lunarlander --seed=101