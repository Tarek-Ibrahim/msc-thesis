#!/bin/bash -l

#SBATCH --time=03-00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=trpo_activedr_svpgddpg_disc_torch
#SBATCH --output=results/logs/trpo_activedr_svpgddpg_disc.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco/2.0

python main.py --mode=1 --verbose=1 --dr_type=active_dr --agent_alg=trpo --env_key=hopper_friction --active_dr_opt=svpg_ddpg --active_dr_rewarder=disc