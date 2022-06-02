#!/bin/bash -l

#SBATCH --time=04-00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=maml_ppo_activedr_sac_map_torch
#SBATCH --output=results/logs/maml_ppo_activedr_sac_map.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco/2.0

python main.py --mode=1 --verbose=1 --dr_type=active_dr --agent_alg=ppo --env_key=hopper_friction --active_dr_opt=sac --active_dr_rewarder=map_delta --maml
