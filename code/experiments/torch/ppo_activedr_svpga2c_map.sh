#!/bin/bash -l

#SBATCH --time=03-00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=ppo_activedr_svpga2c_map_torch
#SBATCH --output=results/logs/ppo_activedr_svpga2c_map.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco/2.0

python main.py --mode=1 --verbose=1 --dr_type=active_dr --agent_alg=ppo --env_key=halfcheetah_friction --active_dr_opt=svpg_a2c --active_dr_rewarder=map_thr