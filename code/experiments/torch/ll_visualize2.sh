#!/bin/bash -l

#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --job-name=visualize_xp2
#SBATCH --output=results/logs/visualize_xp2.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
module load mesa/21.2.3-opengl-osmesa-python3-llvm
module load anaconda
module load mujoco/2.1.0

python visualize2.py --agent_alg=ppo --mode=1 --env_key=lunarlander --plot_ts_results --test_random --save_results --xp_name=xp_full --include_oracle

