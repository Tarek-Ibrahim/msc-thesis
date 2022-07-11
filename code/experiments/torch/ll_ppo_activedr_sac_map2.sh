#!/bin/bash -l

#SBATCH --time=03-00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=ppo_activedr_sac_map_torch
#SBATCH --output=results/logs/ppo_activedr_sac_map_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi
#SBATCH --array=0-2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
module load mesa/21.2.3-opengl-osmesa-python3-llvm
module load anaconda
module load mujoco/2.1.0


case $SLURM_ARRAY_TASK_ID in
   0)  SEED=101 ;;
   1)  SEED=102  ;;
   2)  SEED=103  ;;
esac

python main2.py --mode=1 --verbose=1 --dr_type=active_dr --agent_alg=ppo --env_key=lunarlander --active_dr_opt=sac --active_dr_rewarder=map_delta --seed=$SEED
