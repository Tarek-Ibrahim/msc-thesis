#!/bin/bash -l

#SBATCH --time=02-00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=ppo_udr_torch
#SBATCH --output=results/logs/ppo_udr_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi
#SBATCH --array=0-2

module load anaconda
module load mujoco/2.0

case $SLURM_ARRAY_TASK_ID in
   0)  SEED=101 ;;
   1)  SEED=102  ;;
   2)  SEED=103  ;;
esac

python main2.py --mode=1 --verbose=1 --dr_type=uniform_dr --agent_alg=ppo --env_key=lunarlander --seed=$SEED