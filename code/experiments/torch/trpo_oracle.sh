#!/bin/bash -l

#SBATCH --time=02-00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=trpo_oracle_torch
#SBATCH --output=results/logs/trpo_oracle_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi
#SBATCH --array=0-4

module load anaconda
module load mujoco/2.0

case $SLURM_ARRAY_TASK_ID in
   0)  VAL=0.0 ;;
   1)  VAL=0.25  ;;
   2)  VAL=0.5  ;;
   3)  VAL=0.75  ;;
   4)  VAL=1.0 ;;
esac

python main.py --mode=1 --verbose=1 --dr_type=oracle --agent_alg=trpo --env_key=hopper_friction --oracle_rand_value=$VAL