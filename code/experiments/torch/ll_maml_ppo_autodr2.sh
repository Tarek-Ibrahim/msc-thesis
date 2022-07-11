#!/bin/bash -l

#SBATCH --time=03-00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=maml_ppo_autodr_torch
#SBATCH --output=results/logs/maml_ppo_autodr_%a.out
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

python main2.py --mode=1 --verbose=1 --dr_type=auto_dr --maml --agent_alg=ppo --env_key=lunarlander --seed=$SEED
