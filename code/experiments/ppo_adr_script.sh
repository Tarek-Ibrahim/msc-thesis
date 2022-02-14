#!/bin/bash -l

#SBATCH --time=90:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_ppo_adr
#SBATCH --output=scripts/run_ppo_adr.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python ppo_all_tf.py --dr_type=adr --maml=False
