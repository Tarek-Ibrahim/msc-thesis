#!/bin/bash -l

#SBATCH --time=15:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_maml_trpo
#SBATCH --output=scripts/run_maml_trpo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python maml_trpo_tf.py
