#!/bin/bash -l

#SBATCH --time=00:05:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=debug_dr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python maml_adr_tf.py
