#!/bin/bash -l

#SBATCH --time=30:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_maml_h
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python maml_h_tf.py
