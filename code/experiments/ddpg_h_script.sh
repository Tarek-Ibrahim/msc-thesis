#!/bin/bash -l

#SBATCH --time=20:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_ddpg_h
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python ddpg_h_tf.py
