#!/bin/bash -l

#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=auto_dr
#SBATCH --output=auto_dr.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco/2.0

python auto_dr.py
