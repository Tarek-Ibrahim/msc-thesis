#!/bin/bash -l

#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=debug
#SBATCH --output=results/logs/debug.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load mesa/21.2.3-opengl-osmesa-python3-llvm
module load anaconda
module load mujoco/2.1.0

python test.py
