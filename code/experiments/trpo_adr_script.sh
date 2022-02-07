#!/bin/bash -l

#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_trpo_adr
#SBATCH --output=scripts/run_trpo_adr.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python trpo_adr_tf.py
