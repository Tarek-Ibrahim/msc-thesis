#!/bin/bash -l

#SBATCH --time=15:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_trpo_udr
#SBATCH --output=scripts/run_trpo_udr.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

# python trpo_udr_tf.py

python trpo_all_tf.py --dr_type=udr
