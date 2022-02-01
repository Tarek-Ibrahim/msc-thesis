#!/bin/bash -l

#SBATCH --time=60:00:00
#SBATCH --mem=38G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_ddpg_udr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python ddpg_udr_tf.py
