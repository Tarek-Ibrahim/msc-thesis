#!/bin/bash -l

#SBATCH --time=03-00
#SBATCH --mem=38G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_maml_udr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python maml_udr_tf.py
