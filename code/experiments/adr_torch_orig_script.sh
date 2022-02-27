#!/bin/bash -l

#SBATCH --time=03-00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_adr_torch_orig
#SBATCH --output=scripts/run_adr_torch_orig.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python adr_torch_orig.py

# python trpo_all_tf.py --dr_type=adr
