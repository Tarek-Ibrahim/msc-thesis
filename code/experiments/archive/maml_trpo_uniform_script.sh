#!/bin/bash -l

#SBATCH --time=03-00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --job-name=run_maml_uniform_trpo
#SBATCH --output=scripts/run_maml_uniform_trpo.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarek.ibrahim@tuni.fi

module load anaconda
module load mujoco

python maml_trpo_uniform_tf.py

# python trpo_all_tf.py --dr_type=udr --maml