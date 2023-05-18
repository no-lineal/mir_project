#!/bin/bash
#SBATCH -J grid_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 100:00:00

source activate bit38

python -B train.py