#!/bin/bash

#SBATCH --job-name="rnn_lab"

#SBATCH --workdir=.

#SBATCH --output=rnn_%j.out

#SBATCH --error=rnn_%j.err

#SBATCH --ntasks=1

#SBATCH --gres gpu:1

#SBATCH --time=02:00:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python main.py