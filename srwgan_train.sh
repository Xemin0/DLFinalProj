#!/bin/bash

# Job Name
#SBATCH -J SRWGAN

# Time Requested (Not tested yet)
#SBATCH -t 15:00:00

# Number of Cores (max 4)
#SBATCH -c 4

# Single Node (Not yet optmized for Distributed Computing)
#SBATCH -N 1

# Request GPU partition and Access (max 2)
#SBATCH -p gpu --gres=gpu:2

# Request Memory (Not tested yet)
#SBATCH --mem=30G

# Outputs
#SBATCH -e ./scratch/SRWGAN.err
#SBATCH -o ./scratch/SRWGAN.out

############## END OF SLURM COMMANDS #############

# Show CPU infos
lscpu

# Show GPU infos
nvidia-smi

# Force Printing Out `stdout`
export PYTHONUNBUFFERED=TRUE

# Run the Python file with arguments
python3 train_SRWGAN.py --trainnum 800 --epochs 500 --batchsz 10 --gpweight 10.0 --cweight 0.001 --savemodel True
