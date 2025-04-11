#!/bin/bash
#SBATCH -n 1                 # Number of cores
#SBATCH -N 1                 # Ensure all cores are on one machine
#SBATCH -t 0-10:00           # Runtime in D-HH:MM
#SBATCH -p serial_requeue    # Cluster partition
#SBATCH --mem=30GB           # Memory pool for all cores
#SBATCH -o EXPDIR/out_%j.txt # File to which STDOUT will be written
#SBATCH -e EXPDIR/err_%j.txt # File to which STDERR will be written

python -u run.py EXPDIR EXPNAME KWARGS