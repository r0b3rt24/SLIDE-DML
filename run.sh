#!/bin/sh

#SBATCH -p wacc

#SBATCH -t 0-00:10:00

#SBATCH -J NN_omp

#SBATCH -o nn.out -e nn.err

#SBATCH --nodes=1 --cpus-per-task=20

./network