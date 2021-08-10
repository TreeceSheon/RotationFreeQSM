#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=unet
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=35000
#SBATCH -e unet.err
#SBATCH -o unet.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:2

module load anaconda/3.7
source activate /scratch/itee/uqzxion3/data/envs/main
module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2
#srun python -u main.py --hybrid --deblur-model ResNet --batch-size 10 --joint
srun python -u main.py --mode whole --model Unet --batch-size 36 --lr-rate 2e-5
#srun python -u main.py --model Unet --batch-size 32 --lr-rate 3e-5
