#!/bin/bash
#SBATCH -A research
#SBATCH -n 30
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=9048
#SBATCH --time=4-00:00:00
#SBATCH -w gnode03

module add cuda/8.0 
module add cudnn/6-cuda-8.0

#
echo "I ran on:"
cd $SLURM_SUBMIT_DIR
echo $SLURM_NODELIST
#

#source activate conda_2_7

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_ae.py --name ae_doafn_wide


