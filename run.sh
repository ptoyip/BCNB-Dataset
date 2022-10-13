#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  Uncomment
#       "##SBATCH" line means to remove one # and start with #SBATCH to be a
#       SLURM command or statement.

#SBATCH -J BCNB_test #Slurm job name

# Set the maximum runtime, uncomment if you need it
##SBATCH -t 48:00:00 #Maximum runtime of 48 hours

# Enable email notificaitons when job begins and ends, uncomment if you need it
#SBATCH --mail-user=kcyipae@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Choose partition (queue) to use. Note: replace <partition_to_use> with the name of partition
#SBATCH -p gpu-share

# Use 1 nodes and 10 cores
#SBATCH -N 1 -n 20 --gres=gpu:1

# Setup runtime environment if necessary
# For example, setup intel MPI environment
# Go to the job submission directory and run your application
module load anaconda3
module add cuda
source activate BCNB
cd $HOME/code/BCNB\ Dataset

srun -n 20 --gres=gpu:1 python train.py --excel_path patient-clinical-data.xlsx --patches_path patches --classification_label ER --positive_label Positive --bag_size 5 --train_ratio 0.8 --optimizer Adam --epoch 3
