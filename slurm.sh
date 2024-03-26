#!/bin/bash
## This line requests that the job be allocated one node
#SBATCH -N 1

## This line requests one task for the job
#SBATCH -n 1

## This specifies the amount of memory required for the job. 
## Here, it's set to 10 Gigabytes. This memory is allocated per node.
#SBATCH --mem=10G

## This sets the partition (or queue) to gpu-a100
## Partitions are defined by the system administrators and 
## determine the set of nodes the job can run on. 
#SBATCH -p gpu-a100

## This specifies the account for job charging or accounting. 
## Here, it's set to an account named 'test'. To find out which
## accounts are allowed for your user account, just run
## "sacctmgr list account" and it will list all the valid accounts
## from which you can choose.
#SBATCH -A test

## This sets the time limit for the job. 
## Here, the job can run for up to 1 hour. 
## If the job exceeds this limit, SLURM will terminate it.
#SBATCH -t 1:00:00

## This requests a specific generic resource (GRES), in this case, 
## a GPU. Specifically, it requests one Nvidia A100 GPU with 10 GB of memory.
#SBATCH --gres=gpu:a100_1g.10gb:1

## This sets the name of the job to "jhb11-GPU-mnist". 
## This name will appear in the queue and in various SLURM reports, 
## helping you to identify your job.
#SBATCH --job-name "jhb11-GPU-mnist"

## This specifies that all the output from the job (stdout) will 
## be written to the file output.out. This file will be created in the 
## directory from which the SLURM script is submitted.
#SBATCH --output=output.out

## each of these lines uses the "module load" command
## to load the modules our python script will need to use
ml cuda
ml python/3.9.15
ml miniconda3/4.10.3

## change directories to your user scratch location where your 
## script is
cd /scratch/ptolemy/users/jhb11/hello_world

## now create your conda environment - doesn't work through slurm??
## conda env create --file ./environment.yml --name jhb11-GPU-mnist --force

## activate conda environment
source activate jhb11-GPU-mnist

## Set a path you can use inside your python script to find your data files
export DATA_FILE_PATH="/scratch/ptolemy/users/jhb11/hello_world"

## change directories to your user scratch location where your 
## script is
cd /scratch/ptolemy/users/jhb11/hello_world

## run the script
python hello_world.py
