#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="example.py"

# 48GB GPU
#SBATCH --partition=fos-gpu-l40s
#SBATCH --qos=fos

# --partition=feit-gpu-a100
# --qos=feit

# -p gpu-a100

# -p gpu-h100

#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn

# Request different amount of resources for different nodes
# --constraint=[dlg2|dlg3]
# --constraint=[dlg4]
# --constraint=[dlg5]
# --constraint=[dlg6]
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G

# Maximum number of tasks/CPU cores used by the job:
# --ntasks=4
# --gpus-per-task=1  # number of gpus per task

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-2:00:00

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda deactivate
conda activate Uniformerv2


python ours_prune.py --dataset imagenet --eval-dataset corrupted_imagenet --dataset_path ImageNet-C/frost --finetune_method ours --model resnet34 --num_sample_prune 640  --num_sample_finetune 960  --seed 2021 --epoch 500 --num_rm_blocks 3 --FT MiR
