#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="example.py"

# 48GB GPU
# --partition=fos-gpu-l40s
# --qos=fos

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

# -p gpu-a100

# -p gpu-h100

# --partition deeplearn
# --qos gpgpudeeplearn

# Request different amount of resources for different nodes
# --constraint=[dlg4|dlg5|dlg6]
# --constraint=[dlg4]
# --constraint=[dlg5]
# --constraint=[dlg6]
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G

# Maximum number of tasks/CPU cores used by the job:
# --ntasks=4
# --gpus-per-task=1  # number of gpus per task

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-00:30:00

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate slam

bash ./scripts/eval_MA.sh --no-calib