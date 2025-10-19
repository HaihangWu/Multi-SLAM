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
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G

# Maximum number of tasks/CPU cores used by the job:
# --ntasks=4
# --gpus-per-task=1  # number of gpus per task

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-1:00:00

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate slam

# === Check CUDA and PyTorch before building ===
echo "=== CUDA and PyTorch info ==="
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# === Go to project directory ===
cd /data/gpfs/projects/punim0512/Haihangw-Projects/Multi-SLAM/MASt3R-SLAM

# === Build the package in editable mode ===
pip install --no-build-isolation -e .

echo "Installation complete."


#python ours_prune.py --dataset imagenet --eval-dataset corrupted_imagenet --dataset_path ImageNet-C/frost --finetune_method ours --model resnet34 --num_sample_prune 640  --num_sample_finetune 960  --seed 2021 --epoch 500 --num_rm_blocks 3 --FT MiR
