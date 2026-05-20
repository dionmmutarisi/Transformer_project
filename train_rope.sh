#!/bin/bash
#SBATCH --job-name=transformer_rope
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p proq
#SBATCH -C RTX2080Ti
#SBATCH --gres=gpu:1
#SBATCH --output=logs/rope_%j.out
#SBATCH --error=logs/rope_%j.err

. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source $HOME/.bashrc
conda activate dl_assignment

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

mkdir -p /var/scratch/$USER/experiments/rope
cd /var/scratch/$USER/experiments/rope
mkdir -p run_${SLURM_JOB_ID}
cd run_${SLURM_JOB_ID}

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Started: $(date)"

python /var/scratch/$USER/Transformer_project/transformer_rope.py \
    --mode wp \
    --emb 512 \
    --heads 8 \
    --depth 6 \
    --context 256 \
    --dropout 0.1 \
    --batch-size 32 \
    --max-batches 200000 \
    --lr 3e-4 \
    --warmup 2000 \
    --eval-every 10000 \
    --sample-every 20000 \
    --wandb-name "rope-wp-${SLURM_JOB_ID}"

echo "Finished: $(date)"
