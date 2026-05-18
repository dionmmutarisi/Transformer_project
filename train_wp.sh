#!/bin/bash
#SBATCH --job-name=transformer_wp
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p proq
#SBATCH -C RTX2080Ti
#SBATCH --gres=gpu:1
#SBATCH --output=logs/wp_%j.out
#SBATCH --error=logs/wp_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# DAS5 NOTES (from the cluster guide):
# - proq partition: long GPU jobs, RTX2080Ti (11GB VRAM), no time limit
# - defq partition: 15-min limit during working hours (Mon-Fri 8:00-20:00)
# - We use proq so the job can run overnight / over the weekend freely
# - If proq nodes are busy, submit during off-hours on defq instead:
#     sbatch --begin=20:00 train_wp.sh
# ─────────────────────────────────────────────────────────────────────────────

# Load GPU drivers
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

# Activate conda — installed on scratch as per the guide
source $HOME/.bashrc
conda activate dl_assignment

# Verify GPU is visible — silent failures are common, always check this
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Create a unique output directory for this run
mkdir -p $HOME/experiments/wp
cd $HOME/experiments/wp
RUN_DIR="run_${SLURM_JOB_ID}"
mkdir -p $RUN_DIR
cd $RUN_DIR

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Run dir:   $(pwd)"
echo "Started:   $(date)"

# Run training — Wikipedia data, larger model
python /var/scratch/$USER/Transformer_project/transformer_train.py \
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
    --wandb-name "wp-das5-${SLURM_JOB_ID}"

echo "Finished:  $(date)"
