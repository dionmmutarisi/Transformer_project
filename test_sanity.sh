#!/bin/bash
#SBATCH --job-name=transformer_test
#SBATCH --time=00:14:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Run this FIRST on defq (the default partition) to confirm:
#   1. CUDA is visible
#   2. conda env + packages load correctly
#   3. W&B can authenticate and log
#   4. The model trains without errors
# This runs in <15 mins so it can start any time, even during working hours.
# ─────────────────────────────────────────────────────────────────────────────

. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source $HOME/.bashrc
conda activate dl_assignment

echo "=== GPU check ==="
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Memory:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"

echo "=== Package check ==="
python -c "
import torch, wandb, optuna
print('torch:', torch.__version__)
print('wandb:', wandb.__version__)
print('optuna:', optuna.__version__)
"

mkdir -p $HOME/experiments/test
cd $HOME/experiments/test

echo "=== Short training run (500 steps) ==="
python /var/scratch/$USER/Transformer_project/transformer_train.py \
    --mode toy \
    --emb 128 \
    --heads 4 \
    --depth 2 \
    --context 64 \
    --batch-size 32 \
    --max-batches 500 \
    --eval-every 200 \
    --wandb-name "das5-sanity-check"

echo "=== Done ==="
