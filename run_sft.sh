#!/bin/bash
#SBATCH --job-name=biorlhf_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/sft_%j.log
#SBATCH --error=logs/sft_%j.err

# ============================================================
# BioRLHF SFT Training Job Script for Cayuga HPC
# ============================================================

echo "============================================================"
echo "BioRLHF SFT Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================"

# Create logs directory
mkdir -p logs

# Load modules (adjust based on Cayuga's available modules)
module purge
module load cuda/12.1  # or available CUDA version
# module load anaconda3  # if using system anaconda

# Activate conda environment
source ~/.bashrc
conda activate biorlhf

# Verify GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="./cache/transformers"
export HF_HOME="./cache/huggingface"
export WANDB_DIR="./wandb"

# Create cache directories
mkdir -p $TRANSFORMERS_CACHE $HF_HOME $WANDB_DIR

# Run training
echo "Starting SFT training..."
python sft_train.py \
    --model "mistralai/Mistral-7B-v0.3" \
    --dataset "kmp_sft_dataset.json" \
    --output_dir "./kmp_sft_model" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --max_seq_length 2048 \
    --lora_r 32 \
    --lora_alpha 64 \
    --wandb_project "biorlhf" \
    --wandb_run "kmp_sft_$(date +%Y%m%d_%H%M%S)"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Training completed successfully!"
    echo "Model saved to: ./kmp_sft_model"
    echo "End time: $(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Training failed with exit code $?"
    echo "Check logs/sft_${SLURM_JOB_ID}.err for details"
    echo "============================================================"
    exit 1
fi
