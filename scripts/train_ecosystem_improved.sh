#!/bin/bash
#
# BioRLHF Training Script - Ecosystem Improved Model
# ====================================================
#
# This script trains a model on the combined dataset including:
# - Original KMP study data (363 examples)
# - Ecosystem failure-based examples (15 examples)
#   - Calibration training
#   - Adversarial resistance
#   - Protocol completeness
#   - Fact drilling
#
# Requirements:
# - CUDA-capable GPU (recommended: A100, V100, or 4090)
# - 24GB+ VRAM for Mistral-7B with 4-bit quantization
# - Python environment with: torch, transformers, peft, trl, bitsandbytes
#
# Usage:
#   ./scripts/train_ecosystem_improved.sh
#
# Or on HPC with SLURM:
#   sbatch scripts/train_ecosystem_improved.sh
#

# ==============================================================================
# SLURM Configuration (for HPC clusters - uncomment if using SLURM)
# ==============================================================================
#SBATCH --job-name=biorlhf_ecosystem
#SBATCH --output=logs/biorlhf_ecosystem_%j.out
#SBATCH --error=logs/biorlhf_ecosystem_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# ==============================================================================
# Environment Setup
# ==============================================================================
echo "============================================================"
echo "BioRLHF Ecosystem Training"
echo "============================================================"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo ""

# Activate conda environment (adjust path as needed)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate biorlhf

# Set working directory
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# ==============================================================================
# Training Configuration
# ==============================================================================

# Model settings
MODEL="mistralai/Mistral-7B-v0.3"
DATASET="data/combined_training.json"
OUTPUT_DIR="./ecosystem_improved_model"

# Training hyperparameters (optimized based on prior BioRLHF experiments)
EPOCHS=10           # More epochs for better fact memorization
BATCH_SIZE=4        # Adjust based on GPU memory
GRAD_ACCUM=4        # Effective batch size = 16
LEARNING_RATE=2e-4  # Standard for LoRA fine-tuning
MAX_LENGTH=1024     # Sufficient for most examples

# LoRA configuration (higher rank for domain knowledge)
LORA_R=64           # Higher rank for better capacity
LORA_ALPHA=128      # Alpha = 2 * r

# Logging
WANDB_PROJECT="biorlhf"
WANDB_RUN="ecosystem_improved_$(date +%Y%m%d_%H%M%S)"

# ==============================================================================
# Pre-training Checks
# ==============================================================================
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "Model:        $MODEL"
echo "Dataset:      $DATASET"
echo "Output:       $OUTPUT_DIR"
echo "Epochs:       $EPOCHS"
echo "Batch size:   $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "LoRA r/α:     $LORA_R / $LORA_ALPHA"
echo "Max length:   $MAX_LENGTH"
echo ""

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Run: python scripts/merge_training_data.py"
    exit 1
fi

# Count examples
EXAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$DATASET'))))")
echo "Dataset contains $EXAMPLE_COUNT examples"
echo ""

# ==============================================================================
# Run Training
# ==============================================================================
echo "============================================================"
echo "Starting Training..."
echo "============================================================"

python3 sft_train_v2.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --use_4bit \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run "$WANDB_RUN"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training Complete!"
    echo "============================================================"
    echo "Model saved to: $OUTPUT_DIR"
    echo "End time: $(date)"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate on SpaceOmicsBench: python evaluate_model.py --model $OUTPUT_DIR"
    echo "2. Evaluate on CAMELOT: python evaluate_model.py --model $OUTPUT_DIR --benchmark camelot"
    echo "3. Compare with baseline: python compare_models.py"
else
    echo ""
    echo "============================================================"
    echo "❌ Training Failed!"
    echo "============================================================"
    echo "Check the error messages above."
    exit 1
fi
