#!/bin/bash
#SBATCH --job-name=biogrpo_p4ablation
#SBATCH --partition=your_partition  # CONFIGURE: your GPU partition
#SBATCH --account=your_account      # CONFIGURE: your SLURM account
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# ============================================================
# BioGRPO Phase 4 Ablation Runner
#
# Required env vars when submitting:
#   CONFIG_PATH=<configs/*.json>
#
# Optional env vars:
#   V4_DEFAULT_MODE=legacy|match_v1   (default: match_v1)
# ============================================================

set -euo pipefail

SCRATCH="${BIORLHF_SCRATCH:?Set BIORLHF_SCRATCH to your scratch directory}"
WORKDIR="${SCRATCH}/training/BioRLHF"

CONFIG_PATH="${CONFIG_PATH:-}"
V4_DEFAULT_MODE="${V4_DEFAULT_MODE:-match_v1}"

if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: CONFIG_PATH is required."
    echo "Example:"
    echo "  sbatch --job-name=biogrpo_p4_legacy --export=ALL,CONFIG_PATH=configs/grpo_phase4_ablation_legacy.json,V4_DEFAULT_MODE=legacy scripts/run_grpo_phase4_ablation.sh"
    exit 1
fi

echo "============================================================"
echo "BioGRPO Phase 4 Ablation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working dir: $WORKDIR"
echo "Config: $CONFIG_PATH"
echo "V4 default mode: $V4_DEFAULT_MODE"
echo "Start time: $(date)"
echo "============================================================"

cd "$WORKDIR" || { echo "WORKDIR not found: $WORKDIR"; exit 1; }
mkdir -p logs

module purge
module load cuda/12.1

. "${BIORLHF_CONDA_SH:?Set BIORLHF_CONDA_SH to your conda.sh path}"
conda activate biorlhf

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="${WORKDIR}/cache/transformers"
export HF_HOME="${WORKDIR}/cache/huggingface"
export WANDB_DIR="${WORKDIR}/wandb"
export TOKENIZERS_PARALLELISM=false
export BIORLHF_V4_DEFAULT_MODE="$V4_DEFAULT_MODE"

# Data paths
export GENELAB_BASE="${SCRATCH}/data/GeneLab_benchmark"
export BIOEVAL_DATA="${SCRATCH}/data/BioEval/data"
export SPACEOMICS_DATA="${SCRATCH}/data/SpaceOmicsBench/v3/evaluation/llm"
export BIOEVAL_ROOT="${SCRATCH}/data/BioEval"

mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME" "$WANDB_DIR"

# Symlink SFT checkpoint if not already present
if [ ! -e "${WORKDIR}/kmp_sft_model_final" ]; then
    ln -s "${SCRATCH}/training/biorlhf/kmp_sft_model_final" "${WORKDIR}/kmp_sft_model_final"
    echo "Symlinked kmp_sft_model_final"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    exit 1
fi

echo "Starting BioGRPO training..."
biorlhf-grpo --config "$CONFIG_PATH"

echo ""
echo "============================================================"
echo "BioGRPO ablation training completed"
echo "End time: $(date)"
echo "============================================================"
