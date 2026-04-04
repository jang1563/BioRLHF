#!/bin/bash
#SBATCH --job-name=biogrpo_phase4
#SBATCH --partition=your_partition  # CONFIGURE: your GPU partition
#SBATCH --account=your_account      # CONFIGURE: your SLURM account
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/grpo_phase4_%j.log
#SBATCH --error=logs/grpo_phase4_%j.err

# ============================================================
# BioGRPO Phase 4: V1-Aware V4 Calibration Fix
# V4 weight=0.45 (dominant), V1-aware confidence targeting
# ============================================================

SCRATCH="${BIORLHF_SCRATCH:?Set BIORLHF_SCRATCH to your scratch directory}"
WORKDIR="${SCRATCH}/training/BioRLHF"

echo "============================================================"
echo "BioGRPO Phase 4 Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working dir: $WORKDIR"
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

# Data paths
export GENELAB_BASE="${SCRATCH}/data/GeneLab_benchmark"
export BIOEVAL_DATA="${SCRATCH}/data/BioEval/data"
export SPACEOMICS_DATA="${SCRATCH}/data/SpaceOmicsBench/v3/evaluation/llm"
export BIOEVAL_ROOT="${SCRATCH}/data/BioEval"

mkdir -p $TRANSFORMERS_CACHE $HF_HOME $WANDB_DIR

# Symlink SFT checkpoint if not already present
if [ ! -e "${WORKDIR}/kmp_sft_model_final" ]; then
    ln -s "${SCRATCH}/training/biorlhf/kmp_sft_model_final" "${WORKDIR}/kmp_sft_model_final"
    echo "Symlinked kmp_sft_model_final"
fi

echo "Starting BioGRPO Phase 4 training..."
biorlhf-grpo --config configs/grpo_phase4.json

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "BioGRPO Phase 4 training completed!"
    echo "Model saved to: ./biogrpo_phase4_model"
    echo "End time: $(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "BioGRPO Phase 4 training failed with exit code $?"
    echo "Check logs/grpo_phase4_${SLURM_JOB_ID}.err for details"
    echo "============================================================"
    exit 1
fi
