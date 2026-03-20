#!/bin/bash
#SBATCH --job-name=eval_grpo
#SBATCH --partition=scu-gpu
#SBATCH --account=cayuga_0003
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_grpo_%j.log
#SBATCH --error=logs/eval_grpo_%j.err

# ============================================================
# BioGRPO Post-Training Evaluation
# Evaluates GRPO model + SFT baseline comparison
# ============================================================

SCRATCH="/athena/cayuga_0003/scratch/users/jak4013/otsuka"
WORKDIR="${SCRATCH}/training/BioRLHF"

echo "============================================================"
echo "BioGRPO Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working dir: $WORKDIR"
echo "Start time: $(date)"
echo "============================================================"

cd "$WORKDIR" || { echo "WORKDIR not found: $WORKDIR"; exit 1; }
mkdir -p logs results

module purge
module load cuda/12.1

. /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate biorlhf

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="${WORKDIR}/cache/transformers"
export HF_HOME="${WORKDIR}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# Data paths
export GENELAB_BASE="${SCRATCH}/data/GeneLab_benchmark"
export BIOEVAL_DATA="${SCRATCH}/data/BioEval/data"
export SPACEOMICS_DATA="${SCRATCH}/data/SpaceOmicsBench/v3/evaluation/llm"
export BIOEVAL_ROOT="${SCRATCH}/data/BioEval"

# Model paths
GRPO_MODEL="./biogrpo_mve_model"
SFT_BASELINE="./kmp_sft_model_final"
OUTPUT="results/grpo_mve_eval_$(date +%Y%m%d_%H%M%S).json"

echo "GRPO model:    $GRPO_MODEL"
echo "SFT baseline:  $SFT_BASELINE"
echo "Output:        $OUTPUT"
echo ""

# Check model exists
if [ ! -d "$GRPO_MODEL" ]; then
    echo "ERROR: GRPO model not found at $GRPO_MODEL"
    echo "Available directories:"
    ls -d biogrpo_* 2>/dev/null || echo "  No biogrpo_* dirs found"
    exit 1
fi

echo "Starting BioGRPO evaluation..."
python scripts/evaluate_grpo.py \
    --model "$GRPO_MODEL" \
    --sft-baseline "$SFT_BASELINE" \
    --hold-out-tissues eye \
    --output "$OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "BioGRPO evaluation completed!"
    echo "Results: $OUTPUT"
    echo "End time: $(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "BioGRPO evaluation failed with exit code $?"
    echo "Check logs/eval_grpo_${SLURM_JOB_ID}.err for details"
    echo "============================================================"
    exit 1
fi
