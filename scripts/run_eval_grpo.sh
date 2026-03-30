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

# Model paths — auto-detect MVE vs Full v2 vs checkpoint
# Allow override: GRPO_MODEL_OVERRIDE and MAX_SAMPLES env vars
if [ -n "$GRPO_MODEL_OVERRIDE" ]; then
    GRPO_MODEL="$GRPO_MODEL_OVERRIDE"
    HOLD_OUT="${HOLD_OUT_OVERRIDE:-eye thymus}"
    EVAL_TAG="checkpoint"
elif [ -d "./biogrpo_phase4_model" ]; then
    GRPO_MODEL="./biogrpo_phase4_model"
    HOLD_OUT="eye thymus"
    EVAL_TAG="phase4"
elif [ -d "./biogrpo_full_v2_model" ]; then
    GRPO_MODEL="./biogrpo_full_v2_model"
    HOLD_OUT="eye thymus"
    EVAL_TAG="full_v2"
elif [ -d "./biogrpo_mve_model" ]; then
    GRPO_MODEL="./biogrpo_mve_model"
    HOLD_OUT="eye"
    EVAL_TAG="mve"
else
    echo "ERROR: No GRPO model found"
    ls -d biogrpo_* 2>/dev/null || echo "  No biogrpo_* dirs found"
    exit 1
fi

SFT_BASELINE="./kmp_sft_model_final"
OUTPUT="results/grpo_${EVAL_TAG}_eval_$(date +%Y%m%d_%H%M%S).json"

# For full_v2/checkpoint models, GRPO adapter was trained on SFT-merged base
SFT_ADAPTER_FLAG=""
if [ "$EVAL_TAG" = "phase4" ] || [ "$EVAL_TAG" = "full_v2" ] || [ "$EVAL_TAG" = "checkpoint" ]; then
    SFT_ADAPTER_FLAG="--sft-adapter $SFT_BASELINE"
fi

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max-samples $MAX_SAMPLES"
fi

echo "GRPO model:    $GRPO_MODEL"
echo "Eval type:     $EVAL_TAG"
echo "Hold-out:      $HOLD_OUT"
echo "SFT baseline:  $SFT_BASELINE"
echo "SFT adapter:   ${SFT_ADAPTER_FLAG:-none}"
echo "Max samples:   ${MAX_SAMPLES:-all}"
echo "Output:        $OUTPUT"
echo ""

echo "Starting BioGRPO evaluation..."
python scripts/evaluate_grpo.py \
    --model "$GRPO_MODEL" \
    --sft-baseline "$SFT_BASELINE" \
    --hold-out-tissues $HOLD_OUT \
    $SFT_ADAPTER_FLAG \
    $MAX_SAMPLES_FLAG \
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
