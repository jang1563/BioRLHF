#!/bin/bash
#SBATCH --job-name=biogrpo_full
#SBATCH --partition=scu-gpu
#SBATCH --account=cayuga_0003
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/grpo_full_%j.log
#SBATCH --error=logs/grpo_full_%j.err

# ============================================================
# BioGRPO Full Experiment
# All V1-V4 verifiers, G=8, from SFT checkpoint
# ============================================================

SCRATCH="/athena/cayuga_0003/scratch/users/jak4013/otsuka"
WORKDIR="${SCRATCH}/training/BioRLHF"

echo "============================================================"
echo "BioGRPO Full Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working dir: $WORKDIR"
echo "Start time: $(date)"
echo "============================================================"

cd "$WORKDIR" || { echo "WORKDIR not found: $WORKDIR"; exit 1; }
mkdir -p logs

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

echo "Starting BioGRPO Full training..."
biorlhf-grpo --config configs/grpo_full.json

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "BioGRPO Full training completed!"
    echo "Model saved to: ./biogrpo_full_model"
    echo "End time: $(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "BioGRPO Full training failed with exit code $?"
    echo "Check logs/grpo_full_${SLURM_JOB_ID}.err for details"
    echo "============================================================"
    exit 1
fi
