#!/bin/bash
#SBATCH --job-name=biogrpo_mve
#SBATCH --partition=scu-gpu
#SBATCH --account=cayuga_0003
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/grpo_mve_%j.log
#SBATCH --error=logs/grpo_mve_%j.err

# ============================================================
# BioGRPO Minimum Viable Experiment (MVE)
# V1+V4 verifiers, G=4, from SFT checkpoint
# ============================================================

SCRATCH="/athena/cayuga_0003/scratch/users/jak4013/otsuka"
WORKDIR="${SCRATCH}/training/BioRLHF"

echo "============================================================"
echo "BioGRPO MVE Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working dir: $WORKDIR"
echo "Start time: $(date)"
echo "============================================================"

cd "$WORKDIR" || { echo "WORKDIR not found: $WORKDIR"; exit 1; }
mkdir -p logs

# Load modules
module purge
module load cuda/12.1

# Activate conda environment
. /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate biorlhf

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
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

# Run GRPO MVE training
echo "Starting BioGRPO MVE training..."
biorlhf-grpo --config configs/grpo_mve.json

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "BioGRPO MVE training completed!"
    echo "Model saved to: ./biogrpo_mve_model"
    echo "End time: $(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "BioGRPO MVE training failed with exit code $?"
    echo "Check logs/grpo_mve_${SLURM_JOB_ID}.err for details"
    echo "============================================================"
    exit 1
fi
