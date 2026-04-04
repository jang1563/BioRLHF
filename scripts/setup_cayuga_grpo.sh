#!/bin/bash
# ============================================================
# BioGRPO Environment Setup for Cayuga HPC
# Run once to verify/upgrade GRPO dependencies
# ============================================================

SCRATCH="${BIORLHF_SCRATCH:?Set BIORLHF_SCRATCH to your scratch directory}"
WORKDIR="${SCRATCH}/training/BioRLHF"

echo "============================================================"
echo "BioGRPO Environment Setup"
echo "Working dir: $WORKDIR"
echo "============================================================"

cd "$WORKDIR" || { echo "WORKDIR not found: $WORKDIR"; exit 1; }

# Activate environment
. "${BIORLHF_CONDA_SH:?Set BIORLHF_CONDA_SH to your conda.sh path}"
conda activate biorlhf

# Step 1: Check current versions
echo ""
echo "[1/6] Current package versions..."
python -c "import trl; print(f'  TRL: {trl.__version__}')"
python -c "import peft; print(f'  PEFT: {peft.__version__}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"

# Step 2: Upgrade TRL if needed
echo ""
echo "[2/6] Ensuring TRL >= 0.26.0..."
pip install "trl>=0.26.0" --upgrade --quiet

# Step 3: Verify GRPO imports
echo ""
echo "[3/6] Verifying GRPO imports..."
python -c "
from trl import GRPOTrainer, GRPOConfig
print('  GRPOTrainer: OK')
print('  GRPOConfig: OK')
config = GRPOConfig(output_dir='/tmp/test', scale_rewards='group', loss_type='grpo')
print(f'  scale_rewards={config.scale_rewards}, loss_type={config.loss_type}: OK')
"

# Step 4: Install biorlhf package
echo ""
echo "[4/6] Installing biorlhf package..."
pip install -e . --quiet 2>/dev/null || pip install -e . 2>&1 | tail -3

# Step 5: Verify biorlhf imports
echo ""
echo "[5/6] Verifying biorlhf imports..."
python -c "
from biorlhf.training.grpo import BioGRPOConfig, run_grpo_training
print('  BioGRPOConfig: OK')
from biorlhf.verifiers.composer import make_grpo_reward_function
print('  make_grpo_reward_function: OK')
from biorlhf.data.grpo_dataset import build_grpo_dataset
print('  build_grpo_dataset: OK')
from biorlhf.evaluation.calibration import compute_calibration_metrics
print('  compute_calibration_metrics: OK')
"

# Step 6: Smoke test
echo ""
echo "[6/6] Running smoke test..."
python -c "
from biorlhf.verifiers.composer import make_grpo_reward_function
import json
reward_fn = make_grpo_reward_function(active_verifiers=['V1', 'V4'])
rewards = reward_fn(
    completions=['Oxidative phosphorylation is upregulated. Confidence: high.'],
    ground_truth=[json.dumps({
        'pathway': 'HALLMARK_OXIDATIVE_PHOSPHORYLATION',
        'direction': 'UP',
        'expected_confidence': 'high',
    })],
    question_type=['direction'],
    applicable_verifiers=[json.dumps(['V1', 'V4'])],
)
print(f'  Reward: {rewards[0]:.3f} (expected > 0.5)')
assert rewards[0] > 0.3, 'Reward too low'
print('  Smoke test: PASSED')
"

# Create directories
mkdir -p logs configs results cache/transformers cache/huggingface wandb

# Step 6b: Symlink SFT checkpoint
echo ""
echo "[6b/7] Setting up SFT checkpoint symlink..."
if [ ! -e "${WORKDIR}/kmp_sft_model_final" ]; then
    if [ -d "${SCRATCH}/training/biorlhf/kmp_sft_model_final" ]; then
        ln -s "${SCRATCH}/training/biorlhf/kmp_sft_model_final" "${WORKDIR}/kmp_sft_model_final"
        echo "  Symlinked kmp_sft_model_final: OK"
    else
        echo "  WARNING: kmp_sft_model_final not found at ${SCRATCH}/training/biorlhf/"
        echo "  You will need to provide the SFT checkpoint manually"
    fi
else
    echo "  kmp_sft_model_final already exists: OK"
fi

# Step 7: Verify data paths
echo ""
echo "[7/7] Verifying data availability..."
export GENELAB_BASE="${SCRATCH}/data/GeneLab_benchmark"
export BIOEVAL_DATA="${SCRATCH}/data/BioEval/data"
export SPACEOMICS_DATA="${SCRATCH}/data/SpaceOmicsBench/v3/evaluation/llm"
export BIOEVAL_ROOT="${SCRATCH}/data/BioEval"

for d in "$GENELAB_BASE" "$BIOEVAL_DATA" "$SPACEOMICS_DATA" "$BIOEVAL_ROOT"; do
    if [ -d "$d" ]; then
        echo "  $d: OK"
    else
        echo "  $d: MISSING"
    fi
done

echo ""
echo "============================================================"
echo "BioGRPO setup complete!"
echo ""
echo "Next steps:"
echo "  sbatch scripts/run_grpo_mve.sh"
echo "  tail -f logs/grpo_mve_*.log"
echo "============================================================"
