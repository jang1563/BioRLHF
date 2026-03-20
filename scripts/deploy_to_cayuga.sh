#!/bin/bash
# ============================================================
# Deploy BioRLHF code + data to Cayuga HPC
# Run from local Mac
# ============================================================

set -e

REMOTE="cayuga-login1"
SCRATCH="/athena/cayuga_0003/scratch/users/jak4013/otsuka"
LOCAL_BASE="$HOME/Dropbox/Bioinformatics/Claude"

echo "============================================================"
echo "BioRLHF Cayuga Deployment"
echo "============================================================"

# Step 1: Create directories on Cayuga
echo ""
echo "[1/4] Creating directories on Cayuga..."
ssh ${REMOTE} "mkdir -p ${SCRATCH}/training/BioRLHF ${SCRATCH}/data/GeneLab_benchmark ${SCRATCH}/data/BioEval ${SCRATCH}/data/SpaceOmicsBench/v3/evaluation"

# Step 2: Transfer BioRLHF code (only essential files)
echo ""
echo "[2/4] Transferring BioRLHF code..."
LOCAL_BIORLHF="${LOCAL_BASE}/BioRLHF/biorlhf"
DEST="${REMOTE}:${SCRATCH}/training/BioRLHF"

# Transfer only the package structure needed for GRPO
rsync -avz --progress \
    "${LOCAL_BIORLHF}/src/" \
    ${DEST}/src/

rsync -avz --progress \
    "${LOCAL_BIORLHF}/configs/" \
    ${DEST}/configs/

rsync -avz --progress \
    "${LOCAL_BIORLHF}/scripts/" \
    ${DEST}/scripts/

rsync -avz --progress \
    "${LOCAL_BIORLHF}/tests/" \
    ${DEST}/tests/

rsync -avz --progress \
    "${LOCAL_BIORLHF}/pyproject.toml" \
    "${LOCAL_BIORLHF}/README.md" \
    ${DEST}/

# Step 3: Transfer data (only what GRPO training needs)
echo ""
echo "[3/4] Transferring data..."

echo "  GeneLab fgsea (pathway enrichment scores - required)..."
rsync -avz --progress \
    "${LOCAL_BASE}/GeneLab_benchmark/processed/fgsea/" \
    ${REMOTE}:${SCRATCH}/data/GeneLab_benchmark/processed/fgsea/

echo "  GeneLab evaluation (NES conservation - for conservation questions)..."
rsync -avz --progress \
    "${LOCAL_BASE}/GeneLab_benchmark/evaluation/" \
    ${REMOTE}:${SCRATCH}/data/GeneLab_benchmark/evaluation/

echo "  BioEval data..."
rsync -avz --progress \
    "${LOCAL_BASE}/Evaluation_model/BioEval/data/" \
    ${REMOTE}:${SCRATCH}/data/BioEval/data/

echo "  BioEval scoring (for calibration imports)..."
rsync -avz --progress \
    "${LOCAL_BASE}/Evaluation_model/BioEval/bioeval/" \
    ${REMOTE}:${SCRATCH}/data/BioEval/bioeval/

echo "  SpaceOmicsBench..."
rsync -avz --progress \
    "${LOCAL_BASE}/SpaceOmicsBench/v3/evaluation/llm/" \
    ${REMOTE}:${SCRATCH}/data/SpaceOmicsBench/v3/evaluation/llm/

# Step 4: Verify
echo ""
echo "[4/4] Verifying deployment..."
ssh ${REMOTE} "
echo 'Directory structure:'
echo '  BioRLHF code:'
ls ${SCRATCH}/training/BioRLHF/pyproject.toml 2>/dev/null && echo '    pyproject.toml: OK' || echo '    pyproject.toml: MISSING'
ls ${SCRATCH}/training/BioRLHF/configs/grpo_mve.json 2>/dev/null && echo '    configs/grpo_mve.json: OK' || echo '    configs/grpo_mve.json: MISSING'
ls -d ${SCRATCH}/training/BioRLHF/src/biorlhf/ 2>/dev/null && echo '    src/biorlhf/: OK' || echo '    src/biorlhf/: MISSING'

echo '  SFT checkpoint:'
ls -d ${SCRATCH}/training/biorlhf/kmp_sft_model_final/ 2>/dev/null && echo '    kmp_sft_model_final: OK' || echo '    kmp_sft_model_final: MISSING'

echo '  Data:'
ls ${SCRATCH}/data/GeneLab_benchmark/processed/fgsea/ 2>/dev/null | head -3 && echo '    GeneLab fgsea: OK' || echo '    GeneLab fgsea: MISSING'
ls ${SCRATCH}/data/GeneLab_benchmark/evaluation/ 2>/dev/null | head -3 && echo '    GeneLab evaluation: OK' || echo '    GeneLab evaluation: MISSING'
ls ${SCRATCH}/data/BioEval/data/ 2>/dev/null | head -3 && echo '    BioEval: OK' || echo '    BioEval: MISSING'
ls ${SCRATCH}/data/SpaceOmicsBench/v3/evaluation/llm/ 2>/dev/null | head -3 && echo '    SpaceOmicsBench: OK' || echo '    SpaceOmicsBench: MISSING'
"

echo ""
echo "============================================================"
echo "Deployment complete!"
echo ""
echo "Next steps on Cayuga:"
echo "  ssh ${REMOTE}"
echo "  cd ${SCRATCH}/training/BioRLHF"
echo "  bash scripts/setup_cayuga_grpo.sh"
echo "  sbatch scripts/run_grpo_mve.sh"
echo "============================================================"
