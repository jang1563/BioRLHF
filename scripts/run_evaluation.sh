#!/bin/bash
#
# BioRLHF Model Evaluation Script
# ================================
#
# Evaluates the ecosystem-improved model on:
# - Calibration (uncertainty expression)
# - Adversarial resistance
# - Protocol completeness
# - Fact recall
#
# Usage on HPC:
#   srun -p scu-gpu --gres=gpu:a100:1 --mem=48G -c 8 --time=1:00:00 --pty bash
#   conda activate biorlhf
#   ./scripts/run_evaluation.sh
#

echo "============================================================"
echo "BioRLHF Ecosystem Model Evaluation"
echo "============================================================"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo ""

# Set working directory
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# Configuration
MODEL_PATH="./ecosystem_improved_model"
TEST_DATA="data/ecosystem_failures_training.json"
OUTPUT="ecosystem_eval_results_$(date +%Y%m%d_%H%M%S).json"

echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "Model:     $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Output:    $OUTPUT"
echo ""

# Check files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data not found at $TEST_DATA"
    exit 1
fi

# Run evaluation
echo "============================================================"
echo "Starting Evaluation..."
echo "============================================================"

python3 scripts/evaluate_ecosystem_model.py \
    --model "$MODEL_PATH" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Evaluation Complete!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT"
    echo "End time: $(date)"
else
    echo ""
    echo "============================================================"
    echo "Evaluation Failed!"
    echo "============================================================"
    exit 1
fi
