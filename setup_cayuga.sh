#!/bin/bash
# ============================================================
# BioRLHF Setup Script for Cayuga HPC
# Run this once to set up the environment
# ============================================================

echo "============================================================"
echo "BioRLHF Environment Setup"
echo "============================================================"

# Create conda environment
echo ""
echo "Step 1: Creating conda environment..."
conda create -n biorlhf python=3.10 -y

# Activate environment
echo ""
echo "Step 2: Activating environment..."
source ~/.bashrc
conda activate biorlhf

# Install PyTorch with CUDA
echo ""
echo "Step 3: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install main dependencies
echo ""
echo "Step 4: Installing dependencies..."
pip install transformers datasets accelerate peft trl bitsandbytes
pip install wandb pandas numpy scikit-learn scipy tqdm jsonlines
pip install matplotlib seaborn

# Try to install flash-attn (may fail on some systems)
echo ""
echo "Step 5: Attempting flash-attn installation (optional)..."
pip install flash-attn --no-build-isolation || echo "Flash attention installation failed (optional)"

# Login to services
echo ""
echo "Step 6: Service logins..."
echo "Please run these commands manually:"
echo "  wandb login"
echo "  huggingface-cli login"

# Create directories
echo ""
echo "Step 7: Creating directories..."
mkdir -p logs cache/transformers cache/huggingface wandb

# Verify installation
echo ""
echo "Step 8: Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Login to Weights & Biases: wandb login"
echo "2. Login to Hugging Face: huggingface-cli login"
echo "3. Submit training job: sbatch run_sft.sh"
echo "============================================================"
