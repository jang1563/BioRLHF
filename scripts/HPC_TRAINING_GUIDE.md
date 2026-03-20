# BioRLHF Training on Cayuga HPC (Interactive Session)

**Cluster:** Cornell Cayuga HPC
**Target:** GPU training with Mistral-7B + LoRA

---

## Quick Start (Copy-Paste Commands)

```bash
# 1. Start interactive GPU session (A100 recommended, 80GB VRAM)
srun -p scu-gpu --gres=gpu:a100:1 --mem=48G -c 8 --time=4:00:00 --pty bash

# 2. Set up environment (first time only - see Step 2 below)

# 3. Run training
cd /athena/cayuga_XXXX/scratch/$USER/BioRLHF/biorlhf
./scripts/train_ecosystem_improved.sh
```

---

## Step 1: Transfer Files to HPC

From your local Mac:

```bash
# Replace with your actual paths and CWID
rsync -avz --progress \
    /Users/jak4013/Dropbox/Bioinformatics/Claude/BioRLHF \
    YOUR_CWID@cayuga.cac.cornell.edu:/athena/cayuga_XXXX/scratch/$USER/
```

Or use scp:
```bash
scp -r /Users/jak4013/Dropbox/Bioinformatics/Claude/BioRLHF \
    YOUR_CWID@cayuga.cac.cornell.edu:/athena/cayuga_XXXX/scratch/$USER/
```

---

## Step 2: Set Up Conda Environment (First Time Only)

### 2a. Start Interactive Session
```bash
# SSH to Cayuga
ssh YOUR_CWID@cayuga.cac.cornell.edu

# Request interactive GPU session
srun -p scu-gpu --gres=gpu:a100:1 --mem=48G -c 8 --time=2:00:00 --pty bash
```

### 2b. Install Miniconda (if not already installed)
```bash
# Create directory in scratch space
mkdir -p /athena/cayuga_XXXX/scratch/$USER/miniconda3

# Download and install
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -u -p /athena/cayuga_XXXX/scratch/$USER/miniconda3
rm miniconda.sh

# Initialize conda
source /athena/cayuga_XXXX/scratch/$USER/miniconda3/bin/activate
conda init bash
source ~/.bashrc
```

### 2c. Create BioRLHF Environment
```bash
# Create environment with Python 3.10 (best compatibility)
conda create -n biorlhf python=3.10 -y
conda activate biorlhf

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install training dependencies
pip install transformers>=4.36.0
pip install peft>=0.7.0
pip install trl>=0.7.0
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.25.0
pip install datasets>=2.14.0
pip install wandb
pip install scipy
pip install sentencepiece

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Step 3: Run Training (Interactive)

### 3a. Start GPU Session
```bash
# Request A100 GPU (80GB - best for Mistral-7B)
srun -p scu-gpu --gres=gpu:a100:1 --mem=48G -c 8 --time=4:00:00 --pty bash

# Or use A40 (48GB - also works with 4-bit quantization)
srun -p scu-gpu --gres=gpu:a40:1 --mem=48G -c 8 --time=4:00:00 --pty bash
```

### 3b. Activate Environment and Run
```bash
# Activate conda
source /athena/cayuga_XXXX/scratch/$USER/miniconda3/bin/activate
conda activate biorlhf

# Navigate to BioRLHF
cd /athena/cayuga_XXXX/scratch/$USER/BioRLHF/biorlhf

# Check GPU is available
nvidia-smi

# Set HuggingFace cache (optional - saves space)
export HF_HOME=/athena/cayuga_XXXX/scratch/$USER/.cache/huggingface

# Run training
./scripts/train_ecosystem_improved.sh
```

---

## Step 4: Monitor Training

In a separate terminal (or use tmux/screen):

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Tail training logs
tail -f logs/biorlhf_ecosystem_*.out
```

### Using WandB (Optional)
```bash
# Login to Weights & Biases
wandb login

# Training will automatically log to: https://wandb.ai/YOUR_USERNAME/biorlhf
```

---

## GPU Options on Cayuga

| GPU Type | VRAM | Recommended For | Command |
|----------|------|-----------------|---------|
| A100 | 80GB | Full training, larger batches | `--gres=gpu:a100:1` |
| A40 | 48GB | Standard training with 4-bit | `--gres=gpu:a40:1` |
| H100 | 80GB | Fastest (if available) | `--gres=gpu:h100:1` |

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size in training script:
```bash
# Edit train_ecosystem_improved.sh
BATCH_SIZE=2   # Reduce from 4
GRAD_ACCUM=8   # Increase to maintain effective batch size
```

### "No GPU available"
```bash
# Check GPU allocation
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check if you're on a GPU node
squeue -u $USER
```

### "Module not found"
```bash
# Ensure conda environment is activated
conda activate biorlhf

# Reinstall missing package
pip install <missing_package>
```

### Interactive session times out
Use `tmux` or `screen` to persist sessions:
```bash
# Start tmux before srun
tmux new -s training

# Then request GPU
srun -p scu-gpu --gres=gpu:a100:1 --mem=48G -c 8 --time=4:00:00 --pty bash

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

---

## Expected Training Time

| Configuration | Dataset Size | Estimated Time |
|--------------|--------------|----------------|
| A100 + 4-bit | 378 examples, 10 epochs | ~45-60 min |
| A40 + 4-bit | 378 examples, 10 epochs | ~60-90 min |
| A100 (full) | 378 examples, 10 epochs | ~90-120 min |

---

## After Training

### Copy model back to local machine:
```bash
# From your Mac
scp -r YOUR_CWID@cayuga.cac.cornell.edu:/athena/cayuga_XXXX/scratch/$USER/BioRLHF/biorlhf/ecosystem_improved_model \
    /Users/jak4013/Dropbox/Bioinformatics/Claude/BioRLHF/biorlhf/
```

### Run evaluation:
```bash
python evaluate_model.py --model ecosystem_improved_model
```

---

## Complete Interactive Session Example

```bash
# SSH to Cayuga
ssh jk2042@cayuga.cac.cornell.edu

# Start tmux (optional but recommended)
tmux new -s biorlhf

# Request GPU
srun -p scu-gpu --gres=gpu:a100:1 --mem=48G -c 8 --time=4:00:00 --pty bash

# Set up environment
source ~/miniconda3/bin/activate
conda activate biorlhf

# Navigate and run
cd /athena/cayuga_XXXX/scratch/$USER/BioRLHF/biorlhf
./scripts/train_ecosystem_improved.sh

# Watch progress (in another terminal or after Ctrl+B, c for new window)
watch -n 5 nvidia-smi
```
