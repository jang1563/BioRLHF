# BioRLHF Training on HPC

**Target:** GPU training with Mistral-7B + LoRA (SFT, DPO, GRPO)

---

## Configuration

Before using the scripts, set these environment variables (e.g., in your `.bashrc` or SLURM job preamble):

```bash
# Required
export BIORLHF_SCRATCH="/path/to/your/scratch/directory"
export BIORLHF_CONDA_SH="/path/to/your/conda.sh"

# Optional (for deploy_to_cayuga.sh)
export BIORLHF_HPC_HOST="your-hpc-login-node"
export BIORLHF_LOCAL_BASE="$HOME/BioRLHF-data"
```

Also update `#SBATCH --partition` and `#SBATCH --account` in the SLURM scripts to match your cluster.

---

## Quick Start

```bash
# 1. SSH to your HPC
ssh $USER@your-hpc-login-node

# 2. Submit a GRPO training job
bash -l -c 'sbatch scripts/run_grpo_full.sh'

# 3. Monitor
squeue -u $USER
tail -f logs/grpo_full_*.log
```

---

## Step 1: Transfer Files to HPC

From your local machine:

```bash
rsync -avz --progress \
    /path/to/BioRLHF/biorlhf/ \
    $USER@your-hpc-login-node:$BIORLHF_SCRATCH/training/BioRLHF/
```

---

## Step 2: Set Up Conda Environment (First Time Only)

```bash
# SSH to your HPC
ssh $USER@your-hpc-login-node

# Source conda (non-interactive shell requires explicit sourcing)
. $BIORLHF_CONDA_SH

# Create environment
conda create -n biorlhf python=3.10 -y
conda activate biorlhf

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install training dependencies
pip install transformers>=4.36.0 peft>=0.6.0 trl>=0.14.0
pip install bitsandbytes>=0.41.0 accelerate>=0.24.0 datasets>=2.14.0
pip install wandb scipy scikit-learn sentencepiece jsonlines

# Verify GPU access (on a GPU node)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Step 3: Training Options

### Option A: GRPO Training (Recommended)

GRPO with verifier-based multi-reward training from an SFT checkpoint:

```bash
# Submit via SLURM (use login shell for correct sbatch version)
bash -l -c 'sbatch scripts/run_grpo_full.sh'
```

**Key config** (`configs/grpo_full_v2.json`):
- G=16 generations per prompt
- V1-V4 verifiers with weights [0.30, 0.15, 0.10, 0.45]
- beta=0.02, 2 iterations per batch
- ~48h on A40

### Option B: SFT Training

```bash
# Interactive session (update partition/account for your cluster)
srun -p your_partition --gres=gpu:1 --mem=48G -c 8 --time=4:00:00 --account=your_account --pty bash

# Activate environment
. $BIORLHF_CONDA_SH
conda activate biorlhf

# Run SFT
cd $BIORLHF_SCRATCH/training/BioRLHF
biorlhf-train --model mistralai/Mistral-7B-v0.3 --dataset data/kmp_sft_final.json --output ./my_sft_model
```

### Option C: Interactive GPU Session

```bash
# Request GPU (update partition/account for your cluster)
srun -p your_partition --gres=gpu:1 --mem=48G -c 8 --time=4:00:00 --account=your_account --pty bash

# Activate environment
. $BIORLHF_CONDA_SH
conda activate biorlhf

# Navigate and run
cd $BIORLHF_SCRATCH/training/BioRLHF
biorlhf-grpo --config configs/grpo_full_v2.json
```

---

## Step 4: Monitor Training

```bash
# Check job status
squeue -u $USER

# Tail logs
tail -f logs/grpo_full_*.log

# GPU usage (on compute node)
nvidia-smi

# WandB dashboard (if configured)
# https://wandb.ai/your-workspace/biogrpo
```

---

## Environment Details

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.57.3 |
| TRL | 0.26.2 |
| PEFT | 0.18.0 |

---

## GPU Options

| GPU | VRAM | Best For | SLURM Flag |
|-----|------|----------|------------|
| A40 | 48GB | Standard GRPO/SFT with QLoRA | `--gres=gpu:1` |
| A100 | 80GB | Larger batches, faster training | `--gres=gpu:a100:1` |

---

## Important Notes

### SLURM Version

Some clusters have an outdated default `sbatch`. Use `bash -l -c 'sbatch ...'` to ensure modules are loaded and the correct SLURM version is used.

### Conda in Non-Interactive Shells

`source ~/.bashrc` does not work in non-interactive SSH. Always source conda directly:
```bash
. $BIORLHF_CONDA_SH
conda activate biorlhf
```

### SFT Checkpoint Symlink

The SFT model adapter should be available at `$BIORLHF_SCRATCH/training/biorlhf/kmp_sft_model_final`. GRPO scripts auto-symlink this into the working directory.

### Batch Size with G=16

Both `per_device_eval_batch_size` and `generation_batch_size` must be divisible by `num_generations`. The TRL parameter is `generation_batch_size`, NOT `per_device_generation_batch_size`.

### Eval Performance

GRPOTrainer's eval loop generates completions sequentially (~3 min/sample). With 107 eval samples, each eval pass takes ~5.3h. Set `eval_steps=9999` to skip in-training eval; run post-hoc evaluation instead.

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size or gradient accumulation in the config JSON:
```json
{
    "batch_size": 1,
    "gradient_accumulation_steps": 16
}
```

### "No GPU available"
```bash
nvidia-smi                    # Check GPU allocation
squeue -u $USER               # Verify you're on a GPU node
```

### LoRA adapter loading fails
The SFT checkpoint is a LoRA adapter, not a full model. Load base model first:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")
model = PeftModel.from_pretrained(base, "path/to/kmp_sft_model_final")
model = model.merge_and_unload()  # Merge for GRPO training
```

---

## Key Paths

| Path | Description |
|------|-------------|
| `$BIORLHF_SCRATCH/training/BioRLHF/` | Working directory |
| `$BIORLHF_SCRATCH/training/biorlhf/kmp_sft_model_final` | SFT checkpoint |
| `$BIORLHF_SCRATCH/data/` | Data directory |
| `$BIORLHF_CONDA_SH` | Conda init script |
