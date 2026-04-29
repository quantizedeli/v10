#!/bin/bash
#SBATCH --job-name=v10_training
#SBATCH --output=logs/v10_%j.out
#SBATCH --error=logs/v10_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # Adjust to your HPC node (check: sinfo -o "%n %c")
#SBATCH --mem=128G                   # Adjust: 64G minimum, 128G recommended
#SBATCH --gres=gpu:1                 # 1 GPU (remove line if no GPU partition available)
#SBATCH --partition=gpu              # Partition name: check with "sinfo" on your cluster

# ============================================================
# Nuclear Physics AI Project — SLURM Job Script
# Version: 1.0.0
# Usage:
#   sbatch hpc_slurm_job.sh              (full pipeline)
#   sbatch hpc_slurm_job.sh --pfaz 2     (single phase)
# ============================================================

echo "========================================================"
echo " Job ID       : $SLURM_JOB_ID"
echo " Job Name     : $SLURM_JOB_NAME"
echo " Node         : $SLURMD_NODENAME"
echo " CPUs         : $SLURM_CPUS_PER_TASK"
echo " Start time   : $(date)"
echo "========================================================"

# --- Load modules (adjust to your HPC's module system) ---
module purge
module load python/3.11           # Required
module load cuda/12.1             # Optional: GPU support for XGBoost/TF/PyTorch
# module load matlab/R2023b       # Optional: ANFIS MATLAB backend

# --- Virtual environment ---
VENV_PATH="$HOME/v10_env"
if [ ! -d "$VENV_PATH" ]; then
    echo "[SETUP] Creating virtual environment at $VENV_PATH ..."
    python -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip
    pip install -r requirements-hpc.txt
else
    source "$VENV_PATH/bin/activate"
fi
echo "[OK] Virtual environment activated: $VENV_PATH"

# --- Nested parallelism defense (CRITICAL for stability) ---
export OMP_NUM_THREADS=1           # OpenMP threads (NumPy/SciPy internal)
export MKL_NUM_THREADS=1           # Intel MKL (NumPy backend on Intel systems)
export OPENBLAS_NUM_THREADS=1      # OpenBLAS (NumPy backend on most clusters)
export NUMEXPR_NUM_THREADS=1       # numexpr
export VECLIB_MAXIMUM_THREADS=1    # macOS Accelerate (unused on Linux, harmless)

# --- GPU settings ---
export TF_CPP_MIN_LOG_LEVEL=2      # Suppress TensorFlow info spam
export TF_FORCE_GPU_ALLOW_GROWTH=true   # Don't allocate all GPU memory at once
export CUDA_VISIBLE_DEVICES=0      # Use first GPU

# --- HPC-mode flags (disables interactive prompts, pip auto-install, etc.) ---
export HPC_MODE=1
export PARALLEL_TRAINING=1         # Skip input() in parallel training prompt

# --- Thesis metadata (avoids input() prompts in PFAZ 10) ---
export THESIS_AUTHOR="Your Name"
export THESIS_SUPERVISOR="Prof. Supervisor"
export THESIS_UNIVERSITY="Your University"
export THESIS_DEPARTMENT="Physics"
export THESIS_COMPILE_PDF="n"

# --- I/O: use fast scratch for intermediate results ---
# SCRATCH_DIR=${TMPDIR:-${SCRATCH:-/tmp}}
# export PFAZ_OUTPUT_DIR="$SCRATCH_DIR/v10_output_$SLURM_JOB_ID"
# mkdir -p "$PFAZ_OUTPUT_DIR"

# --- Move to submission directory ---
cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: Cannot cd to submit dir"; exit 1; }

# Create logs dir if needed
mkdir -p logs

# --- Run pipeline ---
# -u = unbuffered output (real-time log on HPC)
echo "[START] Running pipeline at $(date)"

if [ $# -gt 0 ]; then
    # Pass arguments through (e.g. --pfaz 2)
    python -u main.py "$@" 2>&1 | tee "logs/run_$SLURM_JOB_ID.log"
else
    python -u main.py --run-all 2>&1 | tee "logs/run_$SLURM_JOB_ID.log"
fi

EXIT_CODE=$?
echo "========================================================"
echo " Job $SLURM_JOB_ID finished at $(date)"
echo " Exit code: $EXIT_CODE"
echo "========================================================"

exit $EXIT_CODE
