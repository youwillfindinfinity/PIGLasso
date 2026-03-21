#!/bin/bash
#SBATCH --job-name=burn_preprocess
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Go to project root (assumes script is in slurm/)
cd "$(dirname "$0")/.."

mkdir -p logs

# --- Load modules (adjust to Snellius setup) ---
# module purge
# module load 2023
# module load Python/3.10.4
# module load R/4.3.1

# --- Activate venv ---
source .venv/bin/activate

echo "[INFO] Running preprocess_burn..."
python3 monika/preprocess_burn.py

echo "[INFO] Running build_burn_design..."
python3 monika/build_burn_design.py

echo "[INFO] Running prepare_piglasso_inputs..."
python3 monika/prepare_piglasso_inputs.py

echo "[DONE] Preprocessing pipeline finished."