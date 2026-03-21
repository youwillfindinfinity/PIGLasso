#!/bin/bash
#SBATCH --job-name=run_pig
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home2/zblei/Documents/BurnInjuries
mkdir -p logs

module purge
module load 2024
module load R/4.4.2-gfbf-2024a

source /gpfs/home2/zblei/Documents/BurnInjuries/.venv/bin/activate

INPUT_DIR="burn_data/preprocessed/filtered"

shopt -s nullglob   # empty glob 

FILES=("$INPUT_DIR"/*_filtered.tsv)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "[ERROR] No *_filtered.tsv files found in ${INPUT_DIR}"
  exit 1
fi

for FPATH in "${FILES[@]}"; do
  FBASE="$(basename "$FPATH" .tsv)"

  echo "[INFO] ========================================"
  echo "[INFO] Running PIGLASSO for: $FBASE"
  echo "[INFO] Full path: $FPATH"
  echo "[INFO] Using ${SLURM_CPUS_PER_TASK} CPU cores"
  echo "[INFO] ========================================"

  python3 monika/run_piglasso.py \
    --input "$FPATH" \
    --Q 200 \
    --lamlen 20 \
    --llo 0.05 \
    --lhi 0.30 \
    --b_perc 0.65 \
    --seed 42 \
    --n_jobs "${SLURM_CPUS_PER_TASK}"

  echo "[INFO] Completed: $FBASE"
  echo
done

echo "[INFO] All *_filtered.tsv files processed successfully."