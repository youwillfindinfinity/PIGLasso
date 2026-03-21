#!/bin/bash
#SBATCH --job-name=run_pig
#SBATCH --time=120:00:00
#SBATCH --partition=genoa
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

# FILE LIST (runs sequentially)

FILES=(
"burn_data/preprocessed/filtered/Moderate__Elder__Proliferation__n12__filtered.tsv"
"burn_data/preprocessed/filtered/Severe__MidAdult__Remodelling__n12__filtered.tsv"
"burn_data/preprocessed/filtered/Severe__YngAdult__Acute__n61__filtered.tsv"
"burn_data/preprocessed/filtered/Severe__YngAdult__Proliferation__n28__filtered.tsv"
)

TOTAL=${#FILES[@]}
echo "[INFO] Running ${TOTAL} selected files sequentially"

IDX=0
for FPATH in "${FILES[@]}"; do
  IDX=$((IDX + 1))

  if [ ! -f "$FPATH" ]; then
    echo "[ERROR] File not found: $FPATH"
    exit 1
  fi

  FBASE="$(basename "$FPATH" .tsv)"

  echo
  echo "[INFO] ========================================"
  echo "[INFO] ($IDX/$TOTAL) Running PIGLASSO for: $FBASE"
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
done

echo
echo "[INFO] Finished run of selected datasets."