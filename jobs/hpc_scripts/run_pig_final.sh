#!/bin/bash
# Set PROJECT_ROOT to repo parent, or override: export PROJECT_ROOT=/path/to/BurnInjuries
: "${PROJECT_ROOT:=$HOME/BurnInjuries}"
#SBATCH --job-name=run_pig
#SBATCH --time=120:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ${PROJECT_ROOT}
mkdir -p logs

module purge
module load 2024
module load R/4.4.2-gfbf-2024a

source ${PROJECT_ROOT}/.venv/bin/activate

# ============================================================
# CONFIG: choose mode
#   MODE="burn"  -> run selected burn filtered files
#   MODE="bench" -> run all benchmark preprocessed GeneExpression TSVs
# ============================================================
MODE="bench"   

# PIGLASSO PARAMS (shared)
Q=200
LAMLEN=20
LLO=0.05
LHI=0.30
B_PERC=0.65
SEED=42

# ============================================================
# Build file list depending on MODE
# ============================================================
FILES=()

if [ "$MODE" = "burn" ]; then
  FILES=(
    "preprocessing/burn/filtered/GSE37069/phase/PHASE__Acute__n239__zscored__adult18plus__filtered.tsv" # add path to filtered burn data
  )

elif [ "$MODE" = "bench" ]; then
  shopt -s nullglob
  FILES=(benchmarking/data/SGG/160/*_data.tsv) # change to correct file location (SGG, GRN or dream)
  shopt -u nullglob

  # Sort for reproducibility
  if [ ${#FILES[@]} -gt 0 ]; then
    IFS=$'\n' FILES=($(printf "%s\n" "${FILES[@]}" | sort))
    unset IFS
  fi

else
  echo "[ERROR] MODE must be 'burn' or 'bench' (got: $MODE)" >&2
  exit 1
fi

TOTAL=${#FILES[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "[ERROR] No input files found for MODE=$MODE" >&2
  exit 1
fi

echo "[INFO] MODE=$MODE"
echo "[INFO] Running ${TOTAL} file(s) sequentially"
echo

# ============================================================
# Run sequentially with progress
# ============================================================
IDX=0
for FPATH in "${FILES[@]}"; do
  IDX=$((IDX + 1))

  if [ ! -f "$FPATH" ]; then
    echo "[ERROR] File not found: $FPATH" >&2
    exit 1
  fi

  FBASE="$(basename "$FPATH" .tsv)"

  echo "============================================================" >&2
  echo "[PROGRESS] ($IDX/$TOTAL) Running PIGLASSO for: $FBASE" >&2
  echo "[PROGRESS] mode   : $MODE" >&2
  echo "[PROGRESS] input  : $FPATH" >&2
  echo "[PROGRESS] cores  : ${SLURM_CPUS_PER_TASK}" >&2
  echo "============================================================" >&2

  python3 inference/run_piglasso_new.py \
    --mode "$MODE" \
    --input "$FPATH" \
    --Q "$Q" \
    --lamlen "$LAMLEN" \
    --llo "$LLO" \
    --lhi "$LHI" \
    --b_perc "$B_PERC" \
    --seed "$SEED" \
    --n_jobs "${SLURM_CPUS_PER_TASK}"

  echo "[PROGRESS] ($IDX/$TOTAL) Completed: $FBASE" >&2
  echo >&2
done

echo "[INFO] Finished MODE=$MODE run." >&2