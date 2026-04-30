#!/bin/bash
#SBATCH --job-name=pig_prior
#SBATCH --time=120:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home2/zblei/Documents/BurnInjuries/PIGLasso
mkdir -p logs

module purge
module load 2024
module load R/4.4.2-gfbf-2024a

source /gpfs/home2/zblei/Documents/BurnInjuries/NODIS/.venv/bin/activate

# ============================================================
# CONFIG
# ============================================================

# Dataset mode:
#   "burn"  -> run selected burn filtered files
#   "bench" -> run all benchmark preprocessed TSVs
MODE="burn"

# Prior toggle:
#   "yes" -> pass --prior and --prior_weight to run_piglasso_new.py
#   "no"  -> run standard PIGLasso without prior
USE_PRIOR="yes"

# Path to the prior matrix (.npy).  Only used when USE_PRIOR="yes".
# Built by: python pipeline_src/build_prior.py --step 1..4
PRIOR_PATH="pipeline_src/prior/prior_piglasso.npy"  # relative to PIGLasso/

# Prior strength in [0, 1].  0 = no effect, 1 = maximum reduction.
PRIOR_WEIGHT=0.5

# PIGLASSO PARAMS (shared)
Q=200
LAMLEN=20
LLO=0.05
LHI=1.0
B_PERC=0.65
SEED=42

# ============================================================
# Build file list depending on MODE
# ============================================================
FILES=()

if [ "$MODE" = "burn" ]; then
  FILES=(
    "/gpfs/home2/zblei/Documents/BurnInjuries/preprocessing/burn/filtered/GSE182616/phase/PHASE__Acute__n513__zscored__filtered.tsv"
  )

elif [ "$MODE" = "bench" ]; then
  shopt -s nullglob
  FILES=(benchmarking/data/SGG/160/*_data.tsv)
  shopt -u nullglob

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

# ============================================================
# Validate prior file exists when USE_PRIOR=yes
# ============================================================
if [ "$USE_PRIOR" = "yes" ]; then
  if [ ! -f "$PRIOR_PATH" ]; then
    echo "[ERROR] Prior file not found: $PRIOR_PATH" >&2
    echo "        Build it first: python pipeline_src/build_prior.py --step 1" >&2
    echo "        Then: --step 2a, 2b, 2c, 3, 4" >&2
    exit 1
  fi
  PRIOR_ARGS="--prior $PRIOR_PATH --prior_weight $PRIOR_WEIGHT"
  echo "[INFO] Prior: ENABLED  path=$PRIOR_PATH  weight=$PRIOR_WEIGHT"
else
  PRIOR_ARGS=""
  echo "[INFO] Prior: DISABLED (standard PIGLasso)"
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
  echo "[PROGRESS] mode        : $MODE" >&2
  echo "[PROGRESS] input       : $FPATH" >&2
  echo "[PROGRESS] prior       : ${USE_PRIOR}" >&2
  echo "[PROGRESS] cores       : ${SLURM_CPUS_PER_TASK}" >&2
  echo "============================================================" >&2

  python3 pipeline_src/inference/run_piglasso_new.py \
    --mode "$MODE" \
    --input "$FPATH" \
    --Q "$Q" \
    --lamlen "$LAMLEN" \
    --llo "$LLO" \
    --lhi "$LHI" \
    --b_perc "$B_PERC" \
    --seed "$SEED" \
    --n_jobs "${SLURM_CPUS_PER_TASK}" \
    $PRIOR_ARGS

  echo "[PROGRESS] ($IDX/$TOTAL) Completed: $FBASE" >&2
  echo >&2
done

echo "[INFO] Finished MODE=$MODE run (prior=${USE_PRIOR})." >&2
