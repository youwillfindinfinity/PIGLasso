#!/bin/bash
# Set PROJECT_ROOT to repo parent, or override: export PROJECT_ROOT=/path/to/BurnInjuries
: "${PROJECT_ROOT:=$HOME/BurnInjuries}"
#SBATCH --job-name=knockout
#SBATCH --partition=genoa
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ${PROJECT_ROOT}
mkdir -p logs

module purge
module load 2024
module load R/4.4.2-gfbf-2024a
source ${PROJECT_ROOT}/.venv/bin/activate

# ---------------- CONFIG ----------------
# Dataset: GSE182616 or GSE37069
DATASET="GSE182616"

# Model: PIGLasso (with prior) or SSGLasso (no prior) — must match run_diff_sig.sh
MODEL="PIGLasso"

IN_DIR="$(pwd)/PIGLasso/pipeline_src/diffusion/results/${DATASET}/${MODEL}/diff_sig"
OUT_DIR="$(pwd)/PIGLasso/pipeline_src/knockouts/results/${DATASET}/${MODEL}"

# Required file inside IN_DIR
DELTA_NAME="delta.tsv"

# knockout settings
REDUCTION=0.1
TMAX=3.0
TNUM=100
TOPK=50
# ----------------------------------------

mkdir -p "$OUT_DIR"

DELTA_PATH="${IN_DIR}/${DELTA_NAME}"

if [ ! -d "$IN_DIR" ]; then
  echo "[ERROR] Input directory not found: $IN_DIR" >&2
  exit 1
fi

if [ ! -f "$DELTA_PATH" ]; then
  echo "[ERROR] Missing required file: $DELTA_PATH" >&2
  exit 1
fi

START_TS=$(date +%s)

echo "============================================================" >&2
echo "[INFO] Starting node knockout" >&2
echo "[INFO] in_dir   : $IN_DIR" >&2
echo "[INFO] out_dir  : $OUT_DIR" >&2
echo "[INFO] delta    : $DELTA_PATH" >&2
echo "[INFO] reduction: $REDUCTION" >&2
echo "[INFO] t_max    : $TMAX" >&2
echo "[INFO] t_num    : $TNUM" >&2
echo "[INFO] topk     : $TOPK" >&2
echo "============================================================" >&2

python3 PIGLasso/pipeline_src/knockouts/node_knockout.py \
  --in_dir "$IN_DIR" \
  --delta "$DELTA_NAME" \
  --t_max "$TMAX" \
  --t_num "$TNUM" \
  --reduction "$REDUCTION" \
  --topk_traces "$TOPK" \
  --out_dir "$OUT_DIR"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
ELAPSED_FMT=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo "[INFO] Node knockout completed in $ELAPSED_FMT" >&2

ls -lh "$OUT_DIR" | tail -n 20