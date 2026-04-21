#!/bin/bash
#SBATCH --job-name=net_diff
#SBATCH --partition=genoa
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home2/zblei/Documents/BurnInjuries
mkdir -p logs

module purge
module load 2024
module load R/4.4.2-gfbf-2024a
source /gpfs/home2/zblei/Documents/BurnInjuries/.venv/bin/activate

# ---------------- CONFIG ----------------
# Dataset: GSE182616 or GSE37069
DATASET="GSE182616"

# Model: PIGLasso (with prior) or SSGLasso (no prior) — must match run_diff_sig.sh
MODEL="PIGLasso"

IN_DIR="$(pwd)/PIGLasso/pipeline_src/diffusion/results/${DATASET}/${MODEL}/diff_sig"
OUT_DIR="$(pwd)/PIGLasso/pipeline_src/diffusion/results/${DATASET}/${MODEL}/net_diff"

ADJ_NAME="burn_network_adjacency.csv"
DELTA_NAME="delta.tsv"
COMMON_NAME="common_genes.txt"

# diffusion settings
TMIN="1e-4"
TMAX="3.0"
NT="80"

STRICT="--strict_gene_match"

# optional toggles:
# NORM="--normalized_laplacian"
# LCC="--use_lcc"
NORM=""
LCC=""
# --------------------------------------

mkdir -p "$OUT_DIR"

ADJ_PATH="${IN_DIR}/${ADJ_NAME}"
DELTA_PATH="${IN_DIR}/${DELTA_NAME}"
COMMON_PATH="${IN_DIR}/${COMMON_NAME}"

if [ ! -f "$ADJ_PATH" ]; then
  echo "[ERROR] Missing file: $ADJ_PATH" >&2
  exit 1
fi

if [ ! -f "$DELTA_PATH" ]; then
  echo "[ERROR] Missing file: $DELTA_PATH" >&2
  exit 1
fi

if [ ! -f "$COMMON_PATH" ]; then
  echo "[ERROR] Missing file: $COMMON_PATH" >&2
  exit 1
fi

START_TS=$(date +%s)

echo "============================================================" >&2
echo "[INFO] Starting network diffusion" >&2
echo "[INFO] in_dir  : $IN_DIR" >&2
echo "[INFO] out_dir : $OUT_DIR" >&2
echo "============================================================" >&2

python3 PIGLasso/pipeline_src/diffusion/network_diffusion.py \
  --in_dir "$IN_DIR" \
  --adj "$ADJ_NAME" \
  --delta "$DELTA_NAME" \
  --common_genes "$COMMON_NAME" \
  $STRICT \
  $NORM \
  $LCC \
  --out_dir "$OUT_DIR" \
  --tmin "$TMIN" \
  --tmax "$TMAX" \
  --nt "$NT"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
ELAPSED_FMT=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo "[INFO] Network diffusion completed in $ELAPSED_FMT" >&2