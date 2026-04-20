#!/bin/bash
#SBATCH --job-name=diff_sig
#SBATCH --partition=genoa
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home2/zblei/Documents/BurnInjuries
mkdir -p logs

module purge
module load 2024
module load R/4.4.2-gfbf-2024a
source /gpfs/home2/zblei/Documents/BurnInjuries/.venv/bin/activate

# ============================================================
# CONFIG
# ============================================================

# Dataset: GSE182616 or GSE37069
DATASET="GSE182616"

# Model: PIGLasso (with prior) or SSGLasso (no prior)
# Must match the pkl that was produced by run_inf.sh
MODEL="PIGLasso"

# Inferred network pkl — output of network_inference.py
# Adjust the filename stem to match what run_inf.sh produced
INFER_PKL="inference/results/network_inference/${DATASET}/PHASE__Acute__n513__zscored__filtered__Q200__bperc0.65__lam0.05-0.3x20__seed42__pw0.5__inferred.pkl"

# Per-patient trauma pseudobulk directory (10 files, one per patient)
# diffusion_signal.py iterates over *__pseudobulk_genes_x_timepoint.tsv files here
TRAUMA_PB_DIR="trauma_data/preprocessed"

OUT_DIR="diffusion/results/${DATASET}/${MODEL}/diff_sig"

CTRL_COL="Ctrl"
MIN_COMMON_GENES=20

# ============================================================
mkdir -p "$OUT_DIR"

if [ ! -f "$INFER_PKL" ]; then
  echo "[ERROR] Missing inferred network pkl: $INFER_PKL" >&2
  echo "        Run run_inf.sh first (USE_PRIOR=yes for PIGLasso)" >&2
  exit 1
fi

if [ ! -d "$TRAUMA_PB_DIR" ]; then
  echo "[ERROR] Missing trauma pseudobulk dir: $TRAUMA_PB_DIR" >&2
  exit 1
fi

START_TS=$(date +%s)

echo "============================================================" >&2
echo "[INFO] Running diffusion_signal  (model: ${MODEL})" >&2
echo "[INFO] dataset          : $DATASET" >&2
echo "[INFO] inferred network : $INFER_PKL" >&2
echo "[INFO] trauma pb dir    : $TRAUMA_PB_DIR" >&2
echo "[INFO] output dir       : $OUT_DIR" >&2
echo "============================================================" >&2

python3 diffusion/diffusion_signal.py \
  --burn_inferred_pkl  "$INFER_PKL" \
  --trauma_pseudobulk_dir "$TRAUMA_PB_DIR" \
  --ctrl_col           "$CTRL_COL" \
  --min_common_genes   "$MIN_COMMON_GENES" \
  --out_dir            "$OUT_DIR"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
ELAPSED_FMT=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo "[INFO] diffusion_signal completed in $ELAPSED_FMT" >&2
echo "[INFO] Outputs in: $OUT_DIR" >&2
