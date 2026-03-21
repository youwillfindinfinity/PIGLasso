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

# ---------------- CONFIG ----------------
# change the paths below to GSE37069 or GSE182616 and correct .pkl file name
INFER_PKL="inference/results/network_inference/GSE182616/PHASE__Acute__n513__zscored__filtered__Q200__bperc0.65__lam0.05-0.3x20__inferred.pkl"
CONTROL_PB="preprocessing/burn_control/preprocessed/GSE37069_controls__pseudobulk_genes_x_timepoint.tsv"
OUT_DIR="diffusion/results/GSE182616/diff_sig"

CTRL_COL="Ctrl"
MIN_COMMON_GENES=20
# ----------------------------------------

mkdir -p "$OUT_DIR"

if [ ! -f "$INFER_PKL" ]; then
  echo "[ERROR] Missing inferred network file: $INFER_PKL" >&2
  exit 1
fi

if [ ! -f "$CONTROL_PB" ]; then
  echo "[ERROR] Missing control pseudobulk file: $CONTROL_PB" >&2
  exit 1
fi

START_TS=$(date +%s)

echo "============================================================" >&2
echo "[INFO] Running diffusion_signal" >&2
echo "[INFO] inferred network : $INFER_PKL" >&2
echo "[INFO] control pseudobulk: $CONTROL_PB" >&2
echo "[INFO] output dir       : $OUT_DIR" >&2
echo "============================================================" >&2

python3 diffusion/diffusion_signal.py \
  --burn_inferred_pkl "$INFER_PKL" \
  --control_pseudobulk "$CONTROL_PB" \
  --ctrl_col "$CTRL_COL" \
  --min_common_genes "$MIN_COMMON_GENES" \
  --out_dir "$OUT_DIR"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
ELAPSED_FMT=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo "[INFO] Diffusion completed in $ELAPSED_FMT" >&2