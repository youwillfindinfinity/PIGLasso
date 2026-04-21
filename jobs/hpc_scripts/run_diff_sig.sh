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

# Model: "PIGLasso" (with prior) or "SSGLasso" (no prior)
# Must match what was used in run_inf.sh
MODEL="PIGLasso"

# Per-patient ctrl pseudobulk directory — GSE37069 control timepoint (10 files, one per patient)
# diffusion_signal.py globs *__pseudobulk_genes_x_timepoint.tsv here
CTRL_PB_DIR="$(pwd)/preprocessing/burn_control/preprocessed"

# ============================================================
# Resolve inferred pkl path based on MODEL
# PIGLasso pkl has __seed42__pw0.5__ in stem; SSGLasso has __seed42__ only
# ============================================================
INF_DIR="$(pwd)/PIGLasso/pipeline_src/inference/results/network_inference/${DATASET}"

if [ "$MODEL" = "PIGLasso" ]; then
  INFER_PKL=$(ls "${INF_DIR}"/*__seed42__pw*__inferred.pkl 2>/dev/null | head -1)
elif [ "$MODEL" = "SSGLasso" ]; then
  # no-prior pkls do NOT have pw in the stem
  INFER_PKL=$(ls "${INF_DIR}"/*__seed42__inferred.pkl 2>/dev/null | grep -v "__pw" | head -1)
else
  echo "[ERROR] MODEL must be 'PIGLasso' or 'SSGLasso' (got: $MODEL)" >&2
  exit 1
fi

OUT_DIR="$(pwd)/PIGLasso/pipeline_src/diffusion/results/${DATASET}/${MODEL}/diff_sig"

CTRL_COL="Ctrl"
MIN_COMMON_GENES=20

# ============================================================
mkdir -p "$OUT_DIR"

if [ -z "${INFER_PKL:-}" ] || [ ! -f "$INFER_PKL" ]; then
  echo "[ERROR] Could not find inferred network pkl for MODEL=${MODEL} in ${INF_DIR}" >&2
  echo "        Run run_inf.sh first with matching USE_PRIOR setting." >&2
  exit 1
fi

if [ ! -d "$CTRL_PB_DIR" ]; then
  echo "[ERROR] Missing ctrl pseudobulk dir: $CTRL_PB_DIR" >&2
  exit 1
fi

START_TS=$(date +%s)

echo "============================================================" >&2
echo "[INFO] Running diffusion_signal  (model: ${MODEL})" >&2
echo "[INFO] dataset          : $DATASET" >&2
echo "[INFO] inferred network : $INFER_PKL" >&2
echo "[INFO] ctrl pb dir      : $CTRL_PB_DIR" >&2
echo "[INFO] output dir       : $OUT_DIR" >&2
echo "============================================================" >&2

python3 diffusion/diffusion_signal.py \
  --burn_inferred_pkl     "$INFER_PKL" \
  --ctrl_pseudobulk_dir   "$CTRL_PB_DIR" \
  --ctrl_col              "$CTRL_COL" \
  --min_common_genes      "$MIN_COMMON_GENES" \
  --out_dir               "$OUT_DIR"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
ELAPSED_FMT=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo "[INFO] diffusion_signal completed in $ELAPSED_FMT" >&2
echo "[INFO] Outputs in: $OUT_DIR" >&2
