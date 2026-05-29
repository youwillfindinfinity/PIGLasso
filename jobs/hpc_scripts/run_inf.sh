#!/bin/bash
# Set PROJECT_ROOT to repo parent, or override: export PROJECT_ROOT=/path/to/BurnInjuries
: "${PROJECT_ROOT:=$HOME/BurnInjuries}"
#SBATCH --job-name=net_inf
#SBATCH --partition=genoa
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
# CONFIG
# ============================================================

# Dataset: GSE182616 or GSE37069
DATASET="GSE182616"

# Prior toggle:
#   "yes" -> PIGLasso: pass --prior and --prior_weight to network_inference.py
#            (prior is also auto-detected from the pkl, but explicit is safer)
#   "no"  -> SSGLasso: pass --prior none to force no-prior mode
USE_PRIOR="yes"

# Path to the prior matrix (.npy). Only used when USE_PRIOR="yes".
PRIOR_PATH="$(pwd)/PIGLasso/pipeline_src/prior/prior_piglasso.npy"

# Prior strength in [0, 1].
PRIOR_WEIGHT=0.5

INPUT_DIR="$(pwd)/PIGLasso/pipeline_src/inference/results/piglasso/${DATASET}"
OUT_DIR="$(pwd)/PIGLasso/pipeline_src/inference/results/network_inference/${DATASET}"
mkdir -p "${OUT_DIR}"

# ============================================================
# Resolve prior args
# ============================================================
if [ "$USE_PRIOR" = "yes" ]; then
  if [ ! -f "$PRIOR_PATH" ]; then
    echo "[ERROR] Prior file not found: $PRIOR_PATH" >&2
    exit 1
  fi
  PRIOR_ARGS="--prior $PRIOR_PATH --prior_weight $PRIOR_WEIGHT"
  echo "[INFO] PIGLasso mode: prior ENABLED  path=$PRIOR_PATH  weight=$PRIOR_WEIGHT"
else
  PRIOR_ARGS="--prior none"
  echo "[INFO] SSGLasso mode: prior DISABLED"
fi

# ============================================================
# Find piglasso result pkls
# ============================================================
shopt -s nullglob
FILES=( "${INPUT_DIR}"/*__piglasso_results.pkl )
IFS=$'\n' FILES=( $(printf "%s\n" "${FILES[@]}" | sort) )
unset IFS

if [ ${#FILES[@]} -eq 0 ]; then
  echo "[ERROR] No *__piglasso_results.pkl files found in ${INPUT_DIR}" >&2
  exit 1
fi

echo "[INFO] Dataset:    ${DATASET}"
echo "[INFO] Input dir:  ${INPUT_DIR}"
echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Total files found: ${#FILES[@]}"
echo

# ============================================================
# Run network inference for each pkl
# ============================================================
for ((i=0; i<${#FILES[@]}; i++)); do
  FPATH="${FILES[$i]}"
  FBASE="$(basename "$FPATH" .pkl)"

  echo "[INFO] ========================================"
  echo "[INFO] File $((i+1))/${#FILES[@]}: $FBASE"
  echo "[INFO] ========================================"

  set -x
  python3 PIGLasso/pipeline_src/inference/network_inference.py \
    --piglasso_pkl   "$FPATH" \
    --out_dir        "${OUT_DIR}" \
    --edge_threshold 1e-5 \
    --plot \
    $PRIOR_ARGS
  set +x

  base="$(basename "${FPATH}" .pkl | sed 's/__piglasso_results$//')"
  echo "[INFO] Outputs:"
  ls -lh "${OUT_DIR}/${base}__inferred"* 2>/dev/null || echo "[WARN] No outputs matched for ${base}"
  echo
done

echo "[INFO] All network inference jobs completed.  (model: $([ "$USE_PRIOR" = "yes" ] && echo PIGLasso || echo SSGLasso))"