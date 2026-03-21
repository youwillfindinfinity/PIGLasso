#!/bin/bash
#SBATCH --job-name=net_inf
#SBATCH --partition=genoa
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
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

INPUT_DIR="burn_results/piglasso_results"
OUT_DIR="burn_results/network_inference"
mkdir -p "${OUT_DIR}"

shopt -s nullglob
FILES=( "${INPUT_DIR}"/*__piglasso_results.pkl )

# Optional: stable sort
IFS=$'\n' FILES=( $(printf "%s\n" "${FILES[@]}" | sort) )
unset IFS

if [ ${#FILES[@]} -eq 0 ]; then
  echo "[ERROR] No *__piglasso_results.pkl files found in ${INPUT_DIR}"
  exit 1
fi

echo "[INFO] Total files found: ${#FILES[@]}"
echo "[INFO] Output dir: ${OUT_DIR}"
echo

for ((i=0; i<${#FILES[@]}; i++)); do
  FPATH="${FILES[$i]}"
  FBASE="$(basename "$FPATH" .pkl)"

  echo "[INFO] ========================================"
  echo "[INFO] File index: $i  (1-based: $((i+1))/${#FILES[@]})"
  echo "[INFO] Input:  $FPATH"
  echo "[INFO] Base:   $FBASE"
  echo "[INFO] CPUs:   ${SLURM_CPUS_PER_TASK}"
  echo "[INFO] ========================================"

  set -x
  python3 monika/network_inference.py \
    --piglasso_pkl "$FPATH" \
    --out_dir "${OUT_DIR}" \
    --edge_threshold 1e-5 \
    --plot \
    --allow_install_glasso
  set +x

  echo "[INFO] Done. Matching outputs:"
  base="$(basename "${FPATH}" .pkl | sed 's/__piglasso_results$//')"
  ls -lh "${OUT_DIR}/${base}__inferred"* 2>/dev/null || echo "[WARN] No outputs matched for ${base}"
  echo
done

echo "[INFO] All network inference jobs completed."