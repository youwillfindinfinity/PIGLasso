#!/usr/bin/env bash
# run_small_example.sh — end-to-end PIGLasso pipeline on the 20-gene synthetic dataset
#
# Requirements:
#   pip install nodis gglasso
#   pip install -e .          (from repo root)
#
# Usage:
#   bash examples/run_small_example.sh
#   bash examples/run_small_example.sh --skip-piglasso   # if nodis/gglasso not installed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_ROOT/examples/data/small_example"
OUT_DIR="$REPO_ROOT/examples/results/small_example"
SKIP_PIGLASSO=0

for arg in "$@"; do
  [[ "$arg" == "--skip-piglasso" ]] && SKIP_PIGLASSO=1
done

echo "=== PIGLasso small example pipeline ==="
echo "  Repo   : $REPO_ROOT"
echo "  Data   : $DATA_DIR"
echo "  Output : $OUT_DIR"
echo ""

# ── Step 0: generate synthetic data if not already present ───────────────────
if [[ ! -f "$DATA_DIR/expression.tsv" ]]; then
  echo "[0/5] Generating synthetic data..."
  python3 "$REPO_ROOT/examples/make_small_example.py"
else
  echo "[0/5] Synthetic data already present — skipping generation"
fi

# ── Step 1: filter to top genes (pass-through on small data) ─────────────────
echo ""
echo "[1/5] filter_top_genes.py — subset to top 20 by variance..."
python3 "$REPO_ROOT/scripts/filter_top_genes.py" \
  --expr    "$DATA_DIR/expression.tsv" \
  --genes   "$DATA_DIR/genes.txt" \
  --prior   "$DATA_DIR/prior.npy" \
  --n-genes 20 \
  --out-dir "$DATA_DIR"
echo "  Done."

# ── Step 2: PIGLasso inference ────────────────────────────────────────────────
if [[ $SKIP_PIGLASSO -eq 1 ]]; then
  echo ""
  echo "[2/5] Skipping PIGLasso inference (--skip-piglasso). Using pre-built adjacency."
  mkdir -p "$OUT_DIR/network"
  cp "$DATA_DIR/adjacency.csv"  "$OUT_DIR/network/expression_adjacency.csv"
  cp "$DATA_DIR/stability.csv"  "$OUT_DIR/network/expression_stability.csv"
else
  echo ""
  echo "[2/5] piglasso run — stability-selection inference..."
  piglasso run \
    --data          "$DATA_DIR/expression.tsv" \
    --prior         "$DATA_DIR/prior.npy" \
    --prior-weight  0.5 \
    --n-subsamples  20 \
    --lambda-len    8 \
    --lambda-lo     0.05 \
    --lambda-hi     0.40 \
    --pi-thr        0.6 \
    --seed          42 \
    --out           "$OUT_DIR/network"
  echo "  Done."
fi

# ── Step 3: network diffusion ─────────────────────────────────────────────────
echo ""
echo "[3/5] network_diffusion.py — heat-kernel diffusion..."
mkdir -p "$OUT_DIR/diffusion_inputs"
cp "$OUT_DIR/network/expression_adjacency.csv" "$OUT_DIR/diffusion_inputs/adjacency.csv"
cp "$DATA_DIR/delta.tsv"                       "$OUT_DIR/diffusion_inputs/delta.tsv"

python3 "$REPO_ROOT/pipeline_src/diffusion/network_diffusion.py" \
  --in_dir  "$OUT_DIR/diffusion_inputs" \
  --adj     adjacency.csv \
  --delta   delta.tsv \
  --out_dir "$OUT_DIR/diffusion" \
  --tmin    0.01 \
  --tmax    2.0 \
  --nt      20
echo "  Done."

# ── Step 4: node knockout ─────────────────────────────────────────────────────
echo ""
echo "[4/5] node_knockout.py — hub essentiality..."
python3 "$REPO_ROOT/pipeline_src/knockouts/node_knockout.py" \
  --in_dir   "$OUT_DIR/diffusion_inputs" \
  --network  adjacency.csv \
  --delta    delta.tsv \
  --out_dir  "$OUT_DIR/knockouts" \
  --t_max    2.0 \
  --t_num    20 \
  --reduction 0.5 \
  --topk_traces 5
echo "  Done."

# ── Step 5: hub analysis ──────────────────────────────────────────────────────
echo ""
echo "[5/5] hub_analysis_burns.py — centrality ranking..."
python3 "$REPO_ROOT/scripts/hub_analysis_burns.py" \
  --adj   "$OUT_DIR/network/expression_adjacency.csv" \
  --stab  "$OUT_DIR/network/expression_stability.csv" \
  --prior "$DATA_DIR/prior.npy" \
  --genes "$DATA_DIR/genes.txt" \
  --out   "$OUT_DIR/hubs"
echo "  Done."

echo ""
echo "=== Pipeline complete ==="
echo "Results written to: $OUT_DIR"
ls "$OUT_DIR"
