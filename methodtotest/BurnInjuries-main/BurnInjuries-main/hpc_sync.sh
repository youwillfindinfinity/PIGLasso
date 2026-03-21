#!/usr/bin/env bash
# hpc_sync.sh — sync code to Snellius and/or pull results back
#
# Usage:
#   ./hpc_sync.sh push       # push local code → HPC
#   ./hpc_sync.sh pull       # pull HPC results → local
#   ./hpc_sync.sh push pull  # do both
#
# Password is passed via the SSHPASS env var (never appears in process list).

set -euo pipefail

# ── Load .env ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] .env not found. Copy .env.template to .env and fill in your credentials." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

HPC_USER="${HPC_USER:?HPC_USER not set in .env}"
HPC_HOST="${HPC_HOST:?HPC_HOST not set in .env}"
HPC_REMOTE_DIR="${HPC_REMOTE_DIR:?HPC_REMOTE_DIR not set in .env}"
HPC_SSH_PASS="${HPC_SSH_PASS:?HPC_SSH_PASS not set in .env}"
HPC_SSH_PORT="${HPC_SSH_PORT:-22}"

# ── Require sshpass ───────────────────────────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
  echo "[ERROR] sshpass is not installed." >&2
  echo "        Install with: brew install hudochenkov/sshpass/sshpass" >&2
  exit 1
fi

# Export password for sshpass -e (reads from SSHPASS env var, never from argv)
export SSHPASS="$HPC_SSH_PASS"

SSH_OPTS="-p ${HPC_SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# sshpass -e reads password from $SSHPASS env var — password never appears in process list
RSYNC_SSH="sshpass -e ssh ${SSH_OPTS}"
REMOTE="${HPC_USER}@${HPC_HOST}:${HPC_REMOTE_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }

remote_exists() {
  sshpass -e ssh $SSH_OPTS "${HPC_USER}@${HPC_HOST}" \
    "[ -d '${HPC_REMOTE_DIR}/$1' ]" 2>/dev/null
}

do_push() {
  log "PUSH: local → ${REMOTE}"

  # Push source code and scripts only.
  # Data, results, venv, and cache are excluded.
  rsync -avz --progress \
    -e "$RSYNC_SSH" \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.env' \
    --exclude='preprocessing/burn/filtered/' \
    --exclude='preprocessing/burn/stratified/' \
    --exclude='preprocessing/data/' \
    --exclude='benchmarking/data/' \
    --exclude='burn_data/' \
    --exclude='chronic_data/ensembl/' \
    --exclude='inference/results/' \
    --exclude='diffusion/results/' \
    --exclude='diffusion/inputs/' \
    --exclude='knockouts/results/' \
    --exclude='benchmarking/results/' \
    --exclude='multiscale/results/' \
    --exclude='multiscale/plots/' \
    --exclude='plotting/burn_results/' \
    --exclude='archive/' \
    --exclude='results/' \
    --exclude='*.pkl' \
    --exclude='*.npy' \
    "$SCRIPT_DIR/" \
    "$REMOTE/"

  log "PUSH complete."
}

do_pull() {
  log "PULL: ${REMOTE} → local"

  PULL_DIRS=(
    "preprocessing/burn/filtered"
    "preprocessing/burn/stratified"
    "preprocessing/burn_control/preprocessed"
    "benchmarking/data/SGG"
    "benchmarking/data/GRN"
    "benchmarking/results"
    "inference/results/piglasso"
    "inference/results/network_inference"
    "inference/results/pca"
    "diffusion/inputs"
    "diffusion/results"
    "knockouts/results"
    "multiscale/results"
    "multiscale/plots"
    "plotting/burn_results"
  )

  for dir in "${PULL_DIRS[@]}"; do
    LOCAL_PATH="${SCRIPT_DIR}/${dir}/"
    REMOTE_PATH="${HPC_USER}@${HPC_HOST}:${HPC_REMOTE_DIR}/${dir}/"

    if remote_exists "$dir"; then
      log "Pulling ${dir} ..."
      mkdir -p "$LOCAL_PATH"
      rsync -avz --progress \
        -e "$RSYNC_SSH" \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        "$REMOTE_PATH" \
        "$LOCAL_PATH"
    else
      warn "${dir} does not exist on HPC — skipping."
    fi
  done

  log "PULL complete."
}

# ── Entry point ───────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 push | pull | push pull" >&2
  exit 1
fi

for cmd in "$@"; do
  case "$cmd" in
    push) do_push ;;
    pull) do_pull ;;
    *) echo "[ERROR] Unknown command: $cmd. Use push or pull." >&2; exit 1 ;;
  esac
done
