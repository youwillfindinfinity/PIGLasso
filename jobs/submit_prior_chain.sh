#!/bin/bash
# Submit prior construction jobs as a dependency chain on Snellius.
#
# Chain:
#   step1 → step2a ─┐
#                    ├─ step3 → step4
#          step2b ──┤
#          step2c ──┘
#
# Usage (run from NODIS root on Snellius):
#   bash jobs/submit_prior_chain.sh [partition]
# Default partition: rome

set -euo pipefail

PARTITION="${1:-rome}"
JOBS="$HOME/NODIS/jobs"

mkdir -p "$HOME/NODIS/logs"

submit() {
    local args="$*"
    sbatch --partition="$PARTITION" $args | awk '{print $NF}'
}

# Step 1 — extract genes (no dependency)
JID1=$(submit "$JOBS/prior_step1.job")
echo "step1  submitted: $JID1"

# Steps 2a, 2b, 2c — parallel, each waits for step1
JID2A=$(submit --dependency=afterok:$JID1 "$JOBS/prior_step2a.job")
echo "step2a submitted: $JID2A  (after $JID1)"

JID2B=$(submit --dependency=afterok:$JID1 "$JOBS/prior_step2b.job")
echo "step2b submitted: $JID2B  (after $JID1)"

JID2C=$(submit --dependency=afterok:$JID1 "$JOBS/prior_step2c.job")
echo "step2c submitted: $JID2C  (after $JID1)"

# Step 3 — combine, waits for all three 2x jobs
JID3=$(submit --dependency=afterok:$JID2A:$JID2B:$JID2C "$JOBS/prior_step3.job")
echo "step3  submitted: $JID3   (after $JID2A,$JID2B,$JID2C)"

# Step 4 — validate
JID4=$(submit --dependency=afterok:$JID3 "$JOBS/prior_step4.job")
echo "step4  submitted: $JID4   (after $JID3)"

echo ""
echo "Full chain submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  squeue -j $JID1,$JID2A,$JID2B,$JID2C,$JID3,$JID4"
echo ""
echo "Outputs: \$HOME/NODIS/methodtotest/prior/"
echo "Logs:    \$HOME/NODIS/logs/prior_step*"
