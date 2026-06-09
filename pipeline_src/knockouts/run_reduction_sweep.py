#!/usr/bin/env python3
"""
run_reduction_sweep.py
----------------------
Runs node_knockout.py for each reduction factor in the sweep, saving results
to results/GSE182616/PIGLasso/reduction_sweep/r<NN>/

Reduction values: 0.02, 0.05, 0.10, 0.20, 0.30, 0.50
The 0.10 run is skipped if the existing result already exists.

Usage:
    cd PIGLasso/pipeline_src/knockouts/
    python run_reduction_sweep.py
"""

import subprocess
import sys
from pathlib import Path

HERE      = Path(__file__).resolve().parent
IN_DIR    = HERE.parent / "diffusion" / "results" / "GSE182616" / "PIGLasso" / "diff_sig"
BASE_OUT  = HERE / "results" / "GSE182616" / "PIGLasso" / "reduction_sweep"
KNOCKOUT  = HERE / "node_knockout.py"
PYTHON    = sys.executable

REDUCTIONS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def reduction_tag(r: float) -> str:
    return f"r{round(r * 100):03d}"


def main() -> None:
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    for r in REDUCTIONS:
        tag     = reduction_tag(r)
        out_dir = BASE_OUT / tag
        result  = out_dir / f"perturbative_gene_impacts_reduction_{r:.3f}.tsv"

        if result.exists():
            print(f"[SKIP] reduction={r} — already exists: {result}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RUN ] reduction={r} → {out_dir}")

        cmd = [
            PYTHON, str(KNOCKOUT),
            "--in_dir",   str(IN_DIR),
            "--network",  "burn_network_edgelist.tsv",
            "--reduction", str(r),
            "--out_dir",  str(out_dir),
            "--t_max",    "3.0",
            "--t_num",    "100",
        ]

        result_proc = subprocess.run(cmd, text=True)
        if result_proc.returncode != 0:
            print(f"[ERR ] reduction={r} failed with code {result_proc.returncode}")
        else:
            print(f"[DONE] reduction={r}")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
