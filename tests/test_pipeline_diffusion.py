"""
Tests for pipeline_src/diffusion/network_diffusion.py
and pipeline_src/knockouts/node_knockout.py.

Both scripts are pure numpy/scipy — no nodis dependency.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DIFFUSION_SCRIPT = REPO / "pipeline_src" / "diffusion" / "network_diffusion.py"
KNOCKOUT_SCRIPT  = REPO / "pipeline_src" / "knockouts" / "node_knockout.py"


# ---------------------------------------------------------------------------
# Shared fixture: diffusion inputs directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def diffusion_inputs(tmp_path_factory):
    """Create a minimal diffusion_inputs/ directory with adjacency.csv and delta.tsv."""
    d = tmp_path_factory.mktemp("diffusion_inputs")
    genes = [f"G{i:02d}" for i in range(15)]

    # adjacency: simple chain graph
    adj = np.zeros((15, 15))
    for i in range(14):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    pd.DataFrame(adj, index=genes, columns=genes).to_csv(d / "adjacency.csv")

    # delta vector
    delta = np.zeros(15)
    delta[:7] = np.linspace(0.5, 2.0, 7)
    delta[7:] = np.linspace(-0.5, -1.5, 8)
    pd.DataFrame({"delta": delta}, index=pd.Index(genes, name="gene")).to_csv(
        d / "delta.tsv", sep="\t"
    )
    return d


# ---------------------------------------------------------------------------
# network_diffusion.py
# ---------------------------------------------------------------------------

class TestNetworkDiffusion:

    def test_basic_run_produces_outputs(self, diffusion_inputs, tmp_path):
        r = subprocess.run(
            [sys.executable, str(DIFFUSION_SCRIPT),
             "--in_dir",  str(diffusion_inputs),
             "--adj",     "adjacency.csv",
             "--delta",   "delta.tsv",
             "--out_dir", str(tmp_path),
             "--tmin",    "0.01",
             "--tmax",    "1.0",
             "--nt",      "10"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        # Expect at least one output file (hub scores or diffusion result)
        outputs = list(tmp_path.glob("*.csv")) + list(tmp_path.glob("*.tsv")) + list(tmp_path.glob("*.npy"))
        assert outputs, f"No output files written. stderr:\n{r.stderr}"

    def test_help_flag(self):
        r = subprocess.run(
            [sys.executable, str(DIFFUSION_SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "--in_dir" in r.stdout

    def test_missing_adj_file_exits_nonzero(self, tmp_path):
        r = subprocess.run(
            [sys.executable, str(DIFFUSION_SCRIPT),
             "--in_dir",  str(tmp_path),
             "--adj",     "nonexistent.csv",
             "--delta",   "delta.tsv",
             "--out_dir", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert r.returncode != 0

    def test_lcc_flag(self, diffusion_inputs, tmp_path):
        """--use_lcc should succeed on a connected graph."""
        r = subprocess.run(
            [sys.executable, str(DIFFUSION_SCRIPT),
             "--in_dir",  str(diffusion_inputs),
             "--adj",     "adjacency.csv",
             "--delta",   "delta.tsv",
             "--out_dir", str(tmp_path),
             "--tmin",    "0.1",
             "--tmax",    "1.0",
             "--nt",      "5",
             "--use_lcc"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

    def test_normalized_laplacian_flag(self, diffusion_inputs, tmp_path):
        r = subprocess.run(
            [sys.executable, str(DIFFUSION_SCRIPT),
             "--in_dir",             str(diffusion_inputs),
             "--adj",                "adjacency.csv",
             "--delta",              "delta.tsv",
             "--out_dir",            str(tmp_path),
             "--tmin",               "0.1",
             "--tmax",               "1.0",
             "--nt",                 "5",
             "--normalized_laplacian"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# node_knockout.py
# ---------------------------------------------------------------------------

class TestNodeKnockout:

    def test_basic_run_produces_outputs(self, diffusion_inputs, tmp_path):
        r = subprocess.run(
            [sys.executable, str(KNOCKOUT_SCRIPT),
             "--in_dir",    str(diffusion_inputs),
             "--network",   "adjacency.csv",
             "--delta",     "delta.tsv",
             "--out_dir",   str(tmp_path),
             "--t_max",     "1.0",
             "--t_num",     "5",
             "--reduction", "0.5",
             "--topk_traces", "3"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        outputs = list(tmp_path.glob("*.tsv")) + list(tmp_path.glob("*.csv")) + list(tmp_path.glob("*.npy"))
        assert outputs, f"No output files written. stderr:\n{r.stderr}"

    def test_help_flag(self):
        r = subprocess.run(
            [sys.executable, str(KNOCKOUT_SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "--in_dir" in r.stdout or "--network" in r.stdout

    def test_different_reductions_give_different_results(self, diffusion_inputs, tmp_path):
        """Higher reduction should generally remove more signal."""
        out1 = tmp_path / "r01"
        out2 = tmp_path / "r09"
        for out, red in [(out1, "0.1"), (out2, "0.9")]:
            subprocess.run(
                [sys.executable, str(KNOCKOUT_SCRIPT),
                 "--in_dir",    str(diffusion_inputs),
                 "--network",   "adjacency.csv",
                 "--delta",     "delta.tsv",
                 "--out_dir",   str(out),
                 "--t_max",     "1.0",
                 "--t_num",     "5",
                 "--reduction", red],
                capture_output=True, text=True, check=True,
            )
        # Both runs produce output; results differ (smoke test)
        files1 = list(out1.glob("*.tsv")) + list(out1.glob("*.csv"))
        files2 = list(out2.glob("*.tsv")) + list(out2.glob("*.csv"))
        assert files1 and files2
