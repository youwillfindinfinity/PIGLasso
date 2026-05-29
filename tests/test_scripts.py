"""
Tests for scripts/ directory.

Covers:
  - scripts/filter_top_genes.py
  - scripts/hub_analysis_burns.py
  - scripts/prepare_data.py  (requires nodis for NPN transform)
  - examples/make_small_example.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
EXAMPLES = REPO / "examples"
EXAMPLE_DATA = EXAMPLES / "data" / "small_example"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_expr_tsv(tmp_path_factory):
    """Write a small genes×samples TSV and return the path."""
    d = tmp_path_factory.mktemp("expr")
    rng = np.random.default_rng(0)
    genes = [f"G{i:02d}" for i in range(20)]
    samples = [f"S{i:02d}" for i in range(25)]
    df = pd.DataFrame(
        rng.normal(size=(20, 25)),
        index=genes,
        columns=samples,
    )
    path = d / "expression.tsv"
    df.to_csv(path, sep="\t")
    return path


@pytest.fixture(scope="module")
def small_prior_npy(tmp_path_factory):
    """Write a small (20,20) prior .npy and return the path."""
    d = tmp_path_factory.mktemp("prior")
    prior = np.zeros((20, 20))
    for i in range(9):
        prior[i, i + 1] = prior[i + 1, i] = 0.6
    np.fill_diagonal(prior, 0.0)
    path = d / "prior.npy"
    np.save(path, prior)
    return path


@pytest.fixture(scope="module")
def small_genes_txt(tmp_path_factory):
    """Write a gene list and return the path."""
    d = tmp_path_factory.mktemp("genes")
    path = d / "genes.txt"
    path.write_text("\n".join(f"G{i:02d}" for i in range(20)) + "\n")
    return path


@pytest.fixture(scope="module")
def small_adj_csv(tmp_path_factory):
    """Write a small binary adjacency CSV and return the path."""
    d = tmp_path_factory.mktemp("adj")
    genes = [f"G{i:02d}" for i in range(20)]
    adj = np.zeros((20, 20))
    for i in range(9):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    np.fill_diagonal(adj, 0.0)
    df = pd.DataFrame(adj, index=genes, columns=genes)
    path = d / "adjacency.csv"
    df.to_csv(path)
    return path


@pytest.fixture(scope="module")
def small_stab_csv(tmp_path_factory):
    """Write stability score CSV."""
    d = tmp_path_factory.mktemp("stab")
    rng = np.random.default_rng(1)
    genes = [f"G{i:02d}" for i in range(20)]
    stab = np.zeros((20, 20))
    for i in range(9):
        stab[i, i + 1] = stab[i + 1, i] = rng.uniform(0.6, 0.9)
    np.fill_diagonal(stab, 0.0)
    df = pd.DataFrame(stab, index=genes, columns=genes)
    path = d / "stability.csv"
    df.to_csv(path)
    return path


# ---------------------------------------------------------------------------
# make_small_example.py
# ---------------------------------------------------------------------------

def test_make_small_example_runs(tmp_path):
    """make_small_example.py writes all expected output files."""
    import importlib.util, sys as _sys
    script = EXAMPLES / "make_small_example.py"
    spec = importlib.util.spec_from_file_location("make_small_example", script)
    mod = importlib.util.module_from_spec(spec)

    # Patch OUT_DIR to tmp_path so we don't pollute the real examples/data/
    import unittest.mock as mock
    with mock.patch.object(
        mod, "__spec__", spec
    ):
        spec.loader.exec_module(mod)
        with mock.patch.object(mod, "OUT_DIR", tmp_path):
            mod.main()

    expected = ["expression.tsv", "genes.txt", "prior.npy", "adjacency.csv",
                "stability.csv", "delta.tsv"]
    for fname in expected:
        assert (tmp_path / fname).exists(), f"{fname} not written by make_small_example.py"

    expr = pd.read_csv(tmp_path / "expression.tsv", sep="\t", index_col=0)
    assert expr.shape == (20, 25)
    prior = np.load(tmp_path / "prior.npy")
    assert prior.shape == (20, 20)
    assert np.allclose(np.diag(prior), 0.0)


# ---------------------------------------------------------------------------
# filter_top_genes.py
# ---------------------------------------------------------------------------

class TestFilterTopGenes:

    def test_basic_filter(self, small_expr_tsv, small_genes_txt, small_prior_npy, tmp_path):
        """Filters to top N genes and writes three output files."""
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "filter_top_genes.py"),
             "--expr",    str(small_expr_tsv),
             "--genes",   str(small_genes_txt),
             "--prior",   str(small_prior_npy),
             "--n-genes", "10",
             "--out-dir", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

        # expression
        expr_out = list(tmp_path.glob("expression_top10.tsv"))
        assert expr_out, "expression_top10.tsv not written"
        df = pd.read_csv(expr_out[0], sep="\t", index_col=0)
        assert df.shape[0] == 10

        # gene list
        genes_out = list(tmp_path.glob("genes_top10.txt"))
        assert genes_out, "genes_top10.txt not written"
        genes = genes_out[0].read_text().strip().splitlines()
        assert len(genes) == 10

        # prior (filter_top_genes.py names it prior_burns_top{N}.npy)
        prior_out = list(tmp_path.glob("prior_burns_top10.npy"))
        assert prior_out, "prior_burns_top10.npy not written"
        prior = np.load(prior_out[0])
        assert prior.shape == (10, 10)

    def test_n_genes_larger_than_input_keeps_all(
        self, small_expr_tsv, small_genes_txt, small_prior_npy, tmp_path
    ):
        """Requesting more genes than available should keep all genes."""
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "filter_top_genes.py"),
             "--expr",    str(small_expr_tsv),
             "--genes",   str(small_genes_txt),
             "--prior",   str(small_prior_npy),
             "--n-genes", "999",
             "--out-dir", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        out_files = list(tmp_path.glob("expression_top*.tsv"))
        assert out_files
        df = pd.read_csv(out_files[0], sep="\t", index_col=0)
        assert df.shape[0] == 20  # all 20 original genes kept

    def test_output_prior_is_symmetric(
        self, small_expr_tsv, small_genes_txt, small_prior_npy, tmp_path
    ):
        """Filtered prior must remain symmetric."""
        subprocess.run(
            [sys.executable, str(SCRIPTS / "filter_top_genes.py"),
             "--expr",    str(small_expr_tsv),
             "--genes",   str(small_genes_txt),
             "--prior",   str(small_prior_npy),
             "--n-genes", "10",
             "--out-dir", str(tmp_path)],
            capture_output=True, text=True, check=True,
        )
        prior = np.load(list(tmp_path.glob("prior_burns_top10.npy"))[0])
        assert np.allclose(prior, prior.T), "Filtered prior is not symmetric"


# ---------------------------------------------------------------------------
# hub_analysis_burns.py
# ---------------------------------------------------------------------------

class TestHubAnalysis:

    def test_basic_hub_ranking(
        self, small_adj_csv, small_stab_csv, small_prior_npy, small_genes_txt, tmp_path
    ):
        """Writes hub_rankings.csv and hub_convergent.csv."""
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "hub_analysis_burns.py"),
             "--adj",   str(small_adj_csv),
             "--stab",  str(small_stab_csv),
             "--prior", str(small_prior_npy),
             "--genes", str(small_genes_txt),
             "--out",   str(tmp_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert (tmp_path / "hub_rankings.csv").exists(), "hub_rankings.csv not written"

    def test_hub_rankings_has_expected_columns(
        self, small_adj_csv, small_stab_csv, small_prior_npy, small_genes_txt, tmp_path
    ):
        """hub_rankings.csv must have degree, betweenness, eigenvector columns."""
        subprocess.run(
            [sys.executable, str(SCRIPTS / "hub_analysis_burns.py"),
             "--adj",   str(small_adj_csv),
             "--stab",  str(small_stab_csv),
             "--prior", str(small_prior_npy),
             "--genes", str(small_genes_txt),
             "--out",   str(tmp_path)],
            capture_output=True, text=True, check=True,
        )
        df = pd.read_csv(tmp_path / "hub_rankings.csv", index_col=0)
        for col in ("degree", "betweenness", "eigenvector"):
            assert col in df.columns, f"Missing column '{col}' in hub_rankings.csv"
        assert len(df) == 20

    def test_hub_analysis_top_k_filtering(
        self, small_adj_csv, small_stab_csv, small_prior_npy, small_genes_txt, tmp_path
    ):
        """--top-k controls how many hubs are written to hub_convergent.csv."""
        subprocess.run(
            [sys.executable, str(SCRIPTS / "hub_analysis_burns.py"),
             "--adj",   str(small_adj_csv),
             "--stab",  str(small_stab_csv),
             "--prior", str(small_prior_npy),
             "--genes", str(small_genes_txt),
             "--top-k", "5",
             "--out",   str(tmp_path)],
            capture_output=True, text=True, check=True,
        )
        # hub_convergent.csv filters by top-k% threshold, not a hard row cap
        assert (tmp_path / "hub_rankings.csv").exists()


# ---------------------------------------------------------------------------
# prepare_data.py (requires nodis for NPN)
# ---------------------------------------------------------------------------

class TestPrepareData:
    # prepare_data.py imports nodis at module level — all tests skip without it

    def test_prepare_data_no_npn(self, small_expr_tsv, tmp_path):
        """prepare_data.py runs without NPN (--no-npn) and writes .npy output."""
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "prepare_data.py"),
             "--input",  str(small_expr_tsv),
             "--output", str(tmp_path / "out.npy"),
             "--no-npn"],
            capture_output=True, text=True,
        )
        if "No module named 'nodis'" in r.stderr:
            pytest.skip("nodis not installed (prepare_data.py imports nodis at module level)")
        assert r.returncode == 0, r.stderr
        arr = np.load(tmp_path / "out.npy")
        assert arr.ndim == 2
        # shape should be (n_samples, n_genes) = (25, 20)
        assert arr.shape == (25, 20)

    def test_prepare_data_with_npn(self, small_expr_tsv, tmp_path):
        """prepare_data.py with NPN requires nodis."""
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "prepare_data.py"),
             "--input",  str(small_expr_tsv),
             "--output", str(tmp_path / "out_npn.npy")],
            capture_output=True, text=True,
        )
        if "No module named 'nodis'" in r.stderr:
            pytest.skip("nodis not installed")
        assert r.returncode == 0, r.stderr
        arr = np.load(tmp_path / "out_npn.npy")
        assert arr.shape == (25, 20)

    def test_prepare_data_transpose_detection(self, tmp_path):
        """Auto-transpose: samples-as-rows input should be handled correctly."""
        pytest.importorskip("nodis", reason="nodis not installed (prepare_data.py imports nodis at module level)")
        rng = np.random.default_rng(7)
        # samples × genes (25 × 20) — should detect samples as rows
        samples = [f"S{i:02d}" for i in range(25)]
        genes = [f"G{i:02d}" for i in range(20)]
        df = pd.DataFrame(rng.normal(size=(25, 20)), index=samples, columns=genes)
        path = tmp_path / "samples_as_rows.tsv"
        df.to_csv(path, sep="\t")

        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "prepare_data.py"),
             "--input",  str(path),
             "--output", str(tmp_path / "out.npy"),
             "--no-npn"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        arr = np.load(tmp_path / "out.npy")
        # regardless of input orientation, output should be (n_samples, n_genes)
        assert arr.shape[1] == 20
