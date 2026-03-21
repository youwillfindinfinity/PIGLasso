"""
PIGLasso test suite.

Tests the PIGLassoEstimator (from nodis) and the piglasso CLI.
Kept fast: uses small n/p, Q=5 subsamples.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import pathlib

import numpy as np
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_X():
    """50 samples × 20 genes, block-sparse covariance structure."""
    rng = np.random.default_rng(0)
    p = 20
    # Build a simple precision matrix with off-diag block
    Omega = np.eye(p)
    for i in range(0, 10):
        Omega[i, i + 1] = Omega[i + 1, i] = 0.4
    Omega += 0.1 * np.eye(p)  # ensure PD
    Sigma = np.linalg.inv(Omega)
    return rng.multivariate_normal(np.zeros(p), Sigma, size=50)


@pytest.fixture(scope="module")
def fitted_estimator(small_X):
    from nodis.estimators.piglasso import PIGLassoEstimator
    est = PIGLassoEstimator(Q=5, n_lambda=5, seed=42)
    est.fit(small_X)
    return est


# ---------------------------------------------------------------------------
# Import / version
# ---------------------------------------------------------------------------

def test_import():
    import piglasso
    assert hasattr(piglasso, "__version__")


def test_nodis_dependency_available():
    import nodis
    assert hasattr(nodis, "__version__")


# ---------------------------------------------------------------------------
# PIGLassoEstimator — unit tests
# ---------------------------------------------------------------------------

def test_fit_returns_self(small_X):
    from nodis.estimators.piglasso import PIGLassoEstimator
    est = PIGLassoEstimator(Q=5, n_lambda=5, seed=0)
    ret = est.fit(small_X)
    assert ret is est


def test_stability_shape(small_X, fitted_estimator):
    p = small_X.shape[1]
    assert fitted_estimator.precision_.shape == (p, p)


def test_stability_range(fitted_estimator):
    s = fitted_estimator.precision_
    assert s.min() >= 0.0
    assert s.max() <= 1.0


def test_stability_diagonal_zero(fitted_estimator):
    p = fitted_estimator.precision_.shape[0]
    assert np.allclose(np.diag(fitted_estimator.precision_), 0.0)


def test_stability_symmetric(fitted_estimator):
    s = fitted_estimator.precision_
    assert np.allclose(s, s.T)


def test_adjacency_binary(fitted_estimator):
    adj = fitted_estimator.get_adjacency()
    assert set(np.unique(adj)).issubset({0, 1})


def test_adjacency_no_self_loops(fitted_estimator):
    adj = fitted_estimator.get_adjacency()
    assert np.all(np.diag(adj) == 0)


def test_adjacency_symmetric(fitted_estimator):
    adj = fitted_estimator.get_adjacency()
    assert np.array_equal(adj, adj.T)


def test_threshold_override(fitted_estimator):
    adj_loose = fitted_estimator.get_adjacency(threshold=0.1)
    adj_strict = fitted_estimator.get_adjacency(threshold=0.9)
    assert adj_loose.sum() >= adj_strict.sum()


def test_adjacency_before_fit_raises():
    from nodis.estimators.piglasso import PIGLassoEstimator
    est = PIGLassoEstimator()
    with pytest.raises(RuntimeError, match="fit"):
        est.get_adjacency()


def test_precision_before_fit_raises():
    from nodis.estimators.piglasso import PIGLassoEstimator
    est = PIGLassoEstimator()
    with pytest.raises(RuntimeError, match="fit"):
        _ = est.precision_


def test_reproducibility(small_X):
    from nodis.estimators.piglasso import PIGLassoEstimator
    e1 = PIGLassoEstimator(Q=5, n_lambda=5, seed=7).fit(small_X)
    e2 = PIGLassoEstimator(Q=5, n_lambda=5, seed=7).fit(small_X)
    assert np.allclose(e1.precision_, e2.precision_)


def test_different_seeds_differ(small_X):
    from nodis.estimators.piglasso import PIGLassoEstimator
    e1 = PIGLassoEstimator(Q=5, n_lambda=5, seed=1).fit(small_X)
    e2 = PIGLassoEstimator(Q=5, n_lambda=5, seed=99).fit(small_X)
    assert not np.allclose(e1.precision_, e2.precision_)


def test_b_perc_too_large_raises(small_X):
    from nodis.estimators.piglasso import PIGLassoEstimator
    est = PIGLassoEstimator(Q=5, b_perc=1.0, seed=0)
    with pytest.raises(ValueError, match="b_perc"):
        est.fit(small_X)


def test_parallel_matches_sequential(small_X):
    from nodis.estimators.piglasso import PIGLassoEstimator
    e_seq = PIGLassoEstimator(Q=5, n_lambda=5, seed=0, n_jobs=1).fit(small_X)
    e_par = PIGLassoEstimator(Q=5, n_lambda=5, seed=0, n_jobs=2).fit(small_X)
    assert np.allclose(e_seq.precision_, e_par.precision_, atol=1e-6)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "piglasso.cli", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "PIGLasso" in r.stdout


def test_cli_run_help():
    r = subprocess.run([sys.executable, "-m", "piglasso.cli", "run", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--data" in r.stdout


def test_cli_prior_help():
    r = subprocess.run([sys.executable, "-m", "piglasso.cli", "prior", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--step" in r.stdout


def test_cli_diffuse_help():
    r = subprocess.run([sys.executable, "-m", "piglasso.cli", "diffuse", "--help"],
                       capture_output=True, text=True)
    # delegates to node_knockout.py --help
    assert r.returncode == 0


def test_cli_knockout_help():
    r = subprocess.run([sys.executable, "-m", "piglasso.cli", "knockout", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0


def test_cli_run_end_to_end(small_X, tmp_path):
    """Write a small TSV, run `piglasso run`, check output files exist."""
    p = small_X.shape[1]
    gene_names = [f"G{i:02d}" for i in range(p)]
    df = pd.DataFrame(small_X.T, index=gene_names,
                      columns=[f"S{i}" for i in range(small_X.shape[0])])
    tsv = tmp_path / "expr.tsv"
    df.to_csv(tsv, sep="\t")

    out_dir = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "piglasso.cli", "run",
         "--data", str(tsv),
         "--n-subsamples", "5",
         "--lambda-len", "5",
         "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out_dir / "expr_adjacency.csv").exists()
    assert (out_dir / "expr_stability.csv").exists()
    adj = pd.read_csv(out_dir / "expr_adjacency.csv", index_col=0)
    assert adj.shape == (p, p)
    assert set(adj.values.ravel().tolist()).issubset({0, 1, 0.0, 1.0})
