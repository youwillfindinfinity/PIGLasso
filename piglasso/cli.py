"""
PIGLasso command-line interface.

Commands
--------
run       — run PIGLasso stability-selection inference on a TSV expression matrix
prior     — build a biological prior matrix from control expression data
diffuse   — run network diffusion signal analysis on an inferred network
knockout  — perform node-knockout robustness analysis
"""

from __future__ import annotations

import pathlib
import sys

import click


@click.group()
@click.version_option()
def main() -> None:
    """PIGLasso: Prior-Informed Graphical Lasso for transcriptomics networks."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@main.command()
@click.option("--data", required=True,
              help="Path to expression matrix TSV (genes × samples).")
@click.option("--prior", default=None,
              help="Path to prior matrix .npy file (optional). If omitted, runs unweighted.")
@click.option("--Q", default=200, show_default=True,
              help="Number of subsamples.")
@click.option("--b-perc", default=0.65, show_default=True,
              help="Subsample fraction of n.")
@click.option("--lambda-lo", default=0.05, show_default=True,
              help="Lower bound of lambda grid.")
@click.option("--lambda-hi", default=0.30, show_default=True,
              help="Upper bound of lambda grid.")
@click.option("--lambda-len", default=20, show_default=True,
              help="Number of lambda grid points.")
@click.option("--pi-thr", default=0.5, show_default=True,
              help="Stability threshold for edge selection.")
@click.option("--n-jobs", default=1, show_default=True,
              help="Parallel workers (-1 = all CPUs).")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--out", default="results/piglasso/", show_default=True,
              help="Output directory.")
def run(data, prior, q, b_perc, lambda_lo, lambda_hi, lambda_len,
        pi_thr, n_jobs, seed, out):
    """Run PIGLasso stability-selection inference on an expression matrix."""
    import numpy as np
    import pandas as pd
    from nodis.estimators.piglasso import PIGLassoEstimator

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data, sep="\t", index_col=0)
    X = df.T.values.astype(float)
    click.echo(f"Loaded: {X.shape[0]} samples × {X.shape[1]} genes  ({data})")

    prior_W = None
    if prior:
        prior_W = np.load(prior)
        click.echo(f"Prior loaded: {prior_W.shape}  ({prior})")

    est = PIGLassoEstimator(
        Q=q,
        b_perc=b_perc,
        lambda_lo=lambda_lo,
        lambda_hi=lambda_hi,
        n_lambda=lambda_len,
        pi_thr=pi_thr,
        n_jobs=n_jobs,
        seed=seed,
        lambda_wp=1.0 if prior_W is not None else 0.0,
        W=prior_W,
    )
    est.fit(X)
    adj = est.get_adjacency()

    stem = pathlib.Path(data).stem
    adj_path = out_dir / f"{stem}_adjacency.csv"
    stab_path = out_dir / f"{stem}_stability.csv"

    gene_names = list(df.index)
    pd.DataFrame(adj, index=gene_names, columns=gene_names).to_csv(adj_path)
    pd.DataFrame(est.precision_, index=gene_names, columns=gene_names).to_csv(stab_path)

    n_edges = int(adj.sum()) // 2
    click.echo(f"Edges selected (pi >= {pi_thr}): {n_edges}")
    click.echo(f"Adjacency  → {adj_path}")
    click.echo(f"Stability  → {stab_path}")


# ---------------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------------

@main.command()
@click.option("--step", required=True,
              type=click.Choice(["1", "2a", "2b", "2c", "3", "4"]),
              help="Prior construction step to run.")
@click.option("--pipeline-dir",
              default=str(pathlib.Path(__file__).parent.parent / "pipeline_src"),
              show_default=True,
              help="Path to pipeline_src/ directory.")
def prior(step, pipeline_dir):
    """Build biological prior matrix (Steps 1–4 from priorPIGLASSO.md)."""
    import subprocess
    script = pathlib.Path(pipeline_dir) / "build_prior.py"
    if not script.exists():
        click.echo(f"ERROR: build_prior.py not found at {script}", err=True)
        sys.exit(1)
    result = subprocess.run(
        [sys.executable, str(script), "--step", step],
        cwd=str(pathlib.Path(pipeline_dir)),
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# diffuse
# ---------------------------------------------------------------------------

@main.command()
@click.option("--adjacency", required=True,
              help="Path to adjacency matrix CSV (output of `piglasso run`).")
@click.option("--delta", required=True,
              help="Path to delta signal vector .npy or CSV.")
@click.option("--out", default="results/diffusion/", show_default=True,
              help="Output directory.")
def diffuse(adjacency, delta, out):
    """Run network diffusion signal propagation on a PIGLasso inferred network."""
    import numpy as np
    import pandas as pd

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adj = pd.read_csv(adjacency, index_col=0).values.astype(float)
    delta_vec = (
        np.load(delta) if delta.endswith(".npy")
        else pd.read_csv(delta, header=None).values.ravel()
    )

    pipeline_dir = pathlib.Path(__file__).parent.parent / "pipeline_src"
    sys.path.insert(0, str(pipeline_dir / "diffusion"))
    from diffusion_signal import run_diffusion  # type: ignore[import]

    result = run_diffusion(adj, delta_vec, out_dir=out_dir)
    click.echo(f"Diffusion results → {out_dir}")


# ---------------------------------------------------------------------------
# knockout
# ---------------------------------------------------------------------------

@main.command()
@click.option("--adjacency", required=True,
              help="Path to adjacency matrix CSV.")
@click.option("--out", default="results/knockouts/", show_default=True,
              help="Output directory.")
def knockout(adjacency, out):
    """Run node-knockout robustness analysis on an inferred network."""
    import pandas as pd

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adj = pd.read_csv(adjacency, index_col=0)

    pipeline_dir = pathlib.Path(__file__).parent.parent / "pipeline_src"
    sys.path.insert(0, str(pipeline_dir / "knockouts"))
    from node_knockout import run_knockouts  # type: ignore[import]

    run_knockouts(adj, out_dir=out_dir)
    click.echo(f"Knockout results → {out_dir}")


if __name__ == "__main__":
    main()
