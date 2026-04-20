#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import empirical_covariance

# R / glasso plumbing (same style you already use)
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from glasso_installation import ensure_glasso


# -----------------------------
# R function (glasso wrapper)
# -----------------------------
ro.r(
    r"""
weighted_glasso <- function(data, penalty_matrix, nobs) {
    suppressWarnings(suppressMessages(library(glasso, quietly = TRUE)))
    tryCatch({
        result <- glasso(s = as.matrix(data), rho = penalty_matrix, nobs = nobs)
        return(list(precision_matrix = result$wi))
    }, error = function(e) {
        return(list(error_message = toString(e$message)))
    })
}
"""
)


# -----------------------------
# Utilities
# -----------------------------
def load_group_tsv(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Reads genes x samples tsv, returns data as (n_samples, p_genes) array."""
    X = pd.read_csv(path, sep="\t", index_col=0)  # genes x samples
    genes = X.index.astype(str).tolist()
    samples = X.columns.astype(str).tolist()
    # transpose -> samples x genes
    data = X.T.values.astype(float)
    return data, genes, samples


def scalar_edge_curve(edge_counts_all: np.ndarray, Q: int) -> np.ndarray:
    """
    Monika-style scalar curve: average number of edges (undirected) selected per lambda.
    edge_counts_all is (p,p,lamlen) with counts in [0..Q].
    """
    # sum over matrix -> counts both (i,j) and (j,i); divide by 2 for undirected
    lamlen = edge_counts_all.shape[2]
    curve = np.zeros(lamlen, dtype=float)
    for li in range(lamlen):
        curve[li] = np.sum(edge_counts_all[:, :, li]) / (2.0 * Q)
    return curve


def find_knees_fallback(lambda_range: np.ndarray, curve: np.ndarray) -> tuple[int, int, int]:
    """
    Fallback knee logic if your Monika knee finder isn't available.
    Picks:
      - main knee = index of maximum curvature on a smoothed curve
      - left knee = 10% of main
      - right knee = 190% of main (clipped)
    """
    y = curve.astype(float)
    # simple smoothing (moving average)
    w = min(5, len(y))
    if w >= 3:
        kernel = np.ones(w) / w
        y_s = np.convolve(y, kernel, mode="same")
    else:
        y_s = y

    # approximate curvature: second derivative magnitude
    d1 = np.gradient(y_s)
    d2 = np.gradient(d1)
    main = int(np.argmax(np.abs(d2)))

    left = max(0, int(0.1 * main))
    right = min(len(lambda_range) - 1, int(1.9 * main))
    if right <= left:
        left = 0
        right = len(lambda_range) - 1
    return left, main, right


def try_import_monika_knees_and_lambda_np():
    """
    Tries to import your Monika functions if you have them.
    Returns (find_all_knee_points, estimate_lambda_np) or (None, None).
    """
    try:
        from estimate_lambdas import find_all_knee_points, estimate_lambda_np
        return find_all_knee_points, estimate_lambda_np
    except Exception:
        return None, None


def estimate_lambda_np_fallback(edge_counts_all: np.ndarray, Q: int, lambda_range: np.ndarray) -> float:
    """
    Fallback lambda_np: choose lambda where edge curve slope changes most (biggest drop).
    Not identical to Monika, but stable and sane if imports aren't available.
    """
    curve = scalar_edge_curve(edge_counts_all, Q)
    # use discrete drop
    drops = -(np.diff(curve))
    if len(drops) == 0:
        return float(lambda_range[0])
    idx = int(np.argmax(drops))  # drop between idx and idx+1
    return float(lambda_range[min(idx + 1, len(lambda_range) - 1)])


def run_final_glasso(data: np.ndarray, lambdax: float,
                     prior_matrix: np.ndarray | None = None,
                     prior_weight: float = 0.5) -> np.ndarray:
    """
    Run a final glasso on full dataset.

    Without prior: penalty = lambdax * ones(p, p)
    With prior:    penalty[i,j] = lambdax * (1 - prior_weight * prior[i,j])
                   matching the per-edge regularisation used in the subsampling.
    Returns precision matrix (p x p).
    """
    S = empirical_covariance(data)  # data is samples x genes
    nobs = data.shape[0]
    p = data.shape[1]

    if prior_matrix is not None:
        penalty = lambdax * (1.0 - prior_weight * prior_matrix.astype(float))
        # clamp to avoid zero or negative penalties
        penalty = np.clip(penalty, 1e-6, None)
    else:
        penalty = float(lambdax) * np.ones((p, p), dtype=float)

    weighted_glasso = ro.globalenv["weighted_glasso"]
    with localconverter(default_converter + numpy2ri.converter):
        res = weighted_glasso(S, penalty, nobs)

    # Convert R object to dict-like
    try:
        res = dict(res)
    except Exception:
        res = {"precision_matrix": res[0]}

    if "error_message" in res:
        msg = str(res["error_message"])
        raise RuntimeError(f"R glasso failed: {msg}")

    precision = np.array(res["precision_matrix"], dtype=float)
    return precision


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--piglasso_pkl",
        required=True,
        help="Path to __piglasso_results.pkl produced by run_piglasso.py",
    )
    parser.add_argument(
        "--expr_tsv",
        default=None,
        help="Optional: path to the filtered group TSV (genes x samples). "
             "If omitted, uses input_path stored inside the piglasso pkl.",
    )
    parser.add_argument(
        "--out_dir",
        default="burn_results/network_inference",
        help="Output directory (relative to project root unless absolute).",
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=1e-5,
        help="Threshold on |precision| to call an edge in the final adjacency.",
    )
    parser.add_argument(
        "--prior",
        default=None,
        help=(
            "Path to a .npy prior matrix. "
            "If omitted, the script auto-detects from the piglasso pkl "
            "(prior_path key). Pass 'none' to explicitly disable the prior "
            "even when the pkl records one."
        ),
    )
    parser.add_argument(
        "--prior_weight",
        type=float,
        default=None,
        help=(
            "Strength of prior influence (0–1). "
            "If omitted, taken from the piglasso pkl (prior_weight key)."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save an edge-curve plot (requires matplotlib).",
    )
    parser.add_argument(
        "--allow_install_glasso",
        action="store_true",
        help="Allow installing R glasso if missing (usually avoid on HPC).",
    )
    args = parser.parse_args()

    print("="*60)
    print("[1/7] Starting network inference pipeline")
    print("="*60)

    project_root = Path(__file__).resolve().parents[1]  # BurnInjuries/
    pig_pkl = Path(args.piglasso_pkl)
    if not pig_pkl.is_absolute():
        pig_pkl = project_root / pig_pkl
    if not pig_pkl.exists():
        raise FileNotFoundError(f"piglasso pkl not found: {pig_pkl}")

    print(f"[2/7] Ensuring R glasso is available...")
    ensure_glasso(allow_install=args.allow_install_glasso)

    print(f"[3/7] Loading piglasso results from {pig_pkl.name}...")
    with open(pig_pkl, "rb") as f:
        pig = pickle.load(f)

    edge_counts_all = pig["edge_counts_all"]
    success_counts = pig["success_counts"]
    lambda_range = np.array(pig["lambda_range"], dtype=float)
    genes = list(map(str, pig["genes"]))
    samples = list(map(str, pig["samples"]))
    Q = int(pig["Q"])
    print(f"    > Loaded edge counts: {edge_counts_all.shape}, Q={Q}, λ range: [{lambda_range[0]:.3f}, {lambda_range[-1]:.3f}]")

    # ── Resolve prior ────────────────────────────────────────────────────────
    # With prior    → PIGLasso  (stability-based GGM with biological prior)
    # Without prior → SSGLasso  (stability-based GGM, no prior)
    #
    # Priority: explicit --prior flag > pkl-recorded prior_path > no prior
    prior_matrix = None
    resolved_prior_weight = 0.5

    explicit_prior = args.prior
    if explicit_prior is not None and explicit_prior.lower() == "none":
        explicit_prior = None  # user explicitly disabled
        print("    > SSGLasso mode: prior DISABLED (--prior none)")
    elif explicit_prior is not None:
        prior_path = Path(explicit_prior)
        if not prior_path.is_absolute():
            prior_path = project_root / prior_path
        if not prior_path.exists():
            raise FileNotFoundError(f"Prior file not found: {prior_path}")
        resolved_prior_weight = args.prior_weight if args.prior_weight is not None else 0.5
        from run_piglasso_new import load_prior
        prior_matrix = load_prior(prior_path, genes)
        print(f"    > PIGLasso mode: prior from --prior flag  path={prior_path}  weight={resolved_prior_weight}")
    elif pig.get("prior_path") is not None:
        # Auto-detect: the pkl was produced by run_piglasso_new.py with a prior
        prior_path = Path(pig["prior_path"])
        if not prior_path.is_absolute():
            prior_path = project_root / prior_path
        if prior_path.exists():
            resolved_prior_weight = (
                args.prior_weight if args.prior_weight is not None
                else float(pig.get("prior_weight", 0.5))
            )
            from run_piglasso_new import load_prior
            prior_matrix = load_prior(prior_path, genes)
            print(f"    > PIGLasso mode: prior auto-detected from pkl  path={prior_path}  weight={resolved_prior_weight}")
        else:
            print(f"    > SSGLasso mode: prior recorded in pkl but file missing ({prior_path}) — running without prior")
    else:
        print("    > SSGLasso mode: no prior in pkl")

    # expression path
    print(f"[4/7] Loading expression data...")
    if args.expr_tsv is not None:
        expr_path = Path(args.expr_tsv)
        if not expr_path.is_absolute():
            expr_path = project_root / expr_path
    else:
        expr_path = Path(pig["input_path"])
        if not expr_path.is_absolute():
            expr_path = project_root / expr_path

    if not expr_path.exists():
        raise FileNotFoundError(f"Expression TSV not found: {expr_path}")

    data, genes_expr, samples_expr = load_group_tsv(expr_path)
    print(f"    > Loaded expression: {data.shape[0]} samples × {data.shape[1]} genes")

    # sanity: genes ordering should match pig genes; if not, align
    if genes_expr != genes:
        X = pd.read_csv(expr_path, sep="\t", index_col=0)  # genes x samples
        # align to pig gene order
        missing = [g for g in genes if g not in X.index]
        if missing:
            raise RuntimeError(f"{len(missing)} pig genes not found in expr TSV. Example: {missing[:5]}")
        X = X.loc[genes]
        data = X.T.values.astype(float)
        genes_expr = genes

    # -----------------------------
    # Knee points + lambda_np
    # -----------------------------
    print(f"[5/7] Computing edge stability curve and detecting knee points...")
    curve = scalar_edge_curve(edge_counts_all, Q)

    find_all_knee_points, estimate_lambda_np = try_import_monika_knees_and_lambda_np()

    if find_all_knee_points is not None:
        (
            left_knee_point,
            main_knee_point,
            right_knee_point,
            left_idx,
            main_idx,
            right_idx,
        ) = find_all_knee_points(lambda_range, edge_counts_all)
    else:
        left_idx, main_idx, right_idx = find_knees_fallback(lambda_range, curve)
        left_knee_point = float(lambda_range[left_idx])
        main_knee_point = float(lambda_range[main_idx])
        right_knee_point = float(lambda_range[right_idx])
    print(f"    > Knees: left={left_knee_point:.4f}, main={main_knee_point:.4f}, right={right_knee_point:.4f}")

    # slice to knee region (inclusive-ish)
    l_lo = int(left_idx)
    l_hi = int(right_idx) + 1
    l_lo = max(0, min(l_lo, len(lambda_range) - 1))
    l_hi = max(l_lo + 1, min(l_hi, len(lambda_range)))

    sel_lambda_range = lambda_range[l_lo:l_hi]
    sel_edge_counts_all = edge_counts_all[:, :, l_lo:l_hi]

    print(f"    > Estimating optimal λ_np from knee region [{sel_lambda_range[0]:.4f}, {sel_lambda_range[-1]:.4f}]...")
    if estimate_lambda_np is not None:
        lambda_np, _theta = estimate_lambda_np(sel_edge_counts_all, Q, sel_lambda_range)
        lambda_np = float(lambda_np)
    else:
        lambda_np = estimate_lambda_np_fallback(sel_edge_counts_all, Q, sel_lambda_range)

    lambda_wp = resolved_prior_weight if prior_matrix is not None else 0.0
    model_name = "PIGLasso" if prior_matrix is not None else "SSGLasso"
    print(f"    > Selected λ_np = {lambda_np:.6f}  (model: {model_name})")

    # -----------------------------
    # Final glasso -> precision -> adjacency
    # -----------------------------
    print(f"[6/7] Running final graphical lasso ({model_name}: n={data.shape[0]}, p={data.shape[1]}, λ={lambda_np:.6f})...")
    precision = run_final_glasso(data, lambda_np,
                                 prior_matrix=prior_matrix,
                                 prior_weight=resolved_prior_weight)
    print(f"    > Precision matrix computed: {precision.shape}")
    print(f"    > Thresholding at {args.edge_threshold} to construct adjacency matrix...")
    adj = (np.abs(precision) > float(args.edge_threshold)).astype(int)
    np.fill_diagonal(adj, 0)
    n_edges = int(np.sum(adj) / 2)  # undirected
    print(f"    > Network has {n_edges} edges (undirected)")

    adj_df = pd.DataFrame(adj, index=genes, columns=genes)

    # -----------------------------
    # Save outputs
    # -----------------------------
    print(f"[7/7] Saving results...")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pig_pkl.stem.replace("__piglasso_results", "")
    out_base = out_dir / f"{base}__inferred"

    # pkl bundle
    out_pkl = f"{out_base}.pkl"
    out_adj_csv = f"{out_base}__adjacency.csv"
    out_genes_txt = f"{out_base}__genes.txt"

    bundle = {
        "source_piglasso_pkl": str(pig_pkl),
        "expr_tsv": str(expr_path),
        "genes": genes,
        "samples": samples_expr,
        "lambda_range": lambda_range,
        "edge_curve": curve,
        "success_counts": success_counts,
        "knee": {
            "left_idx": int(left_idx),
            "main_idx": int(main_idx),
            "right_idx": int(right_idx),
            "left_lambda": float(left_knee_point),
            "main_lambda": float(main_knee_point),
            "right_lambda": float(right_knee_point),
            "slice_lo": int(l_lo),
            "slice_hi": int(l_hi),
        },
        "lambda_np": float(lambda_np),
        "lambda_wp": float(lambda_wp),
        "model": model_name,  # "PIGLasso" or "SSGLasso"
        "prior_path": str(prior_path) if prior_matrix is not None else None,
        "prior_weight": float(resolved_prior_weight) if prior_matrix is not None else None,
        "edge_threshold": float(args.edge_threshold),
        "precision_matrix": precision,
        "adjacency": adj,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f)
    print(f"    > {out_pkl}")

    adj_df.to_csv(out_adj_csv)
    print(f"    > {out_adj_csv}")
    
    Path(out_genes_txt).write_text("\n".join(genes) + "\n")
    print(f"    > {out_genes_txt}")

    print("="*60)
    print(f"✓ Network inference completed successfully!  [{model_name}]")
    print(f"  Final network: {n_edges} edges, {len(genes)} genes")
    print(f"  Lambda: λ_np={lambda_np:.6f}, λ_wp={lambda_wp:.6f}")
    print(f"  Knees: left={left_knee_point:.4f}, main={main_knee_point:.4f}, right={right_knee_point:.4f}")
    if prior_matrix is not None:
        print(f"  Prior: {prior_path}  weight={resolved_prior_weight}")
    else:
        print("  Prior: none (SSGLasso)")
    print("="*60)

    if args.plot:
        print("    > Generating edge stability curve plot...")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(lambda_range, curve, marker="o")
        plt.axvline(float(left_knee_point), linestyle="--")
        plt.axvline(float(main_knee_point), linestyle="--")
        plt.axvline(float(right_knee_point), linestyle="--")
        plt.xlabel("lambda")
        plt.ylabel("avg edges (undirected)")
        plt.title("Edge stability curve + knees")

        plot_path = out_dir / f"{base}__inferred__edge_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"    > {plot_path}")


if __name__ == "__main__":
    main()