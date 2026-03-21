from __future__ import annotations

import os
import sys
import random
import warnings
import numpy as np

from random import sample
from sklearn.covariance import empirical_covariance
from tqdm import tqdm
from joblib import Parallel, delayed

import rpy2.situation
os.environ["R_HOME"] = rpy2.situation.get_r_home()

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter


def ensure_r_glasso_available(allow_install: bool = False) -> None:
    try:
        importr("glasso")
        return
    except Exception:
        if not allow_install:
            raise RuntimeError(
                "R package 'glasso' is not available in this R environment.\n"
                "Fix options:\n"
                "  • In R: install.packages('glasso')\n"
                "  • On HPC: load an R module that already includes it, or install to a user library.\n"
                "You can also rerun with allow_install=True if your environment permits installs."
            )
    ro.r("install.packages('glasso', repos='https://cloud.r-project.org')")
    importr("glasso")


# -----------------------------
# Worker debug (ONCE per process)
# -----------------------------
_PRINTED_WORKER_PIDS: set[int] = set()

def debug_worker_context_once(tag: str = "worker") -> None:
    """
    Print worker PID + R PID + SLURM cpus + CPU affinity once per *process*.
    Goes to stderr so it lands in Slurm .err.
    """
    pid = os.getpid()
    if pid in _PRINTED_WORKER_PIDS:
        return
    _PRINTED_WORKER_PIDS.add(pid)

    try:
        r_pid = int(ro.r("Sys.getpid()")[0])
    except Exception:
        r_pid = -1

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "N/A")
    slurm_jobid = os.environ.get("SLURM_JOB_ID", "N/A")

    try:
        cpu = os.sched_getcpu()
    except Exception:
        cpu = None

    try:
        aff = sorted(os.sched_getaffinity(0))
    except Exception:
        aff = None

    msg = f"[WORKER][{tag}] pid={pid} r_pid={r_pid} SLURM_JOB_ID={slurm_jobid} SLURM_CPUS_PER_TASK={slurm_cpus}"
    if cpu is not None:
        msg += f" cpu={cpu}"
    if aff is not None:
        msg += f" affinity={aff[:16]}{'...' if len(aff) > 16 else ''} (n={len(aff)})"

    print(msg, flush=True, file=sys.stderr)


# -----------------------------
# R init once per process
# -----------------------------
_R_READY = False

def _init_r_once() -> None:
    """Initialize embedded R bits once per *process*."""
    global _R_READY
    if _R_READY:
        return

    ro.r(r'''
    weighted_glasso <- function(S, rho, nobs) {
      suppressWarnings(suppressMessages(library(glasso, quietly=TRUE)))
      tryCatch({
        result <- glasso(s = as.matrix(S), rho = rho, nobs = nobs)
        return(list(precision_matrix = result$wi))
      }, error = function(e) {
        return(list(error_message = toString(e$message)))
      })
    }
    ''')
    _R_READY = True


class QJSweeper:
    """
    Minimal QJ sweeper: subsample -> empirical cov -> glasso -> edge_counts.
    Accepts an optional prior_matrix for compatibility, but does not use it.
    """
    def __init__(
        self,
        data: np.ndarray,
        b: int,
        Q: int,
        prior_matrix: np.ndarray | None = None,
        rank: int = 0,
        size: int = 1,
        seed: int = 42,
        n_jobs: int = 1,
    ):
        self.data = data
        self.prior_matrix = prior_matrix
        self.p = int(data.shape[1])
        self.n = int(data.shape[0])
        self.Q = int(Q)
        self.n_jobs = int(n_jobs)

        if not isinstance(b, (int, np.integer)):
            raise TypeError(f"b must be int, got {type(b)}")
        self.b = int(b)

        self.subsample_indices = self.get_subsamples_indices(self.n, self.b, self.Q, rank, size, seed)

    @staticmethod
    def get_subsamples_indices(n: int, b: int, Q: int, rank: int, size: int, seed: int):
        if b >= n:
            raise ValueError("b should be less than the number of samples n.")

        random.seed(seed + rank)
        subsamples_indices = set()
        subsamples_per_rank = Q // size
        attempts = 0
        max_attempts = int(1e6)

        while len(subsamples_indices) < subsamples_per_rank and attempts < max_attempts:
            new_comb = tuple(sorted(sample(range(n), b)))
            subsamples_indices.add(new_comb)
            attempts += 1

        if attempts >= max_attempts:
            raise RuntimeError(f"Rank {rank}: max attempts reached generating subsamples.")

        return list(subsamples_indices)

    def optimize_for_subsample_and_lambda(self, subsamp_idx, lambdax: float):
        _init_r_once()
        # NOTE: do NOT print worker banner here; it gets called for every lambda.

        sub = self.data[np.array(subsamp_idx), :]
        S = empirical_covariance(sub)
        nobs = int(sub.shape[0])
        p = int(self.p)

        weighted_glasso = ro.globalenv["weighted_glasso"]

        # Try sklearn first
        try:
            from sklearn.covariance import graphical_lasso as sk_graphical_lasso
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                cov_, precision = sk_graphical_lasso(S, alpha=float(lambdax), max_iter=100, tol=1e-3)
            edge_counts = (np.abs(precision) > 1e-5).astype(np.int8)
            return edge_counts, 1
        except Exception:
            pass

        # Fall back to R
        try:
            with localconverter(default_converter + numpy2ri.converter):
                res = weighted_glasso(S, float(lambdax), nobs)

            try:
                res = dict(res)
            except Exception:
                res = {"precision_matrix": res[0]}

            if "error_message" in res:
                err = res["error_message"]
                err_str = str(err[0]) if hasattr(err, "__len__") else str(err)
                if err_str and err_str != "NULL":
                    print(f"[R ERROR] {err_str}", flush=True, file=sys.stderr)
                return np.zeros((p, p), dtype=np.int8), 0

            precision = np.asarray(res["precision_matrix"], dtype=float)
            edge_counts = (np.abs(precision) > 1e-5).astype(np.int8)
            return edge_counts, 1

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ERROR] glasso call failed: {e}", flush=True, file=sys.stderr)
            return np.zeros((p, p), dtype=np.int8), 0

    def _process_one_subsample(self, subsamp_idx, lambda_range: np.ndarray):
        """
        Process a single subsample across all lambda values.
        This function runs inside the joblib worker process,
        so printing here is "once per process" when guarded by PID.
        """
        debug_worker_context_once(tag="subsample_worker")

        n_lams = len(lambda_range)
        edge_counts_sub = np.zeros((self.p, self.p, n_lams), dtype=float)
        success_sub = np.zeros(n_lams, dtype=float)

        for li, lambdax in enumerate(lambda_range):
            edge_counts, ok = self.optimize_for_subsample_and_lambda(subsamp_idx, float(lambdax))
            edge_counts_sub[:, :, li] += edge_counts
            success_sub[li] += ok

        return edge_counts_sub, success_sub

    def run_subsample_optimization(self, lambda_range: np.ndarray):
        n_lams = len(lambda_range)
        total_calls = len(self.subsample_indices) * n_lams

        print(
            f"[INFO] Starting QJSweeper: "
            f"Q={len(self.subsample_indices)}, "
            f"lamlen={n_lams}, "
            f"total glasso calls={total_calls}, "
            f"n_jobs={self.n_jobs}",
            flush=True,
            file=sys.stderr,
        )

        edge_counts_all = np.zeros((self.p, self.p, n_lams), dtype=float)
        success_counts = np.zeros(n_lams, dtype=float)

        if self.n_jobs == 1:
            debug_worker_context_once(tag="main_process")
            for subsamp_idx in tqdm(self.subsample_indices, desc="Subsamples", file=sys.stderr):
                for li, lambdax in enumerate(lambda_range):
                    edge_counts, ok = self.optimize_for_subsample_and_lambda(subsamp_idx, float(lambdax))
                    edge_counts_all[:, :, li] += edge_counts
                    success_counts[li] += ok
        else:
            print(
                f"[INFO] Parallelizing across {len(self.subsample_indices)} subsamples "
                f"with {self.n_jobs} processes",
                flush=True,
                file=sys.stderr,
            )

            results = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
                prefer="processes",
                batch_size=1,
                verbose=0,
            )(
                delayed(self._process_one_subsample)(subsamp_idx, lambda_range)
                for subsamp_idx in tqdm(self.subsample_indices, desc="Subsamples", file=sys.stderr)
            )

            for edge_counts_sub, success_sub in results:
                edge_counts_all += edge_counts_sub
                success_counts += success_sub

        print("[INFO] QJSweeper finished.", flush=True, file=sys.stderr)
        return edge_counts_all, success_counts