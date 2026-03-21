# src/glasso_installation.py
from __future__ import annotations

from rpy2.robjects import r
from rpy2.robjects.packages import isinstalled


def glasso_is_available() -> bool:
    """Return True if R package 'glasso' is installed/available."""
    try:
        return bool(isinstalled("glasso"))
    except Exception:
        # Fallback to R-side check if rpy2 has issues
        try:
            return bool(
                r('suppressWarnings(suppressMessages(requireNamespace("glasso", quietly=TRUE)))')[0]
            )
        except Exception:
            return False


def ensure_glasso(allow_install: bool = False) -> None:
    """
    Ensure R package 'glasso' is available.

    - If available: prints a message and returns.
    - If missing and allow_install=False: raises a clear error explaining what to do.
    - If missing and allow_install=True: attempts install.packages("glasso").

    Notes:
      * On HPC compute nodes this often fails (no internet / permissions).
      * Prefer installing once on a login node into a user library, or loading a module.
    """
    if glasso_is_available():
        print("glasso package is already installed")
        return

    if not allow_install:
        raise RuntimeError(
            "R package 'glasso' is not available in this R environment.\n"
            "Recommended fixes:\n"
            "  1) On your laptop: open R and run: install.packages('glasso')\n"
            "  2) On HPC: install it once into your user R library on a login node,\n"
            "     or load an R module that includes it.\n"
            "If you are sure this environment can install packages, rerun with allow_install=True."
        )

    print("Installing glasso package (allow_install=True)...")
    try:
        r(
            r'''
            suppressWarnings(suppressMessages({
                install.packages("glasso", repos="https://cloud.r-project.org/", quiet=TRUE)
            }))
            '''
        )
    except Exception as e:
        raise RuntimeError(f"Attempted to install 'glasso' but failed: {e}")

    if not glasso_is_available():
        raise RuntimeError("Tried to install 'glasso' but it still isn't available afterward.")

    print("glasso package successfully installed")