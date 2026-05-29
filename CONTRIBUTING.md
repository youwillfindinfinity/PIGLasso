# Contributing to PIGLasso

Thank you for your interest in contributing. PIGLasso is a research software project developed at Amsterdam UMC. We welcome bug reports, documentation improvements, and well-scoped code contributions.

---

## Scope

PIGLasso is the **burn-injury application pipeline** built on top of [NODIS](https://github.com/youwillfindinfinity/nodis). Contributions in scope for this repository:

- Bug fixes in the prior construction pipeline, diffusion, knockouts, or pathways modules
- Preprocessing improvements for GSE182616 / GSE236713 or compatible datasets
- New SLURM job templates for HPC reproducibility
- Documentation, docstring, and example improvements
- Additional CLI options that do not duplicate NODIS inference functionality

Contributions that belong in NODIS instead (out of scope here):
- Changes to `PIGLassoEstimator`, `SklearnGLasso`, or any estimator class
- Edge p-values, FDR control, confidence intervals, or benchmarking logic
- New GGM inference methods

If you are unsure, open an issue first.

---

## Getting started

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/piglasso
cd piglasso

# 2. Install NODIS (required dependency)
pip install -e ../NODIS   # or: pip install git+https://github.com/youwillfindinfinity/nodis.git

# 3. Install PIGLasso in editable mode with dev extras
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 4. Verify tests pass
pytest tests/ -v
```

---

## Workflow

1. Create a branch: `git checkout -b fix/your-fix-name`
2. Make your changes
3. Run the full test suite — **all 23 tests must pass**: `pytest tests/ -v`
4. Commit with a clear message (present tense, imperative: "Fix prior step 2c path on Windows")
5. Open a pull request against `master` with a description of what changed and why

---

## Code style

- Python: follow the existing style (no formatter is enforced, but keep it close to PEP 8)
- No unnecessary inline comments — code should be self-explanatory; use docstrings
- No `numpy.matrix` — use `np.ndarray` throughout
- No hardcoded paths — all paths via CLI arguments or `pathlib`
- R scripts: use `suppressPackageStartupMessages()` and explicit `library()` calls at the top

---

## Reporting bugs

Open a GitHub issue with:
- PIGLasso version (`pip show piglasso`)
- NODIS version (`pip show nodis`)
- Python version and OS
- Minimal reproducible example or the full traceback
- What you expected vs. what happened

---

## Contact

For research questions or to discuss larger contributions before starting work, contact the corresponding author: Roland V. Bumbuc — rbumbuc@gmail.com
