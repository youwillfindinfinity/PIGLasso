# Changelog

All notable changes to PIGLasso are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.1.0] — 2026-05-30

Initial release of the PIGLasso burn-injury transcriptomics pipeline.

### Added

**Core pipeline**
- Stability-selection GGM inference via `PIGLassoEstimator` (from NODIS) with per-edge biological prior weighting: `λ_ij = λ · (1 − α · P_ij)`
- Prior construction pipeline (steps 1 → 2a/2b/2c → 3 → 4) for STRING PPI and KEGG pathway co-membership priors
- Network diffusion using heat-kernel propagation to identify transcriptional hubs (`pipeline_src/diffusion/`)
- In silico node knockout analysis for hub essentiality scoring (`pipeline_src/knockouts/`)
- Knockout–pathway cross-reference module linking hub knockouts to enriched pathways (`pipeline_src/knockouts/crossref/`)
- Pathway annotation and GSEA on stability scores and acute-phase delta vectors (`pipeline_src/pathways/`)

**Datasets**
- Preprocessing pipelines for GSE182616 (burn injury, Agilent GPL17077) and GSE236713 (healthy + SIRS, Agilent GPL17077)
- Gene filtering to top-5000 genes by variance (`pipeline_src/filter_top_genes.py`)
- Hub gene analysis for burn data (`pipeline_src/hub_analysis_burns.py`)

**CLI** (`piglasso run / prior / diffuse / knockout`)
- `run` — stability-selection inference on a TSV expression matrix with optional prior
- `prior --step {1,2a,2b,2c,3,4}` — stepwise prior construction
- `diffuse` — heat-kernel diffusion signal propagation
- `knockout` — node-knockout robustness analysis

**HPC**
- Individual SLURM jobs for each prior construction step (`jobs/prior_step*.job`)
- SLURM submission chain script (`jobs/submit_prior_chain.sh`)
- Burn inference, diffusion, DREAM5 benchmark, and synthetic benchmark jobs

**Testing**
- 23-test suite (`tests/test_piglasso.py`) covering estimator correctness, CLI smoke tests, and end-to-end run
- 6 tests pass without NODIS; all 23 pass with NODIS installed

**Packaging**
- `pyproject.toml` with setuptools, pytest config, and optional `[r]` and `[dev]` extras
- `requirements.txt` mirroring `pyproject.toml` dependencies

---

[Unreleased]: https://github.com/youwillfindinfinity/piglasso/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/youwillfindinfinity/piglasso/releases/tag/v0.1.0
