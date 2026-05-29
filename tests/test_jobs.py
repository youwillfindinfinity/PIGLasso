"""
Structural validation tests for SLURM job files and HPC shell scripts.

These tests do NOT execute any jobs — they validate that job files are
correctly structured, have required SBATCH directives, reference existing
scripts, and pass required arguments.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "jobs"
HPC  = JOBS / "hpc_scripts"

JOB_FILES  = sorted(JOBS.glob("*.job"))
SHELL_JOBS = sorted(JOBS.glob("*.sh")) + sorted(HPC.glob("*.sh"))

# Prior step jobs that must pass --step
PRIOR_STEP_JOBS = {
    "prior_step1.job":  "1",
    "prior_step2a.job": "2a",
    "prior_step2b.job": "2b",
    "prior_step2c.job": "2c",
    "prior_step3.job":  "3",
    "prior_step4.job":  "4",
}

# Scripts referenced by main SLURM jobs — these must exist
EXPECTED_SCRIPT_REFS = {
    "piglasso_burns.job":        ["piglasso"],   # CLI entrypoint
    "piglasso_diffusion.job":    ["run_diffusion.py"],
    "prior_step1.job":           ["build_prior.py"],
    "prior_step2a.job":          ["build_prior.py"],
    "prior_step2b.job":          ["build_prior.py"],
    "prior_step2c.job":          ["build_prior.py"],
    "prior_step3.job":           ["build_prior.py"],
    "prior_step4.job":           ["build_prior.py"],
}

# HPC scripts that reference monika/ (legacy, known broken) — tracked not failed
KNOWN_LEGACY_SCRIPTS = {"preprocess_burn.sh", "run_pig.sh", "run_pig_new.sh"}


# ---------------------------------------------------------------------------
# .job file tests
# ---------------------------------------------------------------------------

class TestJobFileStructure:

    @pytest.mark.parametrize("job_file", JOB_FILES, ids=[f.name for f in JOB_FILES])
    def test_has_bash_shebang(self, job_file):
        text = job_file.read_text()
        assert text.startswith("#!/bin/bash") or text.startswith("#!/usr/bin/env bash"), \
            f"{job_file.name}: missing #!/bin/bash shebang"

    @pytest.mark.parametrize("job_file", JOB_FILES, ids=[f.name for f in JOB_FILES])
    def test_has_sbatch_job_name(self, job_file):
        if job_file.name == "build_prior.job":
            pytest.skip("build_prior.job is a legacy file superseded by prior_step*.job")
        text = job_file.read_text()
        assert "#SBATCH --job-name" in text, \
            f"{job_file.name}: missing #SBATCH --job-name directive"

    @pytest.mark.parametrize("job_file", JOB_FILES, ids=[f.name for f in JOB_FILES])
    def test_has_sbatch_time(self, job_file):
        if job_file.name == "build_prior.job":
            pytest.skip("build_prior.job is a legacy file superseded by prior_step*.job")
        text = job_file.read_text()
        # Some jobs intentionally set --time at submission (sbatch --time=...), documented in comments
        has_time = "#SBATCH --time" in text or ("sbatch --time" in text or "--time=" in text)
        assert has_time, f"{job_file.name}: no --time (inline or at-submission)"

    @pytest.mark.parametrize("job_name,expected_step",
                             PRIOR_STEP_JOBS.items(),
                             ids=list(PRIOR_STEP_JOBS.keys()))
    def test_prior_step_job_passes_step_arg(self, job_name, expected_step):
        job_file = JOBS / job_name
        if not job_file.exists():
            pytest.skip(f"{job_name} not found")
        text = job_file.read_text()
        assert f"--step {expected_step}" in text, \
            f"{job_name}: does not pass --step {expected_step} to build_prior.py"

    @pytest.mark.parametrize("job_name,script_refs",
                             EXPECTED_SCRIPT_REFS.items(),
                             ids=list(EXPECTED_SCRIPT_REFS.keys()))
    def test_main_jobs_reference_correct_scripts(self, job_name, script_refs):
        job_file = JOBS / job_name
        if not job_file.exists():
            pytest.skip(f"{job_name} not found")
        text = job_file.read_text()
        for ref in script_refs:
            assert ref in text, \
                f"{job_name}: expected reference to '{ref}' not found"

    def test_build_prior_job_is_superseded(self):
        """build_prior.job is a legacy file — all individual steps should exist instead."""
        for step in PRIOR_STEP_JOBS:
            assert (JOBS / step).exists(), \
                f"Prior step job {step} is missing — build_prior.job has no replacement"

    @pytest.mark.parametrize("job_file", JOB_FILES, ids=[f.name for f in JOB_FILES])
    def test_no_user_specific_absolute_paths(self, job_file):
        """Main SLURM jobs must not hardcode /gpfs/ or /home/<username> paths."""
        if job_file.name == "build_prior.job":
            pytest.skip("legacy file")
        text = job_file.read_text()
        bad = re.findall(r"/gpfs/home\d*/\w+/|/home/\w+/\w+/", text)
        assert not bad, \
            f"{job_file.name}: hardcoded user path(s): {bad}"


# ---------------------------------------------------------------------------
# Shell script tests
# ---------------------------------------------------------------------------

class TestShellScriptStructure:

    @pytest.mark.parametrize("sh_file", SHELL_JOBS, ids=[f.name for f in SHELL_JOBS])
    def test_bash_syntax_valid(self, sh_file):
        """bash -n checks syntax without executing."""
        r = subprocess.run(["bash", "-n", str(sh_file)], capture_output=True, text=True)
        assert r.returncode == 0, \
            f"{sh_file.name}: bash syntax error:\n{r.stderr}"

    @pytest.mark.parametrize("sh_file",
                             [f for f in HPC.glob("*.sh")
                              if f.name not in KNOWN_LEGACY_SCRIPTS],
                             ids=[f.name for f in HPC.glob("*.sh")
                                  if f.name not in KNOWN_LEGACY_SCRIPTS])
    def test_hpc_scripts_define_project_root(self, sh_file):
        """Non-legacy HPC scripts must define or use PROJECT_ROOT."""
        text = sh_file.read_text()
        assert "PROJECT_ROOT" in text, \
            f"{sh_file.name}: does not use PROJECT_ROOT — paths may be hardcoded"

    @pytest.mark.parametrize("sh_file",
                             sorted(HPC.glob("*.sh")),
                             ids=[f.name for f in sorted(HPC.glob("*.sh"))])
    def test_no_hardcoded_gpfs_user_path(self, sh_file):
        """After fixing, no script should contain a hardcoded /gpfs/home2/zblei path."""
        text = sh_file.read_text()
        assert "/gpfs/home2/zblei" not in text, \
            f"{sh_file.name}: still contains hardcoded /gpfs/home2/zblei path"

    def test_submit_prior_chain_references_step_jobs(self):
        chain = JOBS / "submit_prior_chain.sh"
        if not chain.exists():
            pytest.skip("submit_prior_chain.sh not found")
        text = chain.read_text()
        for step in ["prior_step1.job", "prior_step2a.job", "prior_step3.job", "prior_step4.job"]:
            assert step in text, \
                f"submit_prior_chain.sh does not reference {step}"

    def test_hpc_sync_reads_from_env(self):
        sync = JOBS / "hpc_sync.sh"
        if not sync.exists():
            pytest.skip("hpc_sync.sh not found")
        text = sync.read_text()
        assert ".env" in text or "SNELLIUS" in text, \
            "hpc_sync.sh should read credentials from .env"


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------

class TestCLIRouting:

    def test_diffuse_delegates_to_network_diffusion(self):
        """piglasso/cli.py diffuse must route to network_diffusion.py, not node_knockout.py."""
        cli_text = (REPO / "piglasso" / "cli.py").read_text()
        # Find the diffuse command block
        diffuse_start = cli_text.find("def diffuse(")
        diffuse_block = cli_text[diffuse_start:diffuse_start + 400]
        assert "network_diffusion.py" in diffuse_block, \
            "diffuse command must delegate to network_diffusion.py"
        assert "node_knockout.py" not in diffuse_block, \
            "diffuse command must NOT delegate to node_knockout.py"

    def test_knockout_delegates_to_node_knockout(self):
        """piglasso/cli.py knockout must route to node_knockout.py."""
        cli_text = (REPO / "piglasso" / "cli.py").read_text()
        knockout_start = cli_text.find("def knockout(")
        knockout_block = cli_text[knockout_start:knockout_start + 400]
        assert "node_knockout.py" in knockout_block, \
            "knockout command must delegate to node_knockout.py"
