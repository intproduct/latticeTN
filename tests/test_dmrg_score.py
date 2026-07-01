"""Lightweight test for the Stage 4A DMRG score script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_dmrg_score_list_command():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/dmrg_score.py", "--list"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0
    for name in [
        "tests/test_dmrg_environments.py",
        "tests/test_dmrg_effective_hamiltonian.py",
        "tests/test_dmrg_two_site_update.py",
        "tests/test_dmrg_sweep_smoke.py",
    ]:
        assert name in proc.stdout
    assert "scripts/run_dmrg_small.py" in proc.stdout
