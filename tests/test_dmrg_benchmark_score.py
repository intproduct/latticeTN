"""Lightweight test for the Stage 4B DMRG benchmark score script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_dmrg_benchmark_score_list_command():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/dmrg_benchmark_score.py", "--list"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0
    for name in [
        "tests/test_dmrg_matrix_free_heff.py",
        "tests/test_dmrg_lanczos_solver.py",
        "tests/test_dmrg_benchmark_smoke.py",
    ]:
        assert name in proc.stdout
    assert "scripts/run_dmrg_benchmark.py" in proc.stdout
