"""Lightweight tests for the benchmark score script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_benchmark_score_list_command():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/benchmark_score.py", "--list"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "tests/test_observables_dense_compare.py" in proc.stdout
    assert "scripts/run_heisenberg_benchmark.py" in proc.stdout
