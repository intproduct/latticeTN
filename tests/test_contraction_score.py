"""Lightweight test for the Stage 3B contraction score script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_contraction_score_list_command():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/contraction_score.py", "--list"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0
    for name in [
        "tests/test_native_norm_contraction.py",
        "tests/test_native_observable_contractions.py",
        "tests/test_native_mpo_energy_contraction.py",
        "tests/test_contraction_scalability_smoke.py",
    ]:
        assert name in proc.stdout
    assert "scripts/run_contraction_smoke.py" in proc.stdout
