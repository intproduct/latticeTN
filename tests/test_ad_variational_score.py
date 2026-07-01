"""Lightweight test for the Stage 4R AD variational score script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ad_variational_score_list_command():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/ad_variational_score.py", "--list"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0
    for name in [
        "tests/test_ad_variational_loss.py",
        "tests/test_ad_variational_gradients.py",
        "tests/test_ad_mps_optimizer_smoke.py",
        "tests/test_ad_vs_dmrg_reference.py",
    ]:
        assert name in proc.stdout
    assert "scripts/run_ad_mps_heisenberg.py" in proc.stdout
