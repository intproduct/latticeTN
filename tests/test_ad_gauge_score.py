"""Lightweight test for the Stage 5A AD gauge score script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ad_gauge_score_list_command():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/ad_gauge_score.py", "--list"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0
    for name in [
        "tests/test_ad_gauge_projection.py",
        "tests/test_ad_gauge_loss_integrity.py",
        "tests/test_ad_gauge_optimizer_smoke.py",
        "tests/test_ad_gauge_vs_baseline.py",
    ]:
        assert name in proc.stdout
    assert "scripts/run_ad_gauge_heisenberg.py" in proc.stdout
