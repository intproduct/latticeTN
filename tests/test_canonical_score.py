"""Lightweight tests for the Stage 3A canonical score script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_canonical_score_list_command():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/canonical_score.py", "--list"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "tests/test_mps_canonicalization.py" in proc.stdout
    assert "tests/test_mps_compression.py" in proc.stdout
    assert "tests/test_canonical_entanglement.py" in proc.stdout
    assert "scripts/run_canonical_smoke.py" in proc.stdout
