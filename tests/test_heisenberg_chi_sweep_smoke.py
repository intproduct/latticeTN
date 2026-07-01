"""Smoke test for the Stage 2 Heisenberg benchmark runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_heisenberg_benchmark_tiny_preset_runs(tmp_path: Path):
    json_path = tmp_path / "benchmark.json"
    md_path = tmp_path / "benchmark.md"
    cmd = [
        sys.executable,
        "scripts/run_heisenberg_benchmark.py",
        "--preset", "tiny",
        "--json-output", str(json_path),
        "--markdown-output", str(md_path),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    data = json.loads(json_path.read_text())
    assert len(data) >= 1
    for row in data:
        assert row["N"] >= 4
        assert row["chi"] >= 2
        assert row["steps"] > 0
        assert row["final_E"] >= row["exact_E0"] - 1e-6
        assert row["energy_per_bond"] < 0.0
        assert row["pass"] is True
    text = md_path.read_text()
    assert "energy per bond" in text
    assert "1/4 - ln(2)" in text
