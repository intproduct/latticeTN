"""Lightweight test for the Stage 6A AD GPU benchmark score script.

Mirrors tests/test_dmrg_benchmark_score.py: runs
`scripts/ad_gpu_benchmark_score.py --list` and asserts the required tests /
runner / docs are listed. CPU-only, fast.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ad_gpu_benchmark_score_list_command():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/ad_gpu_benchmark_score.py", "--list"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0
    for name in [
        "tests/test_ad_gpu_benchmark_config.py",
        "tests/test_ad_gpu_benchmark_smoke.py",
        "tests/test_ad_gpu_benchmark_report.py",
    ]:
        assert name in proc.stdout
    assert "scripts/run_ad_gpu_benchmark.py" in proc.stdout
    for doc in [
        "docs/AD_GPU_BENCHMARK_SPEC.md",
        "docs/AD_GPU_BENCHMARK_PROTOCOL.md",
        "docs/AD_GPU_BENCHMARK_REPORT.md",
        "docs/CLAUDE_PROGRESS_AD_GPU_BENCHMARK.md",
    ]:
        assert doc in proc.stdout
