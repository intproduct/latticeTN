"""Stage 7B: model_builder_score --list exits 0 and lists required files.

A lightweight score-coverage test (mirrors the Stage 6A/7A report tests).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "scripts" / "model_builder_score.py"

REQUIRED_TESTS = [
    "tests/test_model_builder_heisenberg.py",
    "tests/test_model_builder_fermion.py",
    "tests/test_model_builder_mpo_dense.py",
    "tests/test_benchmark_registry.py",
    "tests/test_stage7b_score.py",
]
REQUIRED_MODULES = ["latticetn/model_builder.py", "latticetn/benchmarking.py"]
REQUIRED_SCRIPTS = ["scripts/model_builder_score.py"]
REQUIRED_DOCS = [
    "docs/MODEL_BUILDER_SPEC.md",
    "docs/MODEL_BUILDER_PROTOCOL.md",
    "docs/MODEL_BUILDER_REPORT.md",
    "docs/CLAUDE_PROGRESS_MODEL_BUILDER.md",
]


def test_model_builder_score_list_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(SCORE), "--list"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    # lists the required tests / modules / scripts / docs
    for t in REQUIRED_TESTS:
        assert t in out, t
    for m in REQUIRED_MODULES:
        assert m in out, m
    for s in REQUIRED_SCRIPTS:
        assert s in out, s
    for d in REQUIRED_DOCS:
        assert d in out, d


def test_required_stage7b_files_exist():
    for p in REQUIRED_TESTS + REQUIRED_MODULES + REQUIRED_SCRIPTS + REQUIRED_DOCS:
        assert (ROOT / p).exists(), p
