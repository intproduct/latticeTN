#!/usr/bin/env python3
"""Stage 7B model builder score script for latticeTN.

Runs the Stage 7B model-builder/benchmark-registry tests and generates
``docs/MODEL_BUILDER_REPORT.md``. Other stages' paths are NOT run here.

Default (no ``LATTICETN_RUN_GPU=1``) is CPU-only and always works. The GPU
portion is opt-in: set ``LATTICETN_RUN_GPU=1`` AND have a V100/TITAN V
available (selected by the unified gpu_selector) to run the GPU portion;
otherwise it clean-skips (still exit 0).

Usage:

    python scripts/model_builder_score.py --list
    python scripts/model_builder_score.py --fast
    LATTICETN_RUN_GPU=1 python scripts/model_builder_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTS = [
    "tests/test_model_builder_heisenberg.py",
    "tests/test_model_builder_fermion.py",
    "tests/test_model_builder_mpo_dense.py",
    "tests/test_benchmark_registry.py",
    "tests/test_stage7b_score.py",
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    if os.environ.get("LATTICETN_RUN_GPU", "") != "1":
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=_env())
    return proc.returncode


def missing(paths: list[str]) -> list[str]:
    return [p for p in paths if not (ROOT / p).exists()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Run Stage 7B tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 7B files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 7B model-builder tests:")
        for path in TESTS:
            print(f"  - {path}")
        print("Modules:")
        print("  - latticetn/model_builder.py")
        print("  - latticetn/benchmarking.py")
        print("Required scripts:")
        print("  - scripts/model_builder_score.py")
        print("  - scripts/run_model_builder_benchmark.py")
        print("Required docs:")
        for d in ("docs/MODEL_BUILDER_SPEC.md",
                  "docs/MODEL_BUILDER_PROTOCOL.md",
                  "docs/MODEL_BUILDER_REPORT.md",
                  "docs/CLAUDE_PROGRESS_MODEL_BUILDER.md"):
            print(f"  - {d}")
        return 0

    miss = missing(TESTS)
    if miss:
        print("Missing required Stage 7B tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/MODEL_BUILDER_PROTOCOL.md.")
        return 2

    for f in ("latticetn/model_builder.py", "latticetn/benchmarking.py",
              "scripts/model_builder_score.py",
              "scripts/run_model_builder_benchmark.py"):
        if not (ROOT / f).exists():
            print(f"Missing {f}")
            return 2

    # Generate the report FIRST so the score-coverage test
    # (test_required_stage7b_files_exist) finds docs/MODEL_BUILDER_REPORT.md.
    report_md = ROOT / "docs" / "MODEL_BUILDER_REPORT.md"
    json_out = ROOT / "docs" / "model_builder_fast_results.json"
    code = run([sys.executable, "scripts/run_model_builder_benchmark.py",
                "--markdown-output", str(report_md),
                "--json-output", str(json_out)])
    if code != 0:
        return code

    code = run([sys.executable, "-m", "pytest", "-q", *TESTS])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/MODEL_BUILDER_REPORT.md after run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required = [
        "mainline statement",
        "device info",
        "cpu/gpu comparison",
        "speedup",
        "below ground",
        "overall pass/fail",
        "known limitations",
        "model/mpo construction layer",
        "ad mainline",
        "reference baselines",
    ]
    missing_terms = [t for t in required if t not in text]
    if missing_terms:
        print("MODEL_BUILDER_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nModel builder score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
