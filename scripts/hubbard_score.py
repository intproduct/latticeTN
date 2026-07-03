#!/usr/bin/env python3
"""Stage 7C spinful Hubbard chain score script for latticeTN.

Runs the Stage 7C Hubbard tests and generates ``docs/HUBBARD_REPORT.md``.
Other stages' paths are NOT run here (separate score scripts).

Default (no ``LATTICETN_RUN_GPU=1``) is CPU-only and always works. The GPU
benchmark is opt-in: set ``LATTICETN_RUN_GPU=1`` AND have a V100/TITAN V
available (selected by the unified GPU selector) to run the GPU portion;
otherwise it clean-skips (still exit 0).

Usage:

    python scripts/hubbard_score.py --list
    python scripts/hubbard_score.py --fast
    LATTICETN_RUN_GPU=1 python scripts/hubbard_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTS = [
    "tests/test_hubbard_operators.py",
    "tests/test_hubbard_dense.py",
    "tests/test_hubbard_mpo_dense.py",
    "tests/test_hubbard_native_energy.py",
    "tests/test_hubbard_observables.py",
    "tests/test_model_builder_hubbard.py",
    "tests/test_hubbard_ad_solvers.py",
    "tests/test_hubbard_gpu_timing.py",
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    # CPU-only by default; the runner itself gates the GPU on LATTICETN_RUN_GPU
    # and the unified gpu_selector picks a V100/TITAN V. We do NOT force
    # CUDA_VISIBLE_DEVICES="" when the caller opted into the GPU.
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
                        help="Run Stage 7C Hubbard tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 7C Hubbard files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 7C Hubbard tests:")
        for path in TESTS:
            print(f"  - {path}")
        print("Modules:")
        print("  - latticetn/fermion_operators.py (hubbard_local_operators)")
        print("  - latticetn/operators.py (hubbard_dense)")
        print("  - latticetn/mpo.py (generate_hubbard)")
        print("  - latticetn/model_builder.py (hubbard_model)")
        print("  - latticetn/observables.py (Hubbard observables)")
        print("Required scripts:")
        print("  - scripts/run_hubbard_benchmark.py")
        print("  - scripts/gpu_selector.py")
        print("Required docs:")
        for d in ("docs/HUBBARD_SPEC.md", "docs/HUBBARD_PROTOCOL.md",
                  "docs/HUBBARD_REPORT.md", "docs/CLAUDE_PROGRESS_HUBBARD.md",
                  "docs/GPU_TESTING_PROTOCOL.md"):
            print(f"  - {d}")
        return 0

    miss = missing(TESTS)
    if miss:
        print("Missing required Stage 7C Hubbard tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/HUBBARD_PROTOCOL.md.")
        return 2

    for f in ("latticetn/fermion_operators.py", "latticetn/operators.py",
              "latticetn/mpo.py", "latticetn/model_builder.py",
              "scripts/gpu_selector.py", "scripts/run_hubbard_benchmark.py"):
        if not (ROOT / f).exists():
            print(f"Missing {f}")
            return 2

    # Generate the report FIRST (so the score-coverage test, if any, finds the
    # report), then run the test suite.
    report_md = ROOT / "docs" / "HUBBARD_REPORT.md"
    json_out = ROOT / "docs" / "hubbard_fast_results.json"
    code = run([sys.executable, "scripts/run_hubbard_benchmark.py",
                "--markdown-output", str(report_md),
                "--json-output", str(json_out)])
    if code != 0:
        return code

    code = run([sys.executable, "-m", "pytest", "-q", *TESTS])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/HUBBARD_REPORT.md after run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required = [
        "mainline statement",
        "device info",
        "reference baseline",
        "cpu/gpu comparison",
        "speedup",
        "below ground",
        "overall pass/fail",
        "known limitations",
        "jordan-wigner",
        "ad mainline",
        "site-major",
    ]
    missing_terms = [t for t in required if t not in text]
    if missing_terms:
        print("HUBBARD_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nHubbard score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
