#!/usr/bin/env python3
"""Stage 7A spinless fermion t-V chain score script for latticeTN.

Runs the Stage 7A fermion tests and generates ``docs/FERMION_REPORT.md``.
Other stages' paths are NOT run here (separate score scripts).

Default (no ``LATTICETN_RUN_GPU=1``) is CPU-only and always works. The GPU
benchmark is opt-in: set ``LATTICETN_RUN_GPU=1`` AND have a V100/TITAN V
available (selected by the unified GPU selector) to run the GPU portion;
otherwise it clean-skips (still exit 0).

Usage:

    python scripts/fermion_score.py --list
    python scripts/fermion_score.py --fast
    LATTICETN_RUN_GPU=1 python scripts/fermion_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTS = [
    "tests/test_fermion_operators.py",
    "tests/test_spinless_fermion_dense.py",
    "tests/test_spinless_fermion_mpo_dense.py",
    "tests/test_spinless_fermion_native_energy.py",
    "tests/test_spinless_fermion_ad_solvers.py",
    "tests/test_fermion_gpu_timing.py",
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
                        help="Run Stage 7A fermion tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 7A fermion files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 7A fermion tests:")
        for path in TESTS:
            print(f"  - {path}")
        print("Modules:")
        print("  - latticetn/fermion_operators.py")
        print("  - latticetn/operators.py (spinless_fermion_dense)")
        print("  - latticetn/mpo.py (generate_spinless_fermion)")
        print("  - latticetn/observables.py (fermion observables)")
        print("Required scripts:")
        print("  - scripts/run_spinless_fermion_benchmark.py")
        print("  - scripts/gpu_selector.py")
        print("Required docs:")
        for d in ("docs/FERMION_SPEC.md", "docs/FERMION_PROTOCOL.md",
                  "docs/FERMION_REPORT.md", "docs/CLAUDE_PROGRESS_FERMION.md",
                  "docs/GPU_TESTING_PROTOCOL.md"):
            print(f"  - {d}")
        return 0

    miss = missing(TESTS)
    if miss:
        print("Missing required Stage 7A fermion tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/FERMION_PROTOCOL.md.")
        return 2

    for f in ("latticetn/fermion_operators.py", "scripts/gpu_selector.py",
              "scripts/run_spinless_fermion_benchmark.py"):
        if not (ROOT / f).exists():
            print(f"Missing {f}")
            return 2

    code = run([sys.executable, "-m", "pytest", "-q", *TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "FERMION_REPORT.md"
    json_out = ROOT / "docs" / "fermion_fast_results.json"
    code = run([sys.executable, "scripts/run_spinless_fermion_benchmark.py",
                "--markdown-output", str(report_md),
                "--json-output", str(json_out)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/FERMION_REPORT.md after run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required = [
        "mainline statement",
        "device info",
        "reference baselines only",
        "cpu/gpu comparison",
        "speedup",
        "below ground",
        "overall pass/fail",
        "known limitations",
        "jordan-wigner",
        "ad mainline",
    ]
    missing_terms = [t for t in required if t not in text]
    if missing_terms:
        print("FERMION_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nFermion score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
