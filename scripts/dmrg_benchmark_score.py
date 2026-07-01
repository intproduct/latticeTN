#!/usr/bin/env python3
"""Stage 4B scalable DMRG benchmark score script for latticeTN.

Runs the Stage 4B tests and generates ``docs/DMRG_BENCHMARK_REPORT.md``.
Stage 1/2/3A/3B/4A/GPU paths are NOT run here (separate score scripts).

Usage:

    python scripts/dmrg_benchmark_score.py --list
    python scripts/dmrg_benchmark_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BENCH_TESTS = [
    "tests/test_dmrg_matrix_free_heff.py",
    "tests/test_dmrg_lanczos_solver.py",
    "tests/test_dmrg_benchmark_smoke.py",
    "tests/test_dmrg_benchmark_score.py",
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
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
                        help="Run Stage 4B benchmark tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 4B files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 4B tests:")
        for path in BENCH_TESTS:
            print(f"  - {path}")
        print("Required scripts:")
        print("  - scripts/run_dmrg_benchmark.py")
        print("Required report:")
        print("  - docs/DMRG_BENCHMARK_REPORT.md")
        return 0

    miss = missing(BENCH_TESTS)
    if miss:
        print("Missing required Stage 4B tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/DMRG_BENCHMARK_PROTOCOL.md.")
        return 2

    if not (ROOT / "scripts" / "run_dmrg_benchmark.py").exists():
        print("Missing scripts/run_dmrg_benchmark.py")
        return 2

    code = run([sys.executable, "-m", "pytest", "-q", *BENCH_TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "DMRG_BENCHMARK_REPORT.md"
    code = run([sys.executable, "scripts/run_dmrg_benchmark.py",
                "--markdown-output", str(report_md)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/DMRG_BENCHMARK_REPORT.md after benchmark run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required = [
        "matrix-free vs dense", "lanczos vs dense", "exact comparison",
        "smoke", "chi sweep", "runtime", "pass", "known limitations",
    ]
    missing_terms = [t for t in required if t not in text]
    if missing_terms:
        print("DMRG_BENCHMARK_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nDMRG benchmark score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
