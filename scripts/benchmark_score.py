#!/usr/bin/env python3
"""Stage 2 benchmark score script for latticeTN.

This script is intentionally strict enough to guide an autonomous coding agent:
- Stage 1 fast validation must remain green.
- Stage 2 benchmark/observable tests must exist and pass.
- A tiny benchmark smoke run must succeed.
- A benchmark report must exist.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGE2_TESTS = [
    "tests/test_observables_dense_compare.py",
    "tests/test_entanglement_entropy.py",
    "tests/test_heisenberg_chi_sweep_smoke.py",
    "tests/test_benchmark_score.py",
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
    parser.add_argument("--fast", action="store_true", help="Run fast Stage 2 validation.")
    parser.add_argument("--full", action="store_true", help="Run fast validation plus fuller benchmark.")
    parser.add_argument("--skip-stage1", action="store_true", help="Do not run Stage 1 validation first.")
    parser.add_argument("--list", action="store_true", help="List required Stage 2 files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.full and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 2 tests:")
        for path in STAGE2_TESTS:
            print(f"  - {path}")
        print("Required scripts:")
        print("  - scripts/run_heisenberg_benchmark.py")
        print("Required report:")
        print("  - docs/BENCHMARK_REPORT.md")
        return 0

    miss = missing(STAGE2_TESTS)
    if miss:
        print("Missing required Stage 2 tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/BENCHMARK_PROTOCOL.md.")
        return 2

    if not (ROOT / "scripts" / "run_heisenberg_benchmark.py").exists():
        print("Missing scripts/run_heisenberg_benchmark.py")
        return 2

    if not args.skip_stage1:
        code = run([sys.executable, "scripts/validation_score.py", "--fast"])
        if code != 0:
            print("Stage 1 validation failed; fix regression before Stage 2.")
            return code

    code = run([sys.executable, "-m", "pytest", "-q", *STAGE2_TESTS])
    if code != 0:
        return code

    bench_json = ROOT / "docs" / "benchmark_fast_results.json"
    bench_md = ROOT / "docs" / "BENCHMARK_REPORT.md"
    code = run([
        sys.executable,
        "scripts/run_heisenberg_benchmark.py",
        "--preset", "tiny",
        "--json-output", str(bench_json),
        "--markdown-output", str(bench_md),
    ])
    if code != 0:
        return code

    if not bench_md.exists():
        print("Missing docs/BENCHMARK_REPORT.md after benchmark run")
        return 2

    text = bench_md.read_text(encoding="utf-8", errors="replace")
    required_terms = ["N", "chi", "exact", "final", "energy per bond", "1/4 - ln(2)"]
    missing_terms = [term for term in required_terms if term not in text]
    if missing_terms:
        print("BENCHMARK_REPORT.md is missing required terms:")
        for term in missing_terms:
            print(f"  - {term}")
        return 2

    if args.full:
        code = run([
            sys.executable,
            "scripts/run_heisenberg_benchmark.py",
            "--preset", "fast",
            "--json-output", str(ROOT / "docs" / "benchmark_fullish_results.json"),
            "--markdown-output", str(ROOT / "docs" / "BENCHMARK_REPORT.md"),
        ])
        if code != 0:
            return code

    print("\nBenchmark score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
