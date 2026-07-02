#!/usr/bin/env python3
"""Stage 6A CPU/GPU AD solver benchmark score script for latticeTN.

Runs the Stage 6A benchmark tests and generates ``docs/AD_GPU_BENCHMARK_REPORT.md``.
Other stages' paths are NOT run here (separate score scripts).

Default (no ``LATTICETN_RUN_GPU=1``) is CPU-only and always works. The GPU
benchmark is opt-in: set ``LATTICETN_RUN_GPU=1`` AND have CUDA available to
run the GPU portion; otherwise it clean-skips (still exit 0).

Usage:

    python scripts/ad_gpu_benchmark_score.py --list
    python scripts/ad_gpu_benchmark_score.py --fast
    LATTICETN_RUN_GPU=1 python scripts/ad_gpu_benchmark_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTS = [
    "tests/test_ad_gpu_benchmark_config.py",
    "tests/test_ad_gpu_benchmark_smoke.py",
    "tests/test_ad_gpu_benchmark_report.py",
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    # CPU-only by default; the runner itself gates the GPU on
    # LATTICETN_RUN_GPU. We do NOT force CUDA_VISIBLE_DEVICES="" here when the
    # caller opted into the GPU, so the GPU remains visible to the runner.
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
                        help="Run Stage 6A benchmark tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 6A benchmark files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 6A AD GPU benchmark tests:")
        for path in TESTS:
            print(f"  - {path}")
        print("Required scripts:")
        print("  - scripts/run_ad_gpu_benchmark.py")
        print("Required docs:")
        for d in ("docs/AD_GPU_BENCHMARK_SPEC.md",
                  "docs/AD_GPU_BENCHMARK_PROTOCOL.md",
                  "docs/AD_GPU_BENCHMARK_REPORT.md",
                  "docs/CLAUDE_PROGRESS_AD_GPU_BENCHMARK.md"):
            print(f"  - {d}")
        return 0

    miss = missing(TESTS)
    if miss:
        print("Missing required Stage 6A benchmark tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/AD_GPU_BENCHMARK_PROTOCOL.md.")
        return 2

    if not (ROOT / "scripts" / "run_ad_gpu_benchmark.py").exists():
        print("Missing scripts/run_ad_gpu_benchmark.py")
        return 2

    code = run([sys.executable, "-m", "pytest", "-q", *TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "AD_GPU_BENCHMARK_REPORT.md"
    json_out = ROOT / "docs" / "ad_gpu_benchmark_fast_results.json"
    code = run([sys.executable, "scripts/run_ad_gpu_benchmark.py",
                "--markdown-output", str(report_md),
                "--json-output", str(json_out)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/AD_GPU_BENCHMARK_REPORT.md after run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required = [
        "mainline statement",
        "device info",
        "reference baselines",
        "cpu/gpu comparison",
        "speedup",
        "below ground",
        "overall pass/fail",
        "known limitations",
        "ad mainline",            # mainline statement mentions the AD mainline
        "reference baselines only",  # DMRG/Lanczos flagged reference-only
    ]
    missing_terms = [t for t in required if t not in text]
    if missing_terms:
        print("AD_GPU_BENCHMARK_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nAD GPU benchmark score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
