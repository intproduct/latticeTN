#!/usr/bin/env python3
"""Stage 3B native contraction score script for latticeTN.

Runs the Stage 3B native-contraction tests and generates
``docs/CONTRACTION_REPORT.md``. Stage 1/2/3A/GPU-readiness paths are NOT run
here (they have their own score scripts and must remain green separately).

Usage:

    python scripts/contraction_score.py --list
    python scripts/contraction_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTRACTION_TESTS = [
    "tests/test_native_norm_contraction.py",
    "tests/test_native_observable_contractions.py",
    "tests/test_native_mpo_energy_contraction.py",
    "tests/test_contraction_scalability_smoke.py",
    "tests/test_contraction_score.py",
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # contraction tests are CPU-only
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
                        help="Run Stage 3B contraction tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 3B files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 3B tests:")
        for path in CONTRACTION_TESTS:
            print(f"  - {path}")
        print("Required scripts:")
        print("  - scripts/run_contraction_smoke.py")
        print("Required report:")
        print("  - docs/CONTRACTION_REPORT.md")
        return 0

    miss = missing(CONTRACTION_TESTS)
    if miss:
        print("Missing required Stage 3B tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/CONTRACTION_PROTOCOL.md.")
        return 2

    if not (ROOT / "scripts" / "run_contraction_smoke.py").exists():
        print("Missing scripts/run_contraction_smoke.py")
        return 2

    code = run([sys.executable, "-m", "pytest", "-q", *CONTRACTION_TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "CONTRACTION_REPORT.md"
    code = run([sys.executable, "scripts/run_contraction_smoke.py",
                "--markdown-output", str(report_md)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/CONTRACTION_REPORT.md after smoke run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required_terms = [
        "native", "dense reference", "mpo energy", "gradient",
        "scalability", "pass", "known limitations",
    ]
    missing_terms = [t for t in required_terms if t not in text]
    if missing_terms:
        print("CONTRACTION_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nContraction score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
