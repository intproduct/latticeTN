#!/usr/bin/env python3
"""Stage 3A canonicalization score script for latticeTN.

Runs the Stage 3A canonicalization/compression tests and generates
``docs/CANONICALIZATION_REPORT.md``. Stage 1 / Stage 2 / GPU-readiness paths
are NOT run here (they have their own score scripts and must remain green
separately per the stop conditions).

Usage:

    python scripts/canonical_score.py --list
    python scripts/canonical_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_TESTS = [
    "tests/test_mps_canonicalization.py",
    "tests/test_mps_compression.py",
    "tests/test_canonical_entanglement.py",
    "tests/test_canonical_score.py",
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # canonical tests are CPU-only
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
                        help="Run Stage 3A canonical tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 3A files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 3A tests:")
        for path in CANONICAL_TESTS:
            print(f"  - {path}")
        print("Required scripts:")
        print("  - scripts/run_canonical_smoke.py")
        print("Required report:")
        print("  - docs/CANONICALIZATION_REPORT.md")
        return 0

    miss = missing(CANONICAL_TESTS)
    if miss:
        print("Missing required Stage 3A tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/CANONICALIZATION_PROTOCOL.md.")
        return 2

    if not (ROOT / "scripts" / "run_canonical_smoke.py").exists():
        print("Missing scripts/run_canonical_smoke.py")
        return 2

    code = run([sys.executable, "-m", "pytest", "-q", *CANONICAL_TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "CANONICALIZATION_REPORT.md"
    code = run([sys.executable, "scripts/run_canonical_smoke.py",
                "--markdown-output", str(report_md)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/CANONICALIZATION_REPORT.md after smoke run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace")
    text_lower = text.lower()
    required_terms = [
        "left", "right", "mixed canonical", "entropy", "compression",
        "energy", "fidelity", "pass", "known limitations",
    ]
    missing_terms = [t for t in required_terms if t not in text_lower]
    if missing_terms:
        print("CANONICALIZATION_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nCanonical score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
