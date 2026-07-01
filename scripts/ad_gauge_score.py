#!/usr/bin/env python3
"""Stage 5A gauge-stabilized AD-MPS score script for latticeTN.

Runs the Stage 5A tests and generates ``docs/AD_GAUGE_REPORT.md``.
Stage 1/2/3A/3B/4A/4B/4R/GPU paths are NOT run here (separate score scripts).

Usage:

    python scripts/ad_gauge_score.py --list
    python scripts/ad_gauge_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GAUGE_TESTS = [
    "tests/test_ad_gauge_projection.py",
    "tests/test_ad_gauge_loss_integrity.py",
    "tests/test_ad_gauge_optimizer_smoke.py",
    "tests/test_ad_gauge_vs_baseline.py",
    "tests/test_ad_gauge_score.py",
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
                        help="Run Stage 5A gauge tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 5A files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 5A tests:")
        for path in GAUGE_TESTS:
            print(f"  - {path}")
        print("Required scripts:")
        print("  - scripts/run_ad_gauge_heisenberg.py")
        print("Required report:")
        print("  - docs/AD_GAUGE_REPORT.md")
        return 0

    miss = missing(GAUGE_TESTS)
    if miss:
        print("Missing required Stage 5A tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/AD_GAUGE_PROTOCOL.md.")
        return 2

    if not (ROOT / "scripts" / "run_ad_gauge_heisenberg.py").exists():
        print("Missing scripts/run_ad_gauge_heisenberg.py")
        return 2

    code = run([sys.executable, "-m", "pytest", "-q", *GAUGE_TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "AD_GAUGE_REPORT.md"
    code = run([sys.executable, "scripts/run_ad_gauge_heisenberg.py",
                "--markdown-output", str(report_md)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/AD_GAUGE_REPORT.md after run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required = [
        "projection comparison", "projection statement", "energy history",
        "grad norm", "state norm", "canonical error", "dmg reference" if False else "dmrg reference",
        "pass", "known limitations",
    ]
    missing_terms = [t for t in required if t not in text]
    if missing_terms:
        print("AD_GAUGE_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nAD gauge score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
