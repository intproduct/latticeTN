#!/usr/bin/env python3
"""Stage 5B two-site AD local-tensor optimization score script for latticeTN.

Runs the Stage 5B two-site AD tests and generates ``docs/AD_TWO_SITE_REPORT.md``.
Other stages' paths are NOT run here (separate score scripts).

Usage:

    python scripts/ad_two_site_score.py --list
    python scripts/ad_two_site_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TWO_SITE_TESTS = [
    "tests/test_ad_two_site_loss.py",
    "tests/test_ad_two_site_gradients.py",
    "tests/test_ad_two_site_split.py",
    "tests/test_ad_two_site_sweep_smoke.py",
    "tests/test_ad_two_site_vs_one_site.py",
    "tests/test_ad_two_site_policy.py",
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
                        help="Run Stage 5B two-site AD tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 5B two-site AD files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 5B two-site AD tests:")
        for path in TWO_SITE_TESTS:
            print(f"  - {path}")
        print("Module:")
        print("  - latticetn/ad_two_site.py")
        print("Required scripts:")
        print("  - scripts/run_ad_two_site.py")
        print("Required docs:")
        print("  - docs/AD_TWO_SITE_SPEC.md")
        print("  - docs/AD_TWO_SITE_PROTOCOL.md")
        print("  - docs/AD_TWO_SITE_REPORT.md")
        print("  - docs/CLAUDE_PROGRESS_AD_TWO_SITE.md")
        return 0

    miss = missing(TWO_SITE_TESTS)
    if miss:
        print("Missing required Stage 5B two-site AD tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/AD_TWO_SITE_PROTOCOL.md.")
        return 2

    if not (ROOT / "latticetn" / "ad_two_site.py").exists():
        print("Missing latticetn/ad_two_site.py")
        return 2
    if not (ROOT / "scripts" / "run_ad_two_site.py").exists():
        print("Missing scripts/run_ad_two_site.py")
        return 2

    code = run([sys.executable, "-m", "pytest", "-q", *TWO_SITE_TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "AD_TWO_SITE_REPORT.md"
    code = run([sys.executable, "scripts/run_ad_two_site.py",
                "--markdown-output", str(report_md)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/AD_TWO_SITE_REPORT.md after run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required = [
        "mainline statement",
        "stabilization policy",
        "not the solver",
        "exact comparison",
        "one-site ad comparison",
        "global ad-mps comparison",
        "dmrg reference comparison",
        "energy history",
        "gradient check",
        "bond growth / compression",
        "truncation errors",
        "overall pass/fail",
        "known limitations",
    ]
    missing_terms = [t for t in required if t not in text]
    if missing_terms:
        print("AD_TWO_SITE_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nAD two-site score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
