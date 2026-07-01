#!/usr/bin/env python3
"""Stage 5A AD local-tensor optimization score script for latticeTN.

Runs the Stage 5A AD-local tests and generates ``docs/AD_LOCAL_OPT_REPORT.md``.
Stage 1/2/3A/3B/4A/4B/4R/5A-gauge paths are NOT run here (separate score
scripts).

Usage:

    python scripts/ad_local_opt_score.py --list
    python scripts/ad_local_opt_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOCAL_TESTS = [
    "tests/test_ad_local_opt_loss.py",
    "tests/test_ad_local_opt_gradients.py",
    "tests/test_ad_local_opt_step.py",
    "tests/test_ad_local_opt_vs_global_ad.py",
    "tests/test_ad_local_opt_policy.py",
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
                        help="Run Stage 5A AD-local tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 5A AD-local files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 5A AD-local tests:")
        for path in LOCAL_TESTS:
            print(f"  - {path}")
        print("Module:")
        print("  - latticetn/ad_local.py")
        print("Required scripts:")
        print("  - scripts/run_ad_local_opt.py")
        print("Required docs:")
        print("  - docs/AD_LOCAL_OPT_SPEC.md")
        print("  - docs/AD_LOCAL_OPT_PROTOCOL.md")
        print("  - docs/AD_LOCAL_OPT_REPORT.md")
        print("  - docs/CLAUDE_PROGRESS_AD_LOCAL_OPT.md")
        return 0

    miss = missing(LOCAL_TESTS)
    if miss:
        print("Missing required Stage 5A AD-local tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/AD_LOCAL_OPT_PROTOCOL.md.")
        return 2

    if not (ROOT / "latticetn" / "ad_local.py").exists():
        print("Missing latticetn/ad_local.py")
        return 2
    if not (ROOT / "scripts" / "run_ad_local_opt.py").exists():
        print("Missing scripts/run_ad_local_opt.py")
        return 2

    code = run([sys.executable, "-m", "pytest", "-q", *LOCAL_TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "AD_LOCAL_OPT_REPORT.md"
    code = run([sys.executable, "scripts/run_ad_local_opt.py",
                "--markdown-output", str(report_md)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/AD_LOCAL_OPT_REPORT.md after run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required = [
        "mainline statement",
        "stabilization policy",
        "exact comparison",
        "global ad-mps comparison",
        "dmrg reference comparison",
        "energy history",
        "gradient check",
        "overall pass/fail",
        "known limitations",
        "not the solver",
    ]
    missing_terms = [t for t in required if t not in text]
    if missing_terms:
        print("AD_LOCAL_OPT_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nAD local opt score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
