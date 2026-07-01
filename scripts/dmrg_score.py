#!/usr/bin/env python3
"""Stage 4A two-site DMRG score script for latticeTN.

Runs the Stage 4A DMRG tests and generates ``docs/DMRG_REPORT.md``.
Stage 1/2/3A/3B/GPU paths are NOT run here (separate score scripts).

Usage:

    python scripts/dmrg_score.py --list
    python scripts/dmrg_score.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DMRG_TESTS = [
    "tests/test_dmrg_environments.py",
    "tests/test_dmrg_effective_hamiltonian.py",
    "tests/test_dmrg_two_site_update.py",
    "tests/test_dmrg_sweep_smoke.py",
    "tests/test_dmrg_score.py",
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # DMRG tests are CPU-only
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
                        help="Run Stage 4A DMRG tests and generate the report.")
    parser.add_argument("--list", action="store_true",
                        help="List required Stage 4A files and exit.")
    args = parser.parse_args()

    if not args.fast and not args.list:
        args.fast = True

    if args.list:
        print("Required Stage 4A tests:")
        for path in DMRG_TESTS:
            print(f"  - {path}")
        print("Required scripts:")
        print("  - scripts/run_dmrg_small.py")
        print("Required report:")
        print("  - docs/DMRG_REPORT.md")
        return 0

    miss = missing(DMRG_TESTS)
    if miss:
        print("Missing required Stage 4A tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nCreate these tests according to docs/DMRG_PROTOCOL.md.")
        return 2

    if not (ROOT / "scripts" / "run_dmrg_small.py").exists():
        print("Missing scripts/run_dmrg_small.py")
        return 2

    code = run([sys.executable, "-m", "pytest", "-q", *DMRG_TESTS])
    if code != 0:
        return code

    report_md = ROOT / "docs" / "DMRG_REPORT.md"
    code = run([sys.executable, "scripts/run_dmrg_small.py",
                "--markdown-output", str(report_md)])
    if code != 0:
        return code

    if not report_md.exists():
        print("Missing docs/DMRG_REPORT.md after smoke run")
        return 2

    text = report_md.read_text(encoding="utf-8", errors="replace").lower()
    required_terms = [
        "exact", "dmrg", "energy history", "trunc", "bond", "h_eff herm",
        "pass", "known limitations",
    ]
    missing_terms = [t for t in required_terms if t not in text]
    if missing_terms:
        print("DMRG_REPORT.md is missing required terms:")
        for t in missing_terms:
            print(f"  - {t}")
        return 2

    print("\nDMRG score: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
