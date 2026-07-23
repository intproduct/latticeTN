#!/usr/bin/env python3
"""
Score script for the latticeTN autonomous validation loop.

This script intentionally starts simple. The coding agent should update the TEST_GROUPS below
as it creates the corresponding tests. The script returns non-zero until the
required validation tests pass.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAST_TESTS = [
    "tests/test_reference_models.py",
    "tests/test_mps_dense.py",
    "tests/test_mpo_dense.py",
    "tests/test_tfi_mpo_dense.py",
    "tests/test_energy_rayleigh.py",
    "tests/test_heisenberg_mpo_dense.py",
    "tests/test_heisenberg_energy_dense_compare.py",
    "tests/test_heisenberg_variational_smoke.py",
    "tests/test_p0_scientific_compliance.py",
    "tests/test_tdvp_krylov.py",
    "tests/test_tdvp_effective_hamiltonian.py",
    "tests/test_tdvp_one_site.py",
    "tests/test_tdvp_two_site.py",
    "tests/physics/test_stage11_hamiltonian_audit.py",
    "tests/physics/test_stage11_small_n_energy_benchmarks.py",
    "tests/physics/test_stage11_observables_correlations.py",
    "tests/physics/test_stage11_benchmark_suite_runner.py",
]

FULL_EXTRA_COMMANDS = [
    [sys.executable, "scripts/run_heisenberg_small.py", "--N", "6", "--chi", "8", "--steps", "300", "--lr", "1e-2", "--seed", "0", "--device", "cpu"],
    [sys.executable, "scripts/run_tdvp_heisenberg_quench.py", "--N", "8", "--steps", "10", "--chi-max", "8", "--device", "cpu"],
]


def existing(paths: list[str]) -> list[str]:
    return [p for p in paths if (ROOT / p).exists()]


def missing(paths: list[str]) -> list[str]:
    return [p for p in paths if not (ROOT / p).exists()]


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run fast CPU validation tests.")
    parser.add_argument("--full", action="store_true", help="Run fast tests plus the small Heisenberg solve.")
    args = parser.parse_args()

    if not args.fast and not args.full:
        args.fast = True

    miss = missing(FAST_TESTS)
    if miss:
        print("Missing required validation tests:")
        for path in miss:
            print(f"  - {path}")
        print("\nThe coding agent should create the missing tests according to docs/VALIDATION_PROTOCOL.md.")
        return 2

    tests = existing(FAST_TESTS)
    code = run([sys.executable, "-m", "pytest", "-q", *tests])
    if code != 0:
        return code

    if args.full:
        for cmd in FULL_EXTRA_COMMANDS:
            if not (ROOT / cmd[1]).exists():
                print(f"Missing full validation script: {cmd[1]}")
                return 2
            code = run(cmd)
            if code != 0:
                return code

    report = ROOT / "docs" / "NUMERICAL_REPORT.md"
    if args.full and not report.exists():
        print("Missing docs/NUMERICAL_REPORT.md")
        return 2

    print("\nScore: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
