#!/usr/bin/env python3
"""Stage 2.5 GPU readiness score script for latticeTN (OPT-IN).

Behavior (see docs/GPU_TESTING_PROTOCOL.md):

- Without ``LATTICETN_RUN_GPU=1``:
      print an explanatory message and ``exit 0`` (GPU tests are opt-in; the
      default validation/benchmark paths are not affected).
- With ``LATTICETN_RUN_GPU=1`` but no CUDA:
      cleanly skip; write a Skip note to ``docs/GPU_REPORT.md``; ``exit 0``.
- With ``LATTICETN_RUN_GPU=1`` and CUDA, but no GPU whose name contains
      ``LATTICETN_GPU_NAME_FILTER`` (default ``"Pro 4000 Blackwell"``):
      cleanly skip; DO NOT fall back to any other GPU; write the no-match note to
      ``docs/GPU_REPORT.md``; ``exit 0``.
- With ``LATTICETN_RUN_GPU=1`` and a matching GPU:
      run the GPU correctness smoke on that GPU ONLY, write/update
      ``docs/GPU_REPORT.md``, print the used GPU index/name, and exit 0 on pass
      / 1 on fail.

Recommended invocation:

    LATTICETN_RUN_GPU=1 LATTICETN_GPU_NAME_FILTER="Pro 4000 Blackwell" \\
        python scripts/gpu_score.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gpu_smoke import env_run_gpu, render_report_md, run_smoke  # noqa: E402

REPORT_PATH = ROOT / "docs" / "GPU_REPORT.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run the GPU correctness smoke checks (opt-in).")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if not args.smoke:
        print("gpu_score: nothing to do (use --smoke to run GPU correctness checks).")
        return 0

    if not env_run_gpu():
        print("gpu_score: LATTICETN_RUN_GPU != 1; GPU smoke is opt-in and was "
              "not requested. Writing a Skip note to docs/GPU_REPORT.md and exiting 0.")
        report = run_smoke(N=args.N, chi=args.chi, steps=args.steps,
                           lr=args.lr, seed=args.seed)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report_md(report), encoding="utf-8")
        return 0

    # LATTICETN_RUN_GPU=1: run_smoke handles CUDA-availability and name-match
    # gating internally, producing a clean-skip report when appropriate.
    report = run_smoke(N=args.N, chi=args.chi, steps=args.steps,
                       lr=args.lr, seed=args.seed)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report_md(report), encoding="utf-8")

    if report.get("skip_reason"):
        print(f"gpu_score: SKIPPED - {report['skip_reason']}")
        print(f"  report written to: {args.report}")
        # Clean skip is a successful exit (per stop conditions 4 and 5).
        return 0

    used_idx = report.get("used_gpu_index")
    used_name = report.get("used_gpu_name")
    print(f"gpu_score: used GPU index={used_idx} name={used_name!r}")
    print(f"  report written to: {args.report}")

    if report.get("pass"):
        print("gpu_score: PASS")
        return 0
    print("gpu_score: FAIL")
    cp = report.get("checks_pass") or {}
    for k, v in cp.items():
        if not v:
            print(f"  failed check: {k}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
