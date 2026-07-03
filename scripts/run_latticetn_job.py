#!/usr/bin/env python3
"""Run a Stage 10 structured latticeTN job JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latticetn.runner import run_latticetn_job  # noqa: E402
from latticetn.config_schema import write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    runtime = dict(job["runtime_config"])
    runtime["output"] = str(args.output)
    result = run_latticetn_job(
        job["model_spec"],
        job["method_config"],
        runtime,
        job.get("observables"),
    )
    write_json(result, args.output)
    print(f"result written to {args.output}")
    print(f"final energy = {result['summary']['final_energy']:.12f}")
    print(f"ed_used = {result['diagnostics']['ed_used']}")
    print(f"classical_dmrg_used = {result['diagnostics']['classical_dmrg_used']}")
    print(f"lanczos_used = {result['diagnostics']['lanczos_used']}")
    print(f"ad_used = {result['diagnostics']['ad_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
