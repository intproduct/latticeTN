#!/usr/bin/env python3
"""Stage 5A gauge-stabilized AD-MPS Heisenberg runner.

Trains an AD-MPS with each projection (none / tensor_norm / canonical) on the
open-boundary spin-1/2 Heisenberg chain and writes
``docs/AD_GAUGE_REPORT.md``, comparing the three gauges (energy, grad norm,
state norm, canonical error) against exact (N<=6) and the DMRG reference.

Projection is a NON-differentiable gauge stabilization applied AFTER each
optimizer step, OUTSIDE the loss graph. The main loss is the differentiable
Rayleigh quotient ``rayleigh_energy_native(mps, mpo)``. DMRG/Lanczos are
classical reference baselines, NOT in the AD path.

Conventions unchanged: S = sigma/2, J = 1.0, open boundary, complex128, CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import (  # noqa: E402
    ADVariationalMPS, train_ad_mps, _canonical_error,
)
from latticetn import dmrg as D  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128
PROJECTIONS = ("none", "tensor_norm", "canonical")


def _env():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _exact(N):
    H = heisenberg_dense(N, dtype=DTYPE, device="cpu")
    E0, _ = exact_ground_energy(H)
    return E0


def run_proj(N, chi, steps, projection, lr=1e-2, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    ad = ADVariationalMPS(mps, mpo)
    t0 = time.perf_counter()
    res = train_ad_mps(ad, num_steps=steps, lr=lr, projection=projection,
                       record_every=max(1, steps // 5))
    res["runtime_s"] = time.perf_counter() - t0
    res["N"] = N
    res["chi"] = chi
    res["exact_energy"] = _exact(N)
    res["abs_err_vs_exact"] = abs(res["final_energy"] - res["exact_energy"])
    res["below_ground"] = bool(res["final_energy"] < res["exact_energy"] - 1e-6)
    return res


def run_smoke() -> dict:
    _env()
    report: dict = {
        "convention": "H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary, complex128",
        "dtype": str(DTYPE),
        "projection_statement": (
            "Projection (none / tensor_norm / canonical) is a NON-differentiable "
            "gauge stabilization applied AFTER each optimizer step, OUTSIDE the "
            "autograd loss graph. The main loss remains the differentiable "
            "Rayleigh quotient rayleigh_energy_native(mps, mpo). DMRG/Lanczos "
            "are classical reference baselines and are NOT in the AD path."
        ),
        "optimizer": "Adam, lr=1e-2",
        "cases": [],
        "dmrg_reference": {},
        "checks": {},
        "pass": True,
        "known_limitations": [
            "`canonical` uses the Stage 3A left-canonical QR sweep written back onto .data under no_grad; it preserves the dense state up to a global phase.",
            "`none` lacks scale stabilization and converges slower / less accurately at fixed steps; it is included for ablation, not as the recommended gauge.",
            "Projection does NOT change the physics (Rayleigh quotient is gauge- and scale-invariant); it only stabilizes the optimizer / restores canonical gauge.",
            "Bond dimension is fixed by init (no growing/truncation in the AD path).",
            "No TEBD / TDVP / finite-T / GPU performance benchmark.",
        ],
    }

    # N=4 and N=6 across all three projections
    for N in (4, 6):
        for proj in PROJECTIONS:
            steps = 200
            res = run_proj(N, chi=8, steps=steps, projection=proj)
            # accept if not below ground and (stabilized proj within tight tol,
            # or `none` within a loose tol).
            ok = (not res["below_ground"])
            if proj in ("tensor_norm", "canonical"):
                ok = ok and res["abs_err_vs_exact"] < (1e-5 if N == 4 else 5e-4)
            else:
                ok = ok and res["abs_err_vs_exact"] < 1e-2
            res["pass"] = bool(ok)
            report["cases"].append(res)

    # DMRG reference (NOT in AD path), N=6
    tc.manual_seed(0)
    mps_d = MPS(6, 2, 8, dtype=DTYPE)
    mpo = MPO.from_bonds(6, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    r6 = D.run_dmrg(mps_d, mpo, chi=8, num_sweeps=4, solver="dense")
    canon6 = next(c for c in report["cases"] if c["N"] == 6 and c["projection"] == "canonical")
    report["dmrg_reference"] = {
        "N": 6, "solver": "dense DMRG (classical reference)",
        "final_energy": r6["final_energy"],
        "exact_energy": canon6["exact_energy"],
        "ad_canonical_final": canon6["final_energy"],
        "ad_canonical_vs_dmrg": abs(canon6["final_energy"] - r6["final_energy"]),
    }

    # checks
    canon_n6 = canon6
    tn_n6 = next(c for c in report["cases"] if c["N"] == 6 and c["projection"] == "tensor_norm")
    checks = {
        "all_cases_pass": all(c["pass"] for c in report["cases"]),
        "no_below_ground": all(not c["below_ground"] for c in report["cases"]),
        "canonical_not_worse_than_tensor_norm_n6":
            canon_n6["final_energy"] <= tn_n6["final_energy"] + 1e-4,
        "canonical_error_decreases_n6":
            canon_n6["canonical_error_history"][-1] < canon_n6["canonical_error_history"][0]
            and canon_n6["canonical_error_history"][-1] < 1e-9,
        "ad_canonical_close_to_dmrg_reference":
            report["dmrg_reference"]["ad_canonical_vs_dmrg"] < 1e-3,
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def render_report_md(report: dict) -> str:
    L: list[str] = []
    L.append("# Stage 5A Gauge-Stabilized AD-MPS Report")
    L.append("")
    L.append("Generated by `scripts/run_ad_gauge_heisenberg.py`.")
    L.append("")
    L.append("## Projection statement")
    L.append("")
    L.append(f"> _{report['projection_statement']}_")
    L.append("")
    L.append("## Convention")
    L.append("")
    L.append(f"`{report['convention']}`")
    L.append(f"- dtype: `{report['dtype']}`, optimizer: `{report['optimizer']}`")
    L.append("")
    L.append("## Projection comparison (N=4 / N=6, Adam, 200 steps)")
    L.append("")
    L.append("| N | projection | exact E0 | AD final E | abs err | below ground | init canon err | final canon err | final state norm | final grad norm | runtime_s | pass |")
    L.append("|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|:---|")
    for c in report["cases"]:
        L.append(f"| {c['N']} | {c['projection']} | {c['exact_energy']:.10f} | "
                 f"{c['final_energy']:.10f} | {c['abs_err_vs_exact']:.2e} | "
                 f"{c['below_ground']} | {c['canonical_error_history'][0]:.2e} | "
                 f"{c['canonical_error_history'][-1]:.2e} | "
                 f"{c['state_norm_history'][-1]:.4e} | "
                 f"{c['grad_norm_history'][-1]:.3e} | {c['runtime_s']:.2f} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append("### Energy history (N=6, by projection)")
    L.append("")
    for c in report["cases"]:
        if c["N"] != 6:
            continue
        L.append(f"N=6, projection={c['projection']}:")
        L.append("")
        L.append("| step | energy | grad norm | state norm | canonical error |")
        L.append("|---:|---:|---:|---:|---:|")
        n = len(c["energy_history"])
        for k in range(n):
            step = (c["num_steps"] // max(1, n - 1)) * k if n > 1 else 0
            L.append(f"| {step} | {c['energy_history'][k]:.10f} | "
                     f"{c['grad_norm_history'][k]:.3e} | "
                     f"{c['state_norm_history'][k]:.4e} | "
                     f"{c['canonical_error_history'][k]:.2e} |")
        L.append("")
    L.append("## DMRG reference (N=6, classical baseline — NOT in the AD path)")
    L.append("")
    d = report["dmrg_reference"]
    L.append(f"- classical DMRG final E: `{d['final_energy']:.10f}`")
    L.append(f"- exact E0: `{d['exact_energy']:.10f}`")
    L.append(f"- AD canonical final E: `{d['ad_canonical_final']:.10f}`")
    L.append(f"|AD canonical - DMRG|: `{d['ad_canonical_vs_dmrg']:.2e}`")
    L.append("")
    L.append("## Overall pass/fail")
    L.append("")
    L.append(f"**pass: `{report['pass']}`**")
    L.append("")
    L.append("| check | passed |")
    L.append("|---|:---|")
    for k, v in report["checks"].items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Known limitations")
    L.append("")
    for lim in report["known_limitations"]:
        L.append(f"- {lim}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection", choices=list(PROJECTIONS), default="canonical")
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "AD_GAUGE_REPORT.md")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    if args.json_output or args.print:
        res = run_proj(args.N, args.chi, args.steps, args.projection, args.lr, args.seed)
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(res, indent=2, default=str),
                                        encoding="utf-8")
        if args.print or not args.json_output:
            print(json.dumps(res, indent=2, default=str))
        return 0

    report = run_smoke()
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_report_md(report), encoding="utf-8")
    print(f"ad gauge smoke: pass={report['pass']} report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
