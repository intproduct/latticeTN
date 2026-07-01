#!/usr/bin/env python3
"""Stage 4R AD-MPS Heisenberg runner (the autograd mainline).

Trains a differentiable MPS on the open-boundary 1D spin-1/2 Heisenberg chain
by gradient descent on the Rayleigh quotient (PyTorch autograd + torch
optimizer). Compares against exact diagonalization (N<=6) and against the
classical DMRG reference (DMRG is a REFERENCE ONLY, never part of the AD path).
Writes ``docs/AD_VARIATIONAL_REPORT.md``.

Physics conventions unchanged: H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0,
open boundary, complex128. CPU-only.

Reported content (per docs/AD_VARIATIONAL_PROTOCOL.md stop conditions):
- exact comparison (N<=6),
- DMRG baseline comparison,
- energy history (sampled), gradient/norm/max-bond diagnostics,
- gradient check (autograd populates all params),
- optimizer settings,
- pass/fail, known limitations,
- explicit statement that Lanczos/DMRG are classical baselines, not the AD
  mainline.
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
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn import dmrg as D  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128

# Tolerances for the AD report (Adam, seed 0). Recorded so the report is the
# contract; relaxing them requires justification + a report update.
AD_TOL = {4: 1e-6, 6: 1e-3}


def _env():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _exact(N):
    H = heisenberg_dense(N, dtype=DTYPE, device="cpu")
    E0, _ = exact_ground_energy(H)
    return E0


def _bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def run_ad_case(N, chi, steps, lr, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    ad = ADVariationalMPS(mps, mpo)
    t0 = time.perf_counter()
    res = train_ad_mps(ad, num_steps=steps, lr=lr, optimizer="adam",
                       record_every=max(1, steps // 10))
    res["runtime_s"] = time.perf_counter() - t0
    res["N"] = N
    res["chi"] = chi
    res["exact_energy"] = _exact(N)
    res["abs_err_vs_exact"] = abs(res["final_energy"] - res["exact_energy"])
    res["below_ground"] = bool(res["final_energy"] < res["exact_energy"] - 1e-6)
    res["bond_dims"] = _bond_dims(ad.mps)
    res["tol"] = AD_TOL.get(N, 1e-3)
    res["pass"] = bool(res["abs_err_vs_exact"] < res["tol"]
                       and not res["below_ground"])
    # gradient check (separate from training): fresh grads on the current params
    for p in ad.parameters():
        if p.grad is not None:
            p.grad = None
    e = ad.loss()
    e.backward()
    res["grad_check"] = {
        "all_not_none": all(p.grad is not None for p in ad.parameters()),
        "all_finite": all(tc.isfinite(p.grad).all() if p.grad is not None else False
                          for p in ad.parameters()),
    }
    return res


def run_dmrg_reference(N, chi, sweeps=4):
    tc.manual_seed(0)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return D.run_dmrg(mps, mpo, chi=chi, num_sweeps=sweeps, solver="dense")


def run_smoke() -> dict:
    _env()
    report: dict = {
        "convention": "H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary, complex128",
        "dtype": str(DTYPE),
        "mainline_statement": (
            "The PRIMARY optimization path is a differentiable Rayleigh quotient "
            "trained with PyTorch autograd + a torch optimizer (Adam/LBFGS). "
            "Lanczos/DMRG (Stage 4A/4B) are classical reference baselines ONLY "
            "and are NOT used in the AD optimization path."
        ),
        "optimizer_default": "Adam, lr=1e-2, per-tensor L2 renormalization after each step (outside the loss graph)",
        "cases": [],
        "dmrg_reference": {},
        "checks": {},
        "pass": True,
        "known_limitations": [
            "AD-MPS uses first-order Adam by default; more steps needed for full chi convergence at larger N (see tolerances).",
            "Per-tensor L2 renormalization is a stability projection under no_grad (outside the differentiable loss); the Rayleigh quotient is scale-invariant so it does not change the physics.",
            "Fixed bond dimension (set by init); no bond growing/truncation in the AD path (unlike DMRG).",
            "LBFGS is supported but Adam is the default/smoke optimizer.",
            "DMRG/Lanczos are classical baselines, NOT the AD mainline; they are run only for comparison.",
        ],
    }

    # AD cases (N=4 and N=6)
    for N in (4, 6):
        chi = 8
        steps = 200
        res = run_ad_case(N, chi, steps=steps, lr=1e-2, seed=0)
        report["cases"].append(res)

    # DMRG reference for comparison (NOT in the AD path) on N=6
    r6 = run_dmrg_reference(6, chi=8, sweeps=4)
    ad6 = report["cases"][-1]
    report["dmrg_reference"] = {
        "N": 6, "solver": "dense DMRG (classical reference)",
        "final_energy": r6["final_energy"],
        "exact_energy": ad6["exact_energy"],
        "ad_final_energy": ad6["final_energy"],
        "ad_vs_dmrg_diff": abs(ad6["final_energy"] - r6["final_energy"]),
    }

    checks = {
        "ad_n4_vs_exact": report["cases"][0]["pass"],
        "ad_n6_vs_exact": report["cases"][1]["pass"],
        "grad_check_n4": (report["cases"][0]["grad_check"]["all_not_none"]
                          and report["cases"][0]["grad_check"]["all_finite"]),
        "grad_check_n6": (report["cases"][1]["grad_check"]["all_not_none"]
                          and report["cases"][1]["grad_check"]["all_finite"]),
        "ad_close_to_dmrg_reference": report["dmrg_reference"]["ad_vs_dmrg_diff"] < 1e-3,
        "no_below_ground": all(not c["below_ground"] for c in report["cases"]),
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def render_report_md(report: dict) -> str:
    L: list[str] = []
    L.append("# Stage 4R AD-MPS Variational Solver Report")
    L.append("")
    L.append("Generated by `scripts/run_ad_mps_heisenberg.py`.")
    L.append("")
    L.append("## Mainline statement")
    L.append("")
    L.append(f"> _{report['mainline_statement']}_")
    L.append("")
    L.append("## Convention")
    L.append("")
    L.append(f"`{report['convention']}`")
    L.append(f"- dtype: `{report['dtype']}`")
    L.append(f"- optimizer: `{report['optimizer_default']}`")
    L.append("")
    L.append("## Exact comparison (AD mainline, N<=6, Adam)")
    L.append("")
    L.append("| N | chi | steps | exact E0 | AD final E | abs err | tol | below ground | max bond | grad all not-None | grad finite | runtime_s | pass |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|:---:|---:|:---|")
    for c in report["cases"]:
        L.append(f"| {c['N']} | {c['chi']} | {c['num_steps']} | "
                 f"{c['exact_energy']:.10f} | {c['final_energy']:.10f} | "
                 f"{c['abs_err_vs_exact']:.2e} | {c['tol']:.0e} | "
                 f"{c['below_ground']} | {c['max_bond']} | "
                 f"{c['grad_check']['all_not_none']} | "
                 f"{c['grad_check']['all_finite']} | {c['runtime_s']:.2f} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append("### Energy history (sampled)")
    L.append("")
    for c in report["cases"]:
        L.append(f"N={c['N']} (initial E = {c['initial_energy']:.6f}):")
        L.append("")
        L.append("| step | energy | grad norm | norm(<psi|psi>) |")
        L.append("|---:|---:|---:|---:|")
        n = len(c["energy_history"])
        for k in range(n):
            step = (c["num_steps"] // max(1, n - 1)) * k if n > 1 else 0
            L.append(f"| {step} | {c['energy_history'][k]:.10f} | "
                     f"{c['grad_norm_history'][k]:.3e} | "
                     f"{c['norm_history'][k]:.6e} |")
        L.append("")
    L.append("## DMRG reference comparison (N=6)")
    L.append("")
    d = report["dmrg_reference"]
    L.append(f"- classical DMRG (dense reference) final E: `{d['final_energy']:.10f}`")
    L.append(f"- exact E0: `{d['exact_energy']:.10f}`")
    L.append(f"- AD mainline final E: `{d['ad_final_energy']:.10f}`")
    L.append(f"|AD - DMRG|: `{d['ad_vs_dmrg_diff']:.2e}`")
    L.append("")
    L.append("DMRG/Lanczos are classical baselines and are **not** part of the AD "
             "optimization path; they are run here only to confirm the AD solver "
             "reaches the same variational minimum.")
    L.append("")
    L.append("## Gradient check")
    L.append("")
    L.append("`loss = rayleigh_energy_native(mps, mpo)` (fully differentiable, no "
             "`detach()`/`.data`/unnecessary `.item()`/`no_grad()` in the loss path). "
             "After `loss.backward()`, all trainable MPS parameters receive a "
             "non-None, finite gradient. The per-step L2 renormalization runs under "
             "`no_grad` mutating `.data` — a stability projection outside the loss "
             "graph, scale-invariant for the Rayleigh quotient, identical in spirit "
             "to the Stage 1 `_full_normalize` routine.")
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
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "AD_VARIATIONAL_REPORT.md")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    if args.json_output or args.print:
        # single-case run
        tc.manual_seed(args.seed)
        mps = MPS(args.N, 2, args.chi, dtype=DTYPE)
        mpo = MPO.from_bonds(args.N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
        ad = ADVariationalMPS(mps, mpo)
        res = train_ad_mps(ad, num_steps=args.steps, lr=args.lr, optimizer=args.optimizer)
        res["N"] = args.N; res["chi"] = args.chi; res["exact_energy"] = _exact(args.N)
        res["abs_err_vs_exact"] = abs(res["final_energy"] - res["exact_energy"])
        res["below_ground"] = bool(res["final_energy"] < res["exact_energy"] - 1e-6)
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
    print(f"ad variational smoke: pass={report['pass']} report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
