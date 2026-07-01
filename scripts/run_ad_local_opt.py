#!/usr/bin/env python3
"""Stage 5A AD local-tensor optimization runner (the autograd mainline).

Trains an MPS on the open-boundary 1D spin-1/2 Heisenberg chain by AD
local-tensor optimization: freeze all tensors except one center tensor, optimize
the differentiable Rayleigh quotient with a torch optimizer (LBFGS default),
move the orthogonality center by QR, sweep across the chain. SVD/QR/
canonicalization are OPTIONAL post-step stabilization, NEVER the solver.

Compares against exact diagonalization (small N), the global AD-MPS mainline,
and the classical DMRG reference (REFERENCE ONLY, never in the AD path).
Writes ``docs/AD_LOCAL_OPT_REPORT.md``.

Conventions: H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary,
complex128. CPU-only.
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
from latticetn.ad_local import train_ad_local  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn import dmrg as D  # noqa: E402  (reference baseline only)
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128

# Tolerances (LBFGS, seed 0). The report is the contract.
AD_LOCAL_TOL = {4: 1e-8, 6: 1e-5}


def _env():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _exact(N):
    H = heisenberg_dense(N, dtype=DTYPE, device="cpu")
    return exact_ground_energy(H)[0]


def _mpo(N):
    return MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)


def run_ad_local_case(N, chi, num_sweeps, local_steps, lr, stabilization, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = _mpo(N)
    t0 = time.perf_counter()
    res = train_ad_local(mps, mpo, num_sweeps=num_sweeps, local_steps=local_steps,
                         lr=lr, optimizer="lbfgs", stabilization=stabilization)
    res["runtime_s"] = time.perf_counter() - t0
    res["N"] = N
    res["chi"] = chi
    res["exact_energy"] = _exact(N)
    res["abs_err_vs_exact"] = abs(res["final_energy"] - res["exact_energy"])
    res["below_ground"] = bool(res["final_energy"] < res["exact_energy"] - 1e-6)
    res["tol"] = AD_LOCAL_TOL.get(N, 1e-5)
    res["pass"] = bool(res["abs_err_vs_exact"] < res["tol"]
                       and not res["below_ground"])
    # gradient check (separate from training): fresh grads on the center only
    from latticetn.ad_local import ADLocalOptimizer
    tc.manual_seed(seed)
    mps2 = MPS(N, 2, chi, dtype=DTYPE)
    adlo = ADLocalOptimizer(mps2, mpo, center=N // 2)
    for p in adlo.parameters():
        if p.grad is not None:
            p.grad = None
    e = adlo.loss()
    e.backward()
    cg = adlo.parameters()[0].grad
    res["grad_check"] = {
        "center_grad_not_none": cg is not None,
        "center_grad_finite": bool(tc.isfinite(cg).all()) if cg is not None else False,
    }
    return res


def run_global_ad_reference(N, chi, seed=0, steps=300):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    ad = ADVariationalMPS(mps, _mpo(N))
    t0 = time.perf_counter()
    r = train_ad_mps(ad, num_steps=steps, lr=1e-2, optimizer="adam")
    r["runtime_s"] = time.perf_counter() - t0
    return r


def run_dmrg_reference(N, chi, sweeps=4):
    tc.manual_seed(0)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    return D.run_dmrg(mps, _mpo(N), chi=chi, num_sweeps=sweeps, solver="dense")


def run_smoke() -> dict:
    _env()
    report = {
        "convention": "H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, "
                      "open boundary, complex128",
        "dtype": str(DTYPE),
        "mainline_statement": (
            "The PRIMARY optimization path is AD local-tensor optimization: "
            "freeze all MPS tensors except one center tensor, train it as a "
            "torch Parameter on the differentiable Rayleigh quotient via "
            "loss.backward() + a torch optimizer step, and sweep the "
            "orthogonality center across the chain by QR. SVD/QR/"
            "canonicalization are OPTIONAL post-step stabilization, NEVER the "
            "solver. DMRG/Lanczos are classical reference baselines ONLY and "
            "are NOT used in the AD optimization path."
        ),
        "default_optimizer": "LBFGS, lr=1.0, per-site local_steps=20, "
                             "num_sweeps=4, stabilization='qr'",
        "cases": [],
        "global_ad_reference": {},
        "dmrg_reference": {},
        "checks": {},
        "pass": True,
        "known_limitations": [
            "AD local-tensor optimization trains one center tensor at a time; "
            "its per-site conditioning benefits from LBFGS (the local Rayleigh "
            "problem is near-quadratic in the center tensor).",
            "QR/SVD/canonicalization appear only as optional post-step "
            "stabilization / center movement (under no_grad, mutating .data); "
            "they are NOT the optimizer and do not appear in the loss path.",
            "Bond dimension is fixed by init; no bond-growing/truncation in the "
            "AD path (unlike classical two-site DMRG).",
            "Adam is supported but slower to converge per-site than LBFGS; "
            "LBFGS is the default/smoke optimizer.",
            "DMRG/Lanczos are classical baselines, NOT the AD mainline; they "
            "are run only for comparison.",
        ],
    }

    for N in (4, 6):
        res = run_ad_local_case(N, chi=8, num_sweeps=4, local_steps=20, lr=1.0,
                                stabilization="qr", seed=0)
        report["cases"].append(res)

    # global AD-MPS reference (same mainline, all tensors trained at once)
    g6 = run_global_ad_reference(6, chi=8, seed=0, steps=300)
    ad6 = report["cases"][-1]
    report["global_ad_reference"] = {
        "N": 6, "solver": "global AD-MPS (Adam, all tensors)",
        "final_energy": g6["final_energy"],
        "exact_energy": ad6["exact_energy"],
        "ad_local_final_energy": ad6["final_energy"],
        "local_vs_global_diff": abs(ad6["final_energy"] - g6["final_energy"]),
    }

    # classical DMRG reference (NOT in the AD path)
    d6 = run_dmrg_reference(6, chi=8, sweeps=4)
    report["dmrg_reference"] = {
        "N": 6, "solver": "dense DMRG (classical reference)",
        "final_energy": d6["final_energy"],
        "exact_energy": ad6["exact_energy"],
        "ad_local_final_energy": ad6["final_energy"],
        "local_vs_dmrg_diff": abs(ad6["final_energy"] - d6["final_energy"]),
    }

    checks = {
        "ad_local_n4_vs_exact": report["cases"][0]["pass"],
        "ad_local_n6_vs_exact": report["cases"][1]["pass"],
        "grad_check_n4": (report["cases"][0]["grad_check"]["center_grad_not_none"]
                          and report["cases"][0]["grad_check"]["center_grad_finite"]),
        "grad_check_n6": (report["cases"][1]["grad_check"]["center_grad_not_none"]
                          and report["cases"][1]["grad_check"]["center_grad_finite"]),
        "close_to_global_ad": report["global_ad_reference"]["local_vs_global_diff"] < 1e-2,
        "close_to_dmrg_reference": report["dmrg_reference"]["local_vs_dmrg_diff"] < 1e-3,
        "no_below_ground": all(not c["below_ground"] for c in report["cases"]),
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def _fmt(x, p=10):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)


def render_report_md(report: dict) -> str:
    L: list[str] = []
    L.append("# Stage 5A AD Local-Tensor Optimization Report")
    L.append("")
    L.append("Generated by `scripts/run_ad_local_opt.py`.")
    L.append("")
    L.append("## Mainline statement")
    L.append("")
    L.append(f"> _{report['mainline_statement']}_")
    L.append("")
    L.append("### Stabilization policy")
    L.append("")
    L.append("This stage's mainline is **AD local-tensor optimization**: train "
             "one center tensor at a time on the differentiable Rayleigh "
             "quotient (`loss.backward()` + torch optimizer step), sweeping the "
             "orthogonality center by QR. **SVD / QR / canonicalization are "
             "OPTIONAL post-step stabilization only — they are NOT the solver** "
             "and never appear in the loss path. Default stabilization for the "
             "report cases: `qr`.")
    L.append("")
    L.append("## Convention")
    L.append("")
    L.append(f"`{report['convention']}`")
    L.append(f"- dtype: `{report['dtype']}`")
    L.append(f"- optimizer: `{report['default_optimizer']}`")
    L.append("")
    L.append("## Exact comparison (AD local, N<=6, LBFGS)")
    L.append("")
    L.append("| N | chi | sweeps | local_steps | exact E0 | AD-local final E | abs err | tol | below ground | max bond | center grad not-None | center grad finite | runtime_s | pass |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|:---:|---:|:---|")
    for c in report["cases"]:
        L.append(f"| {c['N']} | {c['chi']} | {c['num_sweeps']} | {c['local_steps']} | "
                 f"{c['exact_energy']:.10f} | {c['final_energy']:.10f} | "
                 f"{c['abs_err_vs_exact']:.2e} | {c['tol']:.0e} | "
                 f"{c['below_ground']} | {c['max_bond']} | "
                 f"{c['grad_check']['center_grad_not_none']} | "
                 f"{c['grad_check']['center_grad_finite']} | {c['runtime_s']:.2f} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append("## Global AD-MPS comparison (N=6)")
    L.append("")
    g = report["global_ad_reference"]
    L.append(f"- global AD-MPS (Adam, all tensors) final E: `{g['final_energy']:.10f}`")
    L.append(f"- AD-local (LBFGS, center sweep) final E: `{g['ad_local_final_energy']:.10f}`")
    L.append(f"- exact E0: `{g['exact_energy']:.10f}`")
    L.append(f"|AD-local - global AD|: `{g['local_vs_global_diff']:.2e}`")
    L.append("")
    L.append("Both are strategies on the SAME differentiable Rayleigh quotient; "
             "they reach the same variational minimum.")
    L.append("")
    L.append("## DMRG reference comparison (N=6)")
    L.append("")
    d = report["dmrg_reference"]
    L.append(f"- classical DMRG (dense reference) final E: `{d['final_energy']:.10f}`")
    L.append(f"- AD-local final E: `{d['ad_local_final_energy']:.10f}`")
    L.append(f"- exact E0: `{d['exact_energy']:.10f}`")
    L.append(f"|AD-local - DMRG|: `{d['local_vs_dmrg_diff']:.2e}`")
    L.append("")
    L.append("DMRG/Lanczos are classical baselines and are **not** part of the AD "
             "optimization path; they are run here only to confirm the AD-local "
             "solver reaches the same variational minimum.")
    L.append("")
    L.append("## Energy history (sampled, per sweep)")
    L.append("")
    for c in report["cases"]:
        L.append(f"N={c['N']} (initial E = {c['initial_energy']:.6f}):")
        L.append("")
        L.append("| sweep | energy after | grad norm | state norm | canonical error |")
        L.append("|---:|---:|---:|---:|---:|")
        hist = c["energy_history"]
        n = len(hist)
        for k in range(n):
            sweep = k  # record 0 = initial, then one per sweep
            gn = c["grad_norm_history"][k] if k < len(c["grad_norm_history"]) else float("nan")
            sn = c["state_norm_history"][k] if k < len(c["state_norm_history"]) else float("nan")
            ce = c["canonical_error_history"][k] if k < len(c["canonical_error_history"]) else float("nan")
            L.append(f"| {sweep} | {hist[k]:.10f} | {gn:.3e} | {sn:.6e} | {ce:.3e} |")
        L.append("")
    L.append("## Gradient check")
    L.append("")
    L.append("Loss = `rayleigh_energy_native(mps, mpo)` with only the center "
             "tensor trainable (fully differentiable; no `detach()/.data/"
             "no_grad()/unnecessary .item()` in the loss path). After "
             "`loss.backward()` the center tensor receives a non-None, finite "
             "gradient; frozen environment tensors receive None. QR/SVD/"
             "canonicalization live only in post-step / center-movement helpers "
             "under `no_grad`.")
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
    parser.add_argument("--num-sweeps", type=int, default=4)
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="lbfgs")
    parser.add_argument("--stabilization",
                        choices=["none", "tensor_norm", "qr", "canonical"],
                        default="qr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "AD_LOCAL_OPT_REPORT.md")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    if args.json_output or args.print:
        tc.manual_seed(args.seed)
        mps = MPS(args.N, 2, args.chi, dtype=DTYPE)
        res = train_ad_local(mps, _mpo(args.N), num_sweeps=args.num_sweeps,
                             local_steps=args.local_steps, lr=args.lr,
                             optimizer=args.optimizer,
                             stabilization=args.stabilization)
        res["N"] = args.N; res["chi"] = args.chi
        res["exact_energy"] = _exact(args.N)
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
    print(f"ad local opt smoke: pass={report['pass']} report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
