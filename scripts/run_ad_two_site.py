#!/usr/bin/env python3
"""Stage 5B two-site AD local-tensor optimization runner (the AD mainline).

Trains an MPS on the open-boundary 1D spin-1/2 Heisenberg chain by two-site AD
local-tensor optimization: bring the chain to two-site mixed-canonical form at
each bond, contract the two adjacent site tensors into a single trainable
two-site center tensor Theta, train it on the differentiable local Rayleigh
quotient with a torch optimizer (LBFGS default), then split Theta back into two
site tensors by SVD with optional max_bond_dim / cutoff truncation. Sweep
left-to-right then right-to-left. SVD/QR/canonicalization are POST-STEP split /
compression / stabilization ONLY, NEVER the solver. No Lanczos / eigh /
classical DMRG in the AD path.

Compares against exact diagonalization (small N), the one-site AD local
mainline, the global AD-MPS mainline, and the classical DMRG reference
(REFERENCE ONLY, never in the AD path). Writes ``docs/AD_TWO_SITE_REPORT.md``.

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
from latticetn.ad_two_site import train_ad_two_site, ADTwoSiteOptimizer  # noqa: E402
from latticetn.ad_local import train_ad_local  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn import dmrg as D  # noqa: E402  (reference baseline only)
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128

# Tolerances (LBFGS, seed 0). The report is the contract.
AD_TWO_SITE_TOL = {4: 1e-8, 6: 1e-5}


def _env():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _exact(N):
    H = heisenberg_dense(N, dtype=DTYPE, device="cpu")
    return exact_ground_energy(H)[0]


def _mpo(N):
    return MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)


def run_ad_two_site_case(N, chi, num_sweeps, local_steps, lr, optimizer,
                         max_bond_dim, cutoff, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = _mpo(N)
    t0 = time.perf_counter()
    res = train_ad_two_site(mps, mpo, num_sweeps=num_sweeps,
                            local_steps=local_steps, lr=lr,
                            optimizer=optimizer, max_bond_dim=max_bond_dim,
                            cutoff=cutoff)
    res["runtime_s"] = time.perf_counter() - t0
    res["N"] = N
    res["chi"] = chi
    res["exact_energy"] = _exact(N)
    res["abs_err_vs_exact"] = abs(res["final_energy"] - res["exact_energy"])
    res["below_ground"] = bool(res["final_energy"] < res["exact_energy"] - 1e-6)
    res["tol"] = AD_TWO_SITE_TOL.get(N, 1e-5)
    res["pass"] = bool(res["abs_err_vs_exact"] < res["tol"]
                       and not res["below_ground"])
    res["energy_decreased"] = bool(
        res["final_energy"] <= res["initial_energy"] + 1e-9)
    # gradient check (separate from training): fresh grads on Theta only
    tc.manual_seed(seed)
    mps2 = MPS(N, 2, chi, dtype=DTYPE)
    adtso = ADTwoSiteOptimizer(mps2, mpo, bond=N // 2 - 1)
    for p in adtso.parameters():
        if p.grad is not None:
            p.grad = None
    e = adtso.loss()
    e.backward()
    g = adtso.theta.grad
    res["grad_check"] = {
        "theta_grad_not_none": g is not None,
        "theta_grad_finite": bool(tc.isfinite(g).all()) if g is not None else False,
    }
    return res


def run_one_site_ad_reference(N, chi, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    t0 = time.perf_counter()
    r = train_ad_local(mps, _mpo(N), num_sweeps=4, local_steps=20, lr=1.0,
                       optimizer="lbfgs", stabilization="qr")
    r["runtime_s"] = time.perf_counter() - t0
    return r


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
            "The PRIMARY optimization path is two-site AD local-tensor "
            "optimization: contract two adjacent MPS site tensors into a single "
            "trainable two-site center tensor Theta, train it as a torch "
            "Parameter on the differentiable local Rayleigh quotient "
            "E(Theta)=<Theta|H_eff|Theta>/<Theta|Theta> via loss.backward() + a "
            "torch optimizer step, then split Theta back into two site tensors "
            "by SVD with optional max_bond_dim/cutoff truncation. Sweep "
            "left-to-right then right-to-left. SVD/QR/canonicalization are "
            "OPTIONAL post-step split / compression / stabilization, NEVER the "
            "solver. DMRG/Lanczos/eigh are classical reference baselines ONLY "
            "and are NOT used in the AD optimization path."
        ),
        "default_optimizer": "LBFGS, lr=1.0, per-bond local_steps=20, "
                             "num_sweeps=4, max_bond_dim=8 (optional growth)",
        "cases": [],
        "one_site_ad_reference": {},
        "global_ad_reference": {},
        "dmrg_reference": {},
        "checks": {},
        "pass": True,
        "known_limitations": [
            "Two-site AD trains one two-site block at a time; its per-bond "
            "conditioning benefits from LBFGS (the local Rayleigh problem is "
            "near-quadratic in Theta).",
            "SVD split / QR re-canonicalization appear only as post-step split "
            "/ compression / inter-bond gauge fixing (under no_grad, mutating "
            "detached data); they are NOT the optimizer and never appear in the "
            "loss path.",
            "Bond growth is optional (max_bond_dim cap); without a cap the bond "
            "can grow up to min(l*d, d*r) at each bond, which is the full "
            "two-site DMRG-style growth but driven by gradient descent on Theta.",
            "Adam is supported but slower to converge per-bond than LBFGS; "
            "LBFGS is the default/smoke optimizer.",
            "DMRG/Lanczos/eigh are classical baselines, NOT the AD mainline; "
            "they are run only for comparison.",
        ],
    }

    for N in (4, 6):
        res = run_ad_two_site_case(N, chi=8, num_sweeps=4, local_steps=20,
                                   lr=1.0, optimizer="lbfgs",
                                   max_bond_dim=8, cutoff=None, seed=0)
        report["cases"].append(res)

    ad6 = report["cases"][-1]

    # one-site AD reference (same AD mainline, single-site center sweep)
    o6 = run_one_site_ad_reference(6, chi=8, seed=0)
    report["one_site_ad_reference"] = {
        "N": 6, "solver": "one-site AD local (LBFGS, center sweep)",
        "final_energy": o6["final_energy"],
        "exact_energy": ad6["exact_energy"],
        "two_site_final_energy": ad6["final_energy"],
        "two_site_vs_one_site_diff": abs(ad6["final_energy"] - o6["final_energy"]),
    }

    # global AD-MPS reference (same mainline, all tensors trained at once)
    g6 = run_global_ad_reference(6, chi=8, seed=0, steps=300)
    report["global_ad_reference"] = {
        "N": 6, "solver": "global AD-MPS (Adam, all tensors)",
        "final_energy": g6["final_energy"],
        "exact_energy": ad6["exact_energy"],
        "two_site_final_energy": ad6["final_energy"],
        "two_site_vs_global_diff": abs(ad6["final_energy"] - g6["final_energy"]),
    }

    # classical DMRG reference (NOT in the AD path)
    d6 = run_dmrg_reference(6, chi=8, sweeps=4)
    report["dmrg_reference"] = {
        "N": 6, "solver": "dense DMRG (classical reference)",
        "final_energy": d6["final_energy"],
        "exact_energy": ad6["exact_energy"],
        "two_site_final_energy": ad6["final_energy"],
        "two_site_vs_dmrg_diff": abs(ad6["final_energy"] - d6["final_energy"]),
    }

    checks = {
        "ad_two_site_n4_vs_exact": report["cases"][0]["pass"],
        "ad_two_site_n6_vs_exact": report["cases"][1]["pass"],
        "energy_decreased_n4": report["cases"][0]["energy_decreased"],
        "energy_decreased_n6": report["cases"][1]["energy_decreased"],
        "grad_check_n4": (report["cases"][0]["grad_check"]["theta_grad_not_none"]
                          and report["cases"][0]["grad_check"]["theta_grad_finite"]),
        "grad_check_n6": (report["cases"][1]["grad_check"]["theta_grad_not_none"]
                          and report["cases"][1]["grad_check"]["theta_grad_finite"]),
        "close_to_one_site_ad": report["one_site_ad_reference"]["two_site_vs_one_site_diff"] < 1e-3,
        "close_to_global_ad": report["global_ad_reference"]["two_site_vs_global_diff"] < 1e-2,
        "close_to_dmrg_reference": report["dmrg_reference"]["two_site_vs_dmrg_diff"] < 1e-3,
        "no_below_ground": all(not c["below_ground"] for c in report["cases"]),
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def _fmt(x, p=10):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)


def render_report_md(report: dict) -> str:
    L: list[str] = []
    L.append("# Stage 5B Two-Site AD Local-Tensor Optimization Report")
    L.append("")
    L.append("Generated by `scripts/run_ad_two_site.py`.")
    L.append("")
    L.append("## Mainline statement")
    L.append("")
    L.append(f"> _{report['mainline_statement']}_")
    L.append("")
    L.append("### Stabilization policy (SVD/QR are not the solver)")
    L.append("")
    L.append("This stage's mainline is **two-site AD local-tensor "
             "optimization**: train one two-site center tensor Theta at a time "
             "on the differentiable local Rayleigh quotient "
             "(`loss.backward()` + torch optimizer step), then split Theta back "
             "into two site tensors by SVD with optional `max_bond_dim`/`cutoff` "
             "truncation. **SVD / QR / canonicalization are OPTIONAL post-step "
             "split / compression / inter-bond gauge fixing ONLY — they are NOT "
             "the solver** and never appear in the loss path. Re-canonicalizing "
             "the chain at the next bond is gauge fixing (under `no_grad`, "
             "mutating detached data). No Lanczos / `eigh` / classical DMRG is "
             "used in the AD optimization path.")
    L.append("")
    L.append("## Convention")
    L.append("")
    L.append(f"`{report['convention']}`")
    L.append(f"- dtype: `{report['dtype']}`")
    L.append(f"- optimizer: `{report['default_optimizer']}`")
    L.append("")
    L.append("## Exact comparison (two-site AD, N<=6, LBFGS)")
    L.append("")
    L.append("| N | chi | sweeps | local_steps | max_bond_dim | exact E0 | two-site AD final E | abs err | tol | below ground | energy decreased |Theta grad not-None | Theta grad finite | runtime_s | pass |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---:|:---|")
    for c in report["cases"]:
        L.append(f"| {c['N']} | {c['chi']} | {c['num_sweeps']} | {c['local_steps']} | "
                 f"{c['max_bond_dim']} | {c['exact_energy']:.10f} | "
                 f"{c['final_energy']:.10f} | {c['abs_err_vs_exact']:.2e} | "
                 f"{c['tol']:.0e} | {c['below_ground']} | "
                 f"{c['energy_decreased']} | "
                 f"{c['grad_check']['theta_grad_not_none']} | "
                 f"{c['grad_check']['theta_grad_finite']} | {c['runtime_s']:.2f} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append("## One-site AD comparison (N=6)")
    L.append("")
    o = report["one_site_ad_reference"]
    L.append(f"- one-site AD local (LBFGS, center sweep) final E: `{o['final_energy']:.10f}`")
    L.append(f"- two-site AD final E: `{o['two_site_final_energy']:.10f}`")
    L.append(f"- exact E0: `{o['exact_energy']:.10f}`")
    L.append(f"|two-site - one-site AD|: `{o['two_site_vs_one_site_diff']:.2e}`")
    L.append("")
    L.append("Both are AD strategies on differentiable Rayleigh quotients; "
             "two-site AD additionally allows bond growth / truncation at each "
             "split and reaches the same variational minimum.")
    L.append("")
    L.append("## Global AD-MPS comparison (N=6)")
    L.append("")
    g = report["global_ad_reference"]
    L.append(f"- global AD-MPS (Adam, all tensors) final E: `{g['final_energy']:.10f}`")
    L.append(f"- two-site AD final E: `{g['two_site_final_energy']:.10f}`")
    L.append(f"- exact E0: `{g['exact_energy']:.10f}`")
    L.append(f"|two-site - global AD|: `{g['two_site_vs_global_diff']:.2e}`")
    L.append("")
    L.append("## DMRG reference comparison (N=6)")
    L.append("")
    d = report["dmrg_reference"]
    L.append(f"- classical DMRG (dense reference) final E: `{d['final_energy']:.10f}`")
    L.append(f"- two-site AD final E: `{d['two_site_final_energy']:.10f}`")
    L.append(f"- exact E0: `{d['exact_energy']:.10f}`")
    L.append(f"|two-site AD - DMRG|: `{d['two_site_vs_dmrg_diff']:.2e}`")
    L.append("")
    L.append("DMRG/Lanczos/eigh are classical baselines and are **not** part of "
             "the AD optimization path; they are run here only to confirm the "
             "two-site AD solver reaches the same variational minimum.")
    L.append("")
    L.append("## Energy history (sampled, per sweep)")
    L.append("")
    for c in report["cases"]:
        L.append(f"N={c['N']} (initial E = {c['initial_energy']:.6f}):")
        L.append("")
        L.append("| sweep | direction | energy after | grad norm | max bond | max trunc |")
        L.append("|---:|:---:|---:|---:|---:|---:|")
        hist = c["energy_history"]
        # history[0] is the pre-sweep record; sweeps start at index 1
        gn = c["grad_norm_history"]
        bd = c["bond_dim_history"]
        tr = c["truncation_error_history"]
        L.append(f"| - | init | {hist[0]:.10f} | {gn[0]:.3e} | {bd[0]} | {tr[0]:.2e} |")
        for k, s in enumerate(c["sweeps"]):
            idx = k + 1
            L.append(f"| {s['sweep']} | {s['direction']} | {hist[idx]:.10f} | "
                     f"{gn[idx]:.3e} | {bd[idx]} | {tr[idx]:.2e} |")
        L.append("")
    L.append("## Gradient check")
    L.append("")
    L.append("Loss = `<Theta|H_eff|Theta>/<Theta|Theta>` with only the two-site "
             "center tensor Theta trainable (fully differentiable einsum on "
             "Theta and the frozen constant MPO environments; no `detach()/.data/"
             "no_grad()/unnecessary .item()` in the loss path). After "
             "`loss.backward()` Theta receives a non-None, finite gradient; the "
             "frozen environment tensors receive None. The SVD split / QR "
             "re-canonicalization live only in post-step / preprocessing helpers "
             "under `no_grad`.")
    L.append("")
    L.append("## Bond growth / compression")
    L.append("")
    L.append("At each bond the optimized Theta is split by SVD into two site "
             "tensors. With `max_bond_dim` set, the kept bond is "
             "`min(max_bond_dim, #singular values above cutoff)`; without a cap "
             "the bond may grow up to `min(l*d, d*r)` (full two-site growth). "
             "The discarded weight is recorded as the per-sweep truncation "
             "error. This is compression / stabilization, NOT the optimizer.")
    L.append("")
    L.append("| N | final bond dims | max bond |")
    L.append("|---:|:---|---:|")
    for c in report["cases"]:
        L.append(f"| {c['N']} | {c['final_bond_dims']} | {c['max_bond']} |")
    L.append("")
    L.append("## Truncation errors")
    L.append("")
    for c in report["cases"]:
        L.append(f"N={c['N']} per-sweep max truncation errors:")
        L.append("")
        L.append("| sweep | direction | max truncation | per-bond truncations |")
        L.append("|---:|:---:|---:|:---|")
        for s in c["sweeps"]:
            pbt = ", ".join(f"{t:.2e}" for t in s["per_bond_trunc"])
            L.append(f"| {s['sweep']} | {s['direction']} | {s['max_trunc']:.2e} | {pbt} |")
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
    parser.add_argument("--max-bond-dim", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "AD_TWO_SITE_REPORT.md")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    if args.json_output or args.print:
        tc.manual_seed(args.seed)
        mps = MPS(args.N, 2, args.chi, dtype=DTYPE)
        res = train_ad_two_site(mps, _mpo(args.N), num_sweeps=args.num_sweeps,
                                local_steps=args.local_steps, lr=args.lr,
                                optimizer=args.optimizer,
                                max_bond_dim=args.max_bond_dim,
                                cutoff=args.cutoff)
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
    print(f"ad two-site smoke: pass={report['pass']} report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
