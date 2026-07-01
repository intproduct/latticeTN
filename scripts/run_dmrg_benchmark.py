#!/usr/bin/env python3
"""Stage 4B scalable DMRG benchmark runner.

Runs the two-site DMRG with both the dense (Stage 4A reference) and Lanczos
(Stage 4B matrix-free) solvers, plus a chi sweep and an N=10/12 CPU smoke, and
writes ``docs/DMRG_BENCHMARK_REPORT.md``.

Reported content (per docs/DMRG_BENCHMARK_PROTOCOL.md stop conditions):
- dense-vs-matrix-free apply comparison,
- Lanczos-vs-dense eigensolver comparison (small system),
- small-system exact comparison (N<=6) for both solvers,
- N=10/12 smoke (finite, energy decrease, bond dims, runtime),
- chi sweep,
- per-case energy history / final energy / energy per bond / runtime /
  max bond / truncation errors / solver,
- pass/fail, known limitations.

CLI: N, chi, sweeps, seed, solver, dtype, device.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn import dmrg as D  # noqa: E402
from latticetn import lanczos as LZ  # noqa: E402
from latticetn import contractions as K  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE_MAP = {"complex128": tc.complex128, "complex64": tc.complex64}

BETHE_E0 = 0.25 - math.log(2.0)


def _env():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _exact(N, dtype, device):
    try:
        H = heisenberg_dense(N, dtype=dtype, device=device)
        E0, _ = exact_ground_energy(H)
        return E0
    except Exception:
        return None


def run_single(N, chi, sweeps, solver, seed, dtype, device):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=dtype, device=device)
    mpo = MPO.from_bonds(N, 2, dtype=dtype, device=device).generate_heisenberg(J=1.0)
    t0 = time.perf_counter()
    res = D.run_dmrg(mps, mpo, chi=chi, num_sweeps=sweeps, solver=solver, seed=seed)
    res["runtime_s"] = time.perf_counter() - t0
    res["exact_energy"] = _exact(N, dtype, device)
    return res


def run_smoke(dtype=tc.complex128, device="cpu") -> dict:
    _env()
    report: dict = {
        "convention": "H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary, complex128",
        "dtype": str(dtype),
        "device": str(device),
        "bethe_reference": BETHE_E0,
        "matrix_free_vs_dense": {},
        "lanczos_vs_dense": {},
        "exact_compare": [],
        "chi_sweep": [],
        "smoke": [],
        "checks": {},
        "pass": True,
        "known_limitations": [
            "DMRG/Lanczos are NON-differentiable; not part of the autograd energy path.",
            "Stage 4A dense H_eff is preserved as the reference; Stage 4B adds a matrix-free apply + Lanczos for larger D.",
            "Lanczos uses simple full-reorthogonalization; robust for small Krylov sizes here, not optimized for very large D.",
            "N=10/12 smoke checks finiteness/energy-decrease/bond-caps/runtime only (no dense ED).",
            "No TEBD / TDVP / finite-T / GPU performance benchmark.",
        ],
    }

    # ---- matrix-free apply vs dense ----
    tc.manual_seed(0)
    Nm = 4
    mps = MPS(Nm, 2, 4, dtype=dtype, device=device)
    mpo = MPO.from_bonds(Nm, 2, dtype=dtype, device=device).generate_heisenberg(J=1.0)
    t = D.mixed_canonical_two_site(mps, 1)
    tmp = MPS.from_tensors(t, dtype=dtype, device=device)
    Hd = D.effective_hamiltonian(tmp, mpo, 1)
    mf = D.matrix_free_apply(tmp, mpo, 1)
    g = tc.Generator().manual_seed(5)
    x = tc.randn(mf.dim, dtype=dtype, device=device, generator=g) \
        + 1j * tc.randn(mf.dim, dtype=dtype, device=device, generator=g)
    mf_diff = float(tc.linalg.norm(mf(x) - Hd @ x))
    report["matrix_free_vs_dense"] = {
        "N": Nm, "bond": 1, "D": mf.dim, "apply_diff_norm": mf_diff,
        "pass": mf_diff < 1e-8,
    }

    # ---- Lanczos vs dense eigensolver ----
    E_lanczos, _ = LZ.lanczos_lowest_eigenpair(
        mf, mf.dim, dtype=dtype, device=device, max_iter=30, tol=1e-12, seed=0)
    E_dense_vec = tc.linalg.eigvalsh(Hd)
    lz_v_dense = abs(float(E_lanczos.real) - float(E_dense_vec[0].real))
    report["lanczos_vs_dense"] = {
        "lanczos_E0": float(E_lanczos.real),
        "dense_E0": float(E_dense_vec[0].real),
        "abs_diff": lz_v_dense,
        "pass": lz_v_dense < 1e-9,
    }

    # ---- small-system exact compare (both solvers) ----
    for solver in ("dense", "lanczos"):
        for N in (4, 6):
            res = run_single(N, chi=8, sweeps=4, solver=solver, seed=0,
                             dtype=dtype, device=device)
            ok = (res["exact_energy"] is not None
                  and abs(res["final_energy"] - res["exact_energy"]) < 1e-6
                  and not res["below_ground"])
            res["pass"] = bool(ok)
            report["exact_compare"].append(res)

    # ---- chi sweep (N=6) ----
    for chi in (4, 8, 16):
        res = run_single(6, chi=chi, sweeps=4, solver="dense", seed=0,
                         dtype=dtype, device=device)
        res["pass"] = bool(res["exact_energy"] is not None
                           and res["final_energy"] >= res["exact_energy"] - 1e-6)
        report["chi_sweep"].append(res)
    # energy non-increasing as chi grows (within tolerance)
    cs_energies = [c["final_energy"] for c in report["chi_sweep"]]
    chi_monotonic = all(b <= a + 1e-6 for a, b in zip(cs_energies, cs_energies[1:]))

    # ---- N=10 / N=12 smoke (no dense ED) ----
    for N, chi, sweeps in [(10, 16, 3), (12, 16, 3)]:
        tc.manual_seed(0)
        mps = MPS(N, 2, 8, dtype=dtype, device=device)
        mpo = MPO.from_bonds(N, 2, dtype=dtype, device=device).generate_heisenberg(J=1.0)
        e_init = float(K.rayleigh_energy_native(mps, mpo))
        t0 = time.perf_counter()
        res = D.run_dmrg(mps, mpo, chi=chi, num_sweeps=sweeps, solver="lanczos", seed=0)
        res["runtime_s"] = time.perf_counter() - t0
        res["initial_energy"] = e_init
        res["pass"] = bool(
            tc.isfinite(tc.as_tensor(res["final_energy"]))
            and res["final_energy"] < e_init + 1e-6
            and res["final_max_bond"] <= chi
            and res["runtime_s"] < 240.0
        )
        report["smoke"].append(res)

    checks = {
        "matrix_free_matches_dense": report["matrix_free_vs_dense"]["pass"],
        "lanczos_matches_dense_eigensolver": report["lanczos_vs_dense"]["pass"],
        "exact_compare_both_solvers": all(c["pass"] for c in report["exact_compare"]),
        "chi_sweep_not_worsening": bool(chi_monotonic),
        "smoke_n10_n12": all(s["pass"] for s in report["smoke"]),
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def _fmt_history(res) -> str:
    rows = []
    for h in res["history"]:
        rows.append(
            f"| {h['sweep']} | {h['direction']} | {h['solver']} | "
            f"{h['energy']:.10f} | {h['max_trunc']:.2e} | {h['max_bond']} |"
        )
    return "\n".join(rows)


def render_report_md(report: dict) -> str:
    L: list[str] = []
    L.append("# Stage 4B Scalable DMRG Benchmark Report")
    L.append("")
    L.append("Generated by `scripts/run_dmrg_benchmark.py`.")
    L.append("")
    L.append("## Convention")
    L.append("")
    L.append(f"`{report['convention']}`")
    L.append(f"- dtype: `{report['dtype']}`, device: `{report['device']}`")
    L.append(f"- Bethe thermodynamic E/bond reference: "
             f"`1/4 - ln(2) ~= {report['bethe_reference']:.16f}` "
             "(trend only; not a finite-open-chain target)")
    L.append("")
    L.append("## Matrix-free vs dense H_eff apply")
    L.append("")
    m = report["matrix_free_vs_dense"]
    L.append(f"- N={m['N']}, bond {m['bond']}, effective dim D={m['D']}")
    L.append(f"- ||matrix-free(x) - dense @ x|| = `{m['apply_diff_norm']:.3e}`")
    L.append(f"- pass: {'PASS' if m['pass'] else 'FAIL'}")
    L.append("")
    L.append("## Lanczos vs dense eigensolver")
    L.append("")
    z = report["lanczos_vs_dense"]
    L.append(f"- Lanczos lowest E0: `{z['lanczos_E0']:.12f}`")
    L.append(f"- dense (eigh) E0:   `{z['dense_E0']:.12f}`")
    L.append(f"- |Lanczos - dense|: `{z['abs_diff']:.3e}`")
    L.append(f"- pass: {'PASS' if z['pass'] else 'FAIL'}")
    L.append("")
    L.append("## Small-system exact comparison (N<=6)")
    L.append("")
    L.append("| N | chi | sweeps | solver | exact E0 | DMRG final E | abs err | energy/bond | below ground | max bond | runtime_s | pass |")
    L.append("|---:|---:|---:|---|---:|---:|---:|---:|:---:|---:|---:|:---|")
    for c in report["exact_compare"]:
        L.append(f"| {c['N']} | {c['chi']} | {c['num_sweeps']} | {c['solver']} | "
                 f"{c['exact_energy']:.10f} | {c['final_energy']:.10f} | "
                 f"{abs(c['final_energy'] - c['exact_energy']):.2e} | "
                 f"{c['energy_per_bond']:.10f} | {c['below_ground']} | "
                 f"{c['final_max_bond']} | {c['runtime_s']:.2f} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append("### Energy history (N=6, both solvers)")
    L.append("")
    for c in report["exact_compare"]:
        if c["N"] != 6:
            continue
        L.append(f"N=6, chi=8, solver={c['solver']}:")
        L.append("")
        L.append("| sweep | direction | solver | energy | max trunc | max bond |")
        L.append("|---:|---|---|---:|---:|---:|")
        L.append(_fmt_history(c))
        L.append("")
    L.append("## chi sweep (N=6, dense solver)")
    L.append("")
    L.append("| chi | final E | energy/bond | exact E0 | below ground | max bond | max trunc | runtime_s | pass |")
    L.append("|---:|---:|---:|---:|:---:|---:|---:|---:|:---|")
    for c in report["chi_sweep"]:
        L.append(f"| {c['chi']} | {c['final_energy']:.10f} | "
                 f"{c['energy_per_bond']:.10f} | {c['exact_energy']:.10f} | "
                 f"{c['below_ground']} | {c['final_max_bond']} | "
                 f"{max(h['max_trunc'] for h in c['history']):.2e} | "
                 f"{c['runtime_s']:.2f} | {'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append("Energy is non-increasing as chi grows (within tolerance): "
             f"`{report['checks']['chi_sweep_not_worsening']}`.")
    L.append("")
    L.append("## N=10 / N=12 smoke (no dense ED)")
    L.append("")
    L.append("| N | chi | sweeps | solver | initial E | final E | energy/bond | max bond | runtime_s | pass |")
    L.append("|---:|---:|---:|---|---:|---:|---:|---:|---:|:---|")
    for s in report["smoke"]:
        L.append(f"| {s['N']} | {s['chi']} | {s['num_sweeps']} | {s['solver']} | "
                 f"{s['initial_energy']:.6f} | {s['final_energy']:.8f} | "
                 f"{s['energy_per_bond']:.8f} | {s['final_max_bond']} | "
                 f"{s['runtime_s']:.2f} | {'PASS' if s['pass'] else 'FAIL'} |")
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
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", choices=["dense", "lanczos"], default="dense")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="complex128")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "DMRG_BENCHMARK_REPORT.md")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        print("Refusing to use CUDA in benchmark; forcing cpu.", file=sys.stderr)
        args.device = "cpu"
    dtype = DTYPE_MAP[args.dtype]

    if args.json_output or args.print:
        # single-case mode
        res = run_single(args.N, args.chi, args.sweeps, args.solver,
                         args.seed, dtype, args.device)
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(res, indent=2, default=str),
                                        encoding="utf-8")
        if args.print or not args.json_output:
            print(json.dumps(res, indent=2, default=str))
        return 0

    report = run_smoke(dtype=dtype, device=args.device)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_report_md(report), encoding="utf-8")
    print(f"dmrg benchmark: pass={report['pass']} report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
