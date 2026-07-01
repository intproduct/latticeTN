#!/usr/bin/env python3
"""Stage 3B native-contraction smoke runner.

Demonstrates the native (no-to_dense) tensor-network contractions on a small
system against dense references, plus a scalability smoke (N=20, chi<=8), and
writes ``docs/CONTRACTION_REPORT.md``.

Reported content (per docs/CONTRACTION_PROTOCOL.md stop conditions):
- native/dense comparison table (norm, <Sz_i>, <Sz_i Sz_j>, bond energy)
- MPO energy comparison (native vs Stage1 energy path vs dense)
- gradient check (native energy backward -> all params grad not None)
- scalability smoke (finite, shape, device/dtype, runtime)
- pass/fail, known limitations
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn import contractions as K  # noqa: E402
from latticetn import observables as O  # noqa: E402
from latticetn.operators import spin_operators, heisenberg_dense  # noqa: E402

DTYPE = tc.complex128


def _bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def run_smoke(N=6, chi=4, seed=0, N_big=20, chi_big=8) -> dict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    report: dict = {
        "convention": "H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary, complex128",
        "dtype": str(DTYPE),
        "compare": {},
        "mpo_energy": {},
        "gradient": {},
        "scalability": {},
        "checks": {},
        "pass": True,
        "known_limitations": [
            "Native contractions scale polynomially in N and chi; to_dense is NOT used.",
            "The differentiable native energy path uses no .detach()/.data/unnecessary .item(); observable/report paths may use torch.no_grad().",
            "Scalability smoke only checks finiteness/shape/device/dtype at N=20 (no dense reference or ED at that size).",
            "No DMRG / TEBD / GPU performance benchmark in this stage.",
        ],
    }

    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    ops = spin_operators(dtype=DTYPE)
    psi = mps.to_dense().detach()

    with tc.no_grad():
        # norm
        norm_dense = float((psi.conj() @ psi).real)
        norm_native = float(K.native_norm_sq(mps).real)
        # local <Sz> : mid site
        i_mid = N // 2
        sz_d = float(O.dense_expect_local(psi, ops["Sz"], i_mid, N).real)
        sz_n = float(K.native_local_expect(mps, ops["Sz"], i_mid).real)
        # two-site <Sz_i Sz_j>
        i2, j2 = 1, N - 2
        c2_d = complex(O.dense_expect_two_site(psi, ops["Sz"], i2, ops["Sz"], j2, N)).real
        c2_n = float(K.native_two_site_expect(mps, ops["Sz"], i2, ops["Sz"], j2).real)
        # bond energy at middle bond
        b = i_mid - 1
        be_d = float(O.dense_bond_energy_heisenberg(psi, b, N))
        be_n = float(K.native_bond_energy_heisenberg(mps, b))

    report["compare"] = {
        "norm_dense": norm_dense, "norm_native": norm_native,
        "norm_diff": abs(norm_dense - norm_native),
        "Sz_i_dense": sz_d, "Sz_i_native": sz_n, "Sz_i_diff": abs(sz_d - sz_n),
        "Sz_iSz_j_dense": c2_d, "Sz_iSz_j_native": c2_n,
        "Sz_iSz_j_diff": abs(c2_d - c2_n),
        "bond_energy_dense": be_d, "bond_energy_native": be_n,
        "bond_energy_diff": abs(be_d - be_n),
        "pass": (abs(norm_dense - norm_native) < 1e-9
                 and abs(sz_d - sz_n) < 1e-9
                 and abs(c2_d - c2_n) < 1e-9
                 and abs(be_d - be_n) < 1e-9),
    }

    # MPO energy comparison
    e_classic = float(mps.energy_with_MPO(mpo))
    e_native = float(K.rayleigh_energy_native(mps, mpo))
    H = heisenberg_dense(N, dtype=DTYPE)
    e_dense = complex(psi.conj() @ H @ psi).real / complex(psi.conj() @ psi).real
    report["mpo_energy"] = {
        "stage1_energy_with_MPO": e_classic,
        "native_rayleigh": e_native,
        "dense_state_energy": e_dense,
        "native_vs_stage1_diff": abs(e_classic - e_native),
        "native_vs_dense_diff": abs(e_native - e_dense),
        "pass": abs(e_classic - e_native) < 1e-9 and abs(e_native - e_dense) < 1e-9,
    }

    # gradient check (differentiable native energy path)
    gmps = MPS(N, 2, chi, dtype=DTYPE)  # fresh for clean grad
    gmpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    tc.manual_seed(seed + 100)
    gmps2 = MPS(N, 2, chi, dtype=DTYPE)
    e = K.rayleigh_energy_native(gmps2, gmpo)
    e.backward()
    grad_ok = all(p.grad is not None for p in gmps2.tensors)
    grad_finite = all(tc.isfinite(p.grad).all() if p.grad is not None else False
                      for p in gmps2.tensors)
    report["gradient"] = {
        "all_grads_not_none": bool(grad_ok),
        "all_grads_finite": bool(grad_finite),
        "energy_requires_grad": bool(e.requires_grad),
        "pass": bool(grad_ok and grad_finite),
    }

    # scalability smoke (no to_dense, no ED)
    tc.manual_seed(seed)
    mps_big = MPS(N_big, 2, chi_big, dtype=DTYPE)
    mpo_big = MPO.from_bonds(N_big, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    t0 = time.perf_counter()
    e_big = float(K.rayleigh_energy_native(mps_big, mpo_big))
    t_energy = time.perf_counter() - t0
    with tc.no_grad():
        loc_big = float(K.native_local_expect(mps_big, ops["Sz"], N_big // 2).real)
        corr_big = float(K.native_correlation(mps_big, ops["Sz"], 3, 11).real)
        bond_dims = _bond_dims(mps_big)
    finite = all(x == x and x != float("inf")
                 for x in [e_big, loc_big, corr_big])
    report["scalability"] = {
        "N": N_big, "chi": chi_big,
        "bond_dims": bond_dims, "max_bond": max(bond_dims),
        "energy_finite": bool(finite),
        "energy": e_big, "Sz_mid": loc_big, "corr_3_11": corr_big,
        "energy_runtime_s": t_energy,
        "device": str(mps_big.tensors[0].device),
        "dtype": str(mps_big.tensors[0].dtype),
        "pass": (bool(finite) and max(bond_dims) <= chi_big
                 and mps_big.tensors[0].device.type == "cpu"
                 and mps_big.tensors[0].dtype == DTYPE
                 and t_energy < 5.0),
    }

    checks = {
        "native_dense_observables_match": report["compare"]["pass"],
        "mpo_energy_native_matches_dense_and_stage1": report["mpo_energy"]["pass"],
        "gradient_check": report["gradient"]["pass"],
        "scalability_smoke": report["scalability"]["pass"],
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def render_report_md(report: dict) -> str:
    L: list[str] = []
    L.append("# Stage 3B Native Contraction Report")
    L.append("")
    L.append("Generated by `scripts/run_contraction_smoke.py`.")
    L.append("")
    L.append("## Convention")
    L.append("")
    L.append(f"`{report['convention']}`")
    L.append(f"- dtype: `{report['dtype']}`")
    L.append("")
    L.append("## Native vs dense reference (small system)")
    L.append("")
    C = report["compare"]
    L.append("| quantity | dense reference | native contraction | abs diff | pass |")
    L.append("|---|---:|---:|---:|:---|")
    L.append(f"| <psi|psi> (norm^2) | {C['norm_dense']:.10g} | "
             f"{C['norm_native']:.10g} | {C['norm_diff']:.2e} | "
             f"{'PASS' if C['pass'] else 'FAIL'} |")
    L.append(f"| <Sz_i> (mid site) | {C['Sz_i_dense']:.10g} | "
             f"{C['Sz_i_native']:.10g} | {C['Sz_i_diff']:.2e} | "
             f"{'PASS' if C['pass'] else 'FAIL'} |")
    L.append(f"| <Sz_i Sz_j> | {C['Sz_iSz_j_dense']:.10g} | "
             f"{C['Sz_iSz_j_native']:.10g} | {C['Sz_iSz_j_diff']:.2e} | "
             f"{'PASS' if C['pass'] else 'FAIL'} |")
    L.append(f"| bond energy <S_i.S_(i+1)> | {C['bond_energy_dense']:.10g} | "
             f"{C['bond_energy_native']:.10g} | {C['bond_energy_diff']:.2e} | "
             f"{'PASS' if C['pass'] else 'FAIL'} |")
    L.append("")
    L.append("## MPO energy comparison")
    L.append("")
    M = report["mpo_energy"]
    L.append("| path | energy |")
    L.append("|---|---:|")
    L.append(f"| Stage 1 `MPS.energy_with_MPO` | {M['stage1_energy_with_MPO']:.12g} |")
    L.append(f"| native `rayleigh_energy_native` | {M['native_rayleigh']:.12g} |")
    L.append(f"| dense-state `<psi|H|psi>/<psi|psi>` | {M['dense_state_energy']:.12g} |")
    L.append(f"| |native - stage1| | {M['native_vs_stage1_diff']:.2e} |")
    L.append(f"| |native - dense| | {M['native_vs_dense_diff']:.2e} |")
    L.append(f"| pass | {'PASS' if M['pass'] else 'FAIL'} |")
    L.append("")
    L.append("## Gradient check (differentiable native energy path)")
    L.append("")
    G = report["gradient"]
    L.append(f"- all MPS params grad not None: `{G['all_grads_not_none']}`")
    L.append(f"- all grads finite: `{G['all_grads_finite']}`")
    L.append(f"- energy `requires_grad`: `{G['energy_requires_grad']}`")
    L.append(f"- pass: {'PASS' if G['pass'] else 'FAIL'}")
    L.append("")
    L.append("## Scalability smoke (no `to_dense`, no exact diagonalization)")
    L.append("")
    S = report["scalability"]
    L.append(f"- N: `{S['N']}`, chi cap: `{S['chi']}`")
    L.append(f"- bond dims: `{S['bond_dims']}`, max bond: `{S['max_bond']}`")
    L.append(f"- energy (native Rayleigh): `{S['energy']:.6g}` "
             f"(finite: `{S['energy_finite']}`)")
    L.append(f"- <Sz_mid>: `{S['Sz_mid']:.6g}`, <Sz_3 Sz_11>: `{S['corr_3_11']:.6g}`")
    L.append(f"- energy runtime: `{S['energy_runtime_s']:.3f}s`")
    L.append(f"- device: `{S['device']}`, dtype: `{S['dtype']}`")
    L.append(f"- pass: {'PASS' if S['pass'] else 'FAIL'}")
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
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "CONTRACTION_REPORT.md")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    report = run_smoke()
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_report_md(report), encoding="utf-8")
    print(f"contraction smoke: pass={report['pass']} "
          f"report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
