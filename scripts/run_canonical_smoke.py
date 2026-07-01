#!/usr/bin/env python3
"""Stage 3A canonicalization smoke runner.

Performs a small, CPU-only demonstration of the canonical-form capabilities and
writes ``docs/CANONICALIZATION_REPORT.md``. Not a long optimization; not a
performance benchmark.

Reported content (per docs/CANONICALIZATION_PROTOCOL.md stop conditions):
- left/right/mixed canonical test results (orthonormality + fidelity)
- compression test results (bond dims, fidelity, energy error)
- entropy comparison (canonical vs dense SVD reference)
- energy/fidelity errors
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

from latticetn.mpo import MPO  # noqa: E402
from latticetn.mps import MPS  # noqa: E402
from latticetn import canonical as C  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402
from latticetn.observables import dense_entanglement_entropy  # noqa: E402

DTYPE = tc.complex128


def _fidelity(mps_a: MPS, mps_b: MPS) -> float:
    a = mps_a.to_dense()
    b = mps_b.to_dense()
    a = a / tc.linalg.norm(a)
    b = b / tc.linalg.norm(b)
    return abs(tc.vdot(b, a)).item()


def _bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def run_smoke(N_rand: int = 5, chi_rand: int = 4, N_gs: int = 6) -> dict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    tc.manual_seed(0)
    report: dict = {
        "convention": "H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary, complex128",
        "dtype": str(DTYPE),
        "left": {},
        "right": {},
        "mixed": [],
        "compression": {},
        "entropy": [],
        "pass": True,
        "checks": {},
        "known_limitations": [
            "Canonicalization/compression is non-differentiable preprocessing/postprocessing; not part of the training energy path.",
            "QR/LQ sweeps are exact (rank-preserving); SVD compression reduces bond dimension and reports discarded weight.",
            "MPS-from-dense uses successive SVDs; suitable only for small systems (exponential dense state).",
            "No DMRG / TEBD / performance benchmarking in this stage.",
        ],
    }

    # ---- random MPS canonicalization ------------------------------------
    mps = MPS(N_rand, 2, chi_rand, dtype=DTYPE)
    dense_norm = float(tc.linalg.norm(mps.to_dense()))

    L = C.left_canonical(mps)
    report["left"] = {
        "orthonormality_error": C.left_orthonormal_all(L),
        "fidelity": _fidelity(mps, L),
        "norm": C.canonical_norm(L),
        "dense_norm": dense_norm,
        "pass": (C.left_orthonormal_all(L) < 1e-10
                 and _fidelity(mps, L) > 1.0 - 1e-10
                 and abs(C.canonical_norm(L) - dense_norm) < 1e-9),
    }

    R = C.right_canonical(mps)
    report["right"] = {
        "orthonormality_error": C.right_orthonormal_all(R),
        "fidelity": _fidelity(mps, R),
        "norm": C.canonical_norm(R),
        "pass": (C.right_orthonormal_all(R) < 1e-10
                 and _fidelity(mps, R) > 1.0 - 1e-10),
    }

    for center in range(1, N_rand - 1):
        M = C.mixed_canonical(mps, center)
        report["mixed"].append({
            "center": center,
            "left_orth_error": C.left_orthonormal_all(M, upto=center),
            "right_orth_error": C.right_orthonormal_all(M, from_=center),
            "fidelity": _fidelity(mps, M),
            "center_frob_norm": C.center_frob_norm(M, center),
            "pass": (C.left_orthonormal_all(M, upto=center) < 1e-10
                     and C.right_orthonormal_all(M, from_=center) < 1e-10
                     and _fidelity(mps, M) > 1.0 - 1e-10
                     and abs(C.center_frob_norm(M, center) - dense_norm) < 1e-9),
        })

    # ---- entropy comparison vs dense SVD reference -----------------------
    psi_n = mps.to_dense()
    psi_n = psi_n / tc.linalg.norm(psi_n)
    for cut in range(1, N_rand):
        dS = float(dense_entanglement_entropy(psi_n, cut, N_rand))
        cS = float(C.entanglement_entropy(mps, cut))
        report["entropy"].append({
            "cut": cut,
            "dense_S": dS,
            "canonical_S": cS,
            "abs_diff": abs(dS - cS),
            "pass": abs(dS - cS) < 1e-9,
        })

    # ---- compression on the exact Heisenberg ground state ---------------
    H = heisenberg_dense(N_gs, dtype=DTYPE)
    E0, gs = exact_ground_energy(H)
    mpo = MPO.from_bonds(N_gs, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    mps_full = C.from_dense(gs, N_gs, chi=None)
    e_full = float(mps_full.energy_with_MPO(mpo))

    comp_full, info_full = C.svd_compress(mps_full, chi=8)
    e_comp_full = float(comp_full.energy_with_MPO(mpo))

    comp_trunc, info_trunc = C.svd_compress(mps_full, chi=2)
    e_comp_trunc = float(comp_trunc.energy_with_MPO(mpo))

    report["compression"] = {
        "N": N_gs,
        "exact_E0": E0,
        "full_mps_energy": e_full,
        "full_mps_energy_err": abs(e_full - E0),
        "chi8_bond_dims": _bond_dims(comp_full),
        "chi8_max_bond": info_full["max_bond_dim"],
        "chi8_energy": e_comp_full,
        "chi8_energy_err": abs(e_comp_full - E0),
        "chi8_truncation_total": info_full["total_truncation"],
        "chi8_pass": (info_full["max_bond_dim"] <= 8
                      and abs(e_comp_full - E0) < 1e-9
                      and e_comp_full >= E0 - 1e-6),
        "chi2_bond_dims": _bond_dims(comp_trunc),
        "chi2_max_bond": info_trunc["max_bond_dim"],
        "chi2_energy": e_comp_trunc,
        "chi2_energy_err": abs(e_comp_trunc - E0),
        "chi2_truncation_total": info_trunc["total_truncation"],
        "chi2_pass": (info_trunc["max_bond_dim"] <= 2
                      and e_comp_trunc >= E0 - 1e-6
                      and abs(e_comp_trunc - E0) < 1.0),
    }

    checks = {
        "left_canonical": report["left"]["pass"],
        "right_canonical": report["right"]["pass"],
        "mixed_canonical_all_centers": all(m_["pass"] for m_ in report["mixed"]),
        "entropy_matches_dense": all(e_["pass"] for e_ in report["entropy"]),
        "compression_chi8_exact": report["compression"]["chi8_pass"],
        "compression_chi2_physical": report["compression"]["chi2_pass"],
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def render_report_md(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Stage 3A Canonicalization Report")
    lines.append("")
    lines.append("Generated by `scripts/run_canonical_smoke.py`.")
    lines.append("")
    lines.append("## Convention")
    lines.append("")
    lines.append(f"`{report['convention']}`")
    lines.append(f"- dtype: `{report['dtype']}`")
    lines.append("")
    lines.append("## Left / Right canonical")
    lines.append("")
    L = report["left"]
    R = report["right"]
    lines.append("| form | orthonormality error | fidelity | norm | dense norm | pass |")
    lines.append("|---|---:|---:|---:|---:|:---|")
    lines.append(f"| left | {L['orthonormality_error']:.2e} | "
                 f"{L['fidelity']:.12f} | {L['norm']:.10f} | "
                 f"{L['dense_norm']:.10f} | {'PASS' if L['pass'] else 'FAIL'} |")
    lines.append(f"| right | {R['orthonormality_error']:.2e} | "
                 f"{R['fidelity']:.12f} | {R['norm']:.10f} | "
                 f"{L['dense_norm']:.10f} | {'PASS' if R['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append("## Mixed canonical")
    lines.append("")
    lines.append("| center | left-orth err (<center) | right-orth err (>center) | fidelity | center Frob norm | pass |")
    lines.append("|---:|---:|---:|---:|---:|:---|")
    for m in report["mixed"]:
        lines.append(f"| {m['center']} | {m['left_orth_error']:.2e} | "
                     f"{m['right_orth_error']:.2e} | {m['fidelity']:.12f} | "
                     f"{m['center_frob_norm']:.10f} | "
                     f"{'PASS' if m['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append("## Entanglement entropy: canonical vs dense SVD reference")
    lines.append("")
    lines.append("| cut | dense S | canonical S | abs diff | pass |")
    lines.append("|---:|---:|---:|---:|:---|")
    for e in report["entropy"]:
        lines.append(f"| {e['cut']} | {e['dense_S']:.10f} | "
                     f"{e['canonical_S']:.10f} | {e['abs_diff']:.2e} | "
                     f"{'PASS' if e['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append("## Compression (exact Heisenberg ground state)")
    lines.append("")
    c = report["compression"]
    lines.append(f"- N: `{c['N']}`, exact E0: `{c['exact_E0']:.12f}`")
    lines.append(f"- full-MPS energy: `{c['full_mps_energy']:.12f}` "
                 f"(err {c['full_mps_energy_err']:.2e})")
    lines.append("")
    lines.append("| chi | bond dims | max bond | energy | energy err | total trunc | pass |")
    lines.append("|---:|---|---:|---:|---:|---:|:---|")
    lines.append(f"| 8 | {c['chi8_bond_dims']} | {c['chi8_max_bond']} | "
                 f"{c['chi8_energy']:.12f} | {c['chi8_energy_err']:.2e} | "
                 f"{c['chi8_truncation_total']:.2e} | "
                 f"{'PASS' if c['chi8_pass'] else 'FAIL'} |")
    lines.append(f"| 2 | {c['chi2_bond_dims']} | {c['chi2_max_bond']} | "
                 f"{c['chi2_energy']:.12f} | {c['chi2_energy_err']:.2e} | "
                 f"{c['chi2_truncation_total']:.2e} | "
                 f"{'PASS' if c['chi2_pass'] else 'FAIL'} |")
    lines.append("")
    lines.append("Energy/fidelity notes: chi=8 equals the full bond dimension "
                 "for N=6, so compression is exact (energy err ~machine eps, "
                 "fidelity 1). chi=2 is an aggressive truncation and must still "
                 "satisfy the variational bound (energy >= exact E0 - tol).")
    lines.append("")
    lines.append("## Overall pass/fail")
    lines.append("")
    lines.append(f"**pass: `{report['pass']}`**")
    lines.append("")
    lines.append("| check | passed |")
    lines.append("|---|:---|")
    for k, v in report["checks"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Known limitations")
    lines.append("")
    for lim in report["known_limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "CANONICALIZATION_REPORT.md")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    report = run_smoke()
    report["runtime_s"] = time.perf_counter() - start

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_report_md(report), encoding="utf-8")
    print(f"canonical smoke: pass={report['pass']} "
          f"runtime={report['runtime_s']:.3f}s "
          f"report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
