#!/usr/bin/env python3
"""CPU/GPU Heisenberg Neel-quench benchmark for traditional two-site TDVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latticetn import canonical, contractions  # noqa: E402
from latticetn.initial_states import neel_spin_state  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.operators import heisenberg_dense, spin_operators  # noqa: E402
from latticetn.tdvp import TDVP  # noqa: E402


def run_quench(
    *,
    n: int = 8,
    dt: float = 0.02,
    steps: int = 10,
    max_bond_dim: int = 16,
    truncation_tol: float = 1e-10,
    device: str = "cpu",
) -> dict:
    """Run a Neel quench and return serializable conservation/physics data."""
    dtype = tc.complex128
    initial = neel_spin_state(n, dtype=dtype, device=device)
    mpo = MPO.from_bonds(n, 2, dtype=dtype, device=device).generate_heisenberg()
    sz = spin_operators(dtype=dtype, device=device)["Sz"]
    result = TDVP(
        initial,
        mpo,
        dt=dt,
        method="two_site",
        device=device,
        max_bond_dim=max_bond_dim,
        truncation_tol=truncation_tol,
    ).evolve(
        steps=steps,
        observables={
            "sz_mid": lambda state: contractions.native_local_expect(state, sz, n // 2)
            / contractions.native_norm_sq(state),
            "entropy_mid": lambda state: canonical.entanglement_entropy(state, n // 2),
        },
    )

    fidelity = None
    if n <= 12:
        dense_h = heisenberg_dense(n, dtype=dtype, device=device)
        exact = tc.matrix_exp(-1j * result.times[-1] * dense_h) @ initial.to_dense()
        evolved = result.mps.to_dense()
        evolved = evolved / tc.linalg.vector_norm(evolved)
        fidelity = float(abs(tc.vdot(exact, evolved)) ** 2)

    rows = []
    for index, time in enumerate(result.times):
        truncation = result.truncation_history[index - 1] if index else None
        rows.append({
            "step": index,
            "time": time,
            "norm": result.norm_history[index],
            "energy": result.energy_history[index],
            "sz_mid": result.observables_history["sz_mid"][index],
            "entropy_mid": result.observables_history["entropy_mid"][index],
            "max_bond": truncation["max_bond"] if truncation else 1,
            "max_truncation": truncation["max_truncation"] if truncation else 0.0,
        })

    return {
        "configuration": {
            "N": n,
            "dt": dt,
            "steps": steps,
            "max_bond_dim": max_bond_dim,
            "truncation_tol": truncation_tol,
            "device": device,
            "dtype": str(dtype),
            "model": "open Heisenberg, H=sum_i S_i.S_{i+1}, S=sigma/2",
            "initial_state": "Neel |up down up down ...>",
        },
        "norm_drift": max(abs(value - result.norm_history[0]) for value in result.norm_history),
        "energy_drift": max(abs(value - result.energy_history[0]) for value in result.energy_history),
        "final_fidelity_vs_ed": fidelity,
        "history": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--chi-max", type=int, default=16)
    parser.add_argument("--truncation-tol", type=float, default=1e-10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = run_quench(
        n=args.N,
        dt=args.dt,
        steps=args.steps,
        max_bond_dim=args.chi_max,
        truncation_tol=args.truncation_tol,
        device=args.device,
    )

    print(json.dumps(report["configuration"], indent=2))
    print("step time norm energy Sz(mid) entropy(mid) max_chi max_trunc")
    for row in report["history"]:
        print(
            f"{row['step']:4d} {row['time']:.6f} {row['norm']:.12f} "
            f"{row['energy']:.12f} {row['sz_mid']:.12f} "
            f"{row['entropy_mid']:.12f} {row['max_bond']:7d} "
            f"{row['max_truncation']:.3e}"
        )
    print(f"norm drift: {report['norm_drift']:.3e}")
    print(f"energy drift: {report['energy_drift']:.3e}")
    if report["final_fidelity_vs_ed"] is not None:
        print(f"final fidelity vs ED: {report['final_fidelity_vs_ed']:.12f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
