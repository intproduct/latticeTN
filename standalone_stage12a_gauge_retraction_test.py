"""Standalone Stage 12A evidence: Global AD with exact gauge retractions.

This is deliberately independent of the job runner.  It compares pure Global
AD, periodic no-grad QR retraction, and periodic exact non-truncating SVD
retraction on a small dense TFI Hamiltonian with dense ED reference.
"""

from __future__ import annotations

import argparse
import json

import torch as tc

from latticetn.canonical import canonical_residual, left_canonicalize, normalize_center
from latticetn.mps import MPS
from latticetn.operators import tfi_dense


def _rayleigh(mps: MPS, h: tc.Tensor) -> tc.Tensor:
    psi = mps.to_dense()
    return (tc.vdot(psi, h @ psi) / tc.vdot(psi, psi)).real


def _copy_(target: MPS, source: MPS) -> None:
    with tc.no_grad():
        for dst, src in zip(target.tensors, source.tensors):
            if dst.shape != src.shape:
                raise RuntimeError("exact retraction unexpectedly changed a tensor shape")
            dst.copy_(src)


def _grad_norm(mps: MPS) -> float:
    total = tc.zeros((), dtype=tc.float64)
    for p in mps.parameters():
        if p.grad is not None:
            total += (p.grad.conj() * p.grad).real.sum().cpu()
    return float(total.sqrt())


def run_case(mode: str, h: tc.Tensor, ed_state: tc.Tensor, *, N: int, chi: int,
             steps: int, lr: float, interval: int, seed: int) -> dict:
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=tc.complex128, device="cpu")
    opt = tc.optim.Adam(mps.parameters(), lr=lr)
    last_grad = 0.0
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = _rayleigh(mps, h)
        loss.backward()
        last_grad = _grad_norm(mps)
        opt.step()
        if mode in {"qr", "svd"} and step % interval == 0:
            retracted = left_canonicalize(mps, method=mode)
            retracted = normalize_center(retracted, center=N - 1)
            _copy_(mps, retracted)
            # Stage 12A does not transport Adam moments through gauge changes.
            opt = tc.optim.Adam(mps.parameters(), lr=lr)

    psi = mps.to_dense().detach()
    physical_norm = float(psi.norm())
    unit = psi / psi.norm()
    overlap_complex = tc.vdot(ed_state, unit)
    overlap = float(overlap_complex.abs())
    phase = overlap_complex / overlap_complex.abs() if overlap > 0 else tc.ones((), dtype=unit.dtype)
    distance = float(tc.linalg.vector_norm(unit - phase * ed_state))
    return {
        "mode": mode,
        "energy": float(_rayleigh(mps, h).detach()),
        "physical_norm": physical_norm,
        "canonical_residual": canonical_residual(mps, center=N - 1),
        "gradient_norm": last_grad,
        "ed_overlap": overlap,
        "ed_state_distance": distance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1200)
    args = parser.parse_args()
    h = tfi_dense(args.N, J=1.0, h=0.8, dtype=tc.complex128, device="cpu")
    evals, evecs = tc.linalg.eigh(h)
    ed_state = evecs[:, 0]
    results = [
        run_case(mode, h, ed_state, N=args.N, chi=args.chi, steps=args.steps,
                 lr=args.lr, interval=args.interval, seed=args.seed)
        for mode in ("pure", "qr", "svd")
    ]
    print(json.dumps({"exact_energy": float(evals[0]), "results": results}, indent=2))


if __name__ == "__main__":
    main()
